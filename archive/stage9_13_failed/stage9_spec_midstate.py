"""DSpark → RWKV · Speculative Decoding with Mid-State Preservation

核心优化：单条 T=K forward 时，在 tmix 的 for t 循环里保存中间 wkv_state
拒绝时直接取 slot n_accept 的 state，不用 rollback+replay（省 71.5ms）

方案：
1. target forward T=K 时，每步保存 [wkv_state.clone(), shift_state.clone()]
2. rejection sampling 后，根据 n_accept 直接取中间 state
3. 省掉 rollback+replay 的 forward

但中间 state 保存需要修改 tmix，这里用简化方案：
- target 用逐 token forward T=1（K 次），每次 forward 后 clone state
- 虽然比单条 T=K 慢 5x，但省掉 rollback+replay（71.5ms）
- 净效果：K*60ms vs 60ms + 71.5ms*reject_rate
"""
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from stage9_target_2p9b import RWKV7Target2p9B, DTYPE, DEVICE
from stage9_logit_fusion_01b import RWKV7Draft
from rwkv_tokenizer import TRIE_TOKENIZER

G1A_TARGET = Path(r"C:\work\niceui\g1a-2.9B.pth")
DRAFT_WEIGHTS = Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1a-0.4b-20250905-ctx4096.pth")


def softmax_cpu(logits):
    m = logits.max()
    e = torch.exp(logits - m)
    return e / e.sum()


def rejection_sample(candidate, small_probs, large_probs):
    idx = candidate
    q = small_probs[idx].item()
    p = large_probs[idx].item()
    large_argmax = large_probs.argmax().item()
    if large_argmax == candidate:
        return True, candidate
    if q <= 0:
        accept_prob = 0.0
    else:
        accept_prob = min(1.0, p / q)
    r = torch.rand(1).item()
    if r < accept_prob:
        return True, candidate
    else:
        residual = torch.clamp(large_probs - small_probs, min=0.0)
        s = residual.sum().item()
        if s <= 0:
            return False, large_argmax
        r2 = torch.rand(1).item() * s
        acc = 0.0
        for i in range(residual.shape[0]):
            acc += residual[i].item()
            if acc >= r2:
                return False, i
        return False, residual.shape[0] - 1


@torch.no_grad()
def speculative_decode_midstate(draft, target, tokenizer, prompt, n_generate=40, K=4, alpha=0.5):
    """投机解码：target 用逐 token forward + 保存中间 state

    优化点：
    1. target 验证时逐 token forward T=1，每次保存 state clone
    2. 拒绝时直接取 slot n_accept 的 state，不用 rollback+replay
    3. CUDA graph 加速 draft forward

    但 target 逐 token forward 比单条 T=K 慢 5x（344ms vs 68ms）
    只有当 rollback+replay 频率很高时才划算
    """
    ids = tokenizer.encode(prompt)
    out = list(ids)

    t_state = target.zero_state(1)
    d_state = draft.zero_state(1)

    # prompt forward
    ctx = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    t_logits_full, _ = target.forward(ctx, t_state, return_hidden_layers=[])
    d_logits_full = draft.forward(ctx, d_state)
    t_logits = t_logits_full[:, -1, :]  # [1, V]
    d_logits = d_logits_full[:, -1, :]

    stats = {"accepted": 0, "rejected": 0, "total_draft": 0,
             "target_forwards": 1, "draft_forwards": 1}

    while len(out) - len(ids) < n_generate:
        # === draft 生成 K 个候选 ===
        d_state_backup = [s.clone() for s in d_state]
        draft_tokens = []
        small_probs_list = []
        cur_d_logits = d_logits
        large_logits_for_fusion = t_logits.clone()

        for i in range(K):
            if i == 0:
                fused = cur_d_logits + alpha * large_logits_for_fusion
                probs = softmax_cpu(fused[0])
            else:
                probs = softmax_cpu(cur_d_logits[0])

            tok = probs.argmax().item()
            small_probs_list.append(probs)
            draft_tokens.append(tok)
            nt = torch.tensor([[tok]], device=DEVICE, dtype=torch.long)
            d_logits_full = draft.forward(nt, d_state)
            cur_d_logits = d_logits_full[:, -1, :]
            stats["draft_forwards"] += 1

        stats["total_draft"] += K

        # === target 验证：逐 token forward + 保存中间 state ===
        # slot 0: state=S0, forward [t1] → logits_1, state=S1
        # slot 1: state=S1, forward [t2] → logits_2, state=S2
        # ...
        # 每步保存 state，拒绝时直接取 slot n_accept 的 state

        # 保存初始 state（prompt 后的 state）
        t_state_snapshots = [[s.clone() for s in t_state]]  # t_state_snapshots[0] = S0
        verify_logits_list = [t_logits[0]]  # verify_logits_list[0] = prompt 后 logits（验证 t1）
        cur_t_logits = t_logits

        for i in range(K):
            nt = torch.tensor([[draft_tokens[i]]], device=DEVICE, dtype=torch.long)
            t_logits_full, _ = target.forward(nt, t_state, return_hidden_layers=[])
            cur_t_logits = t_logits_full[:, -1, :]
            verify_logits_list.append(cur_t_logits[0])
            t_state_snapshots.append([s.clone() for s in t_state])
            stats["target_forwards"] += 1

        # === rejection sampling ===
        n_accept = 0
        correction_token = None
        for i in range(K):
            large_probs = softmax_cpu(verify_logits_list[i])
            accept, resampled = rejection_sample(draft_tokens[i], small_probs_list[i], large_probs)
            if accept:
                n_accept += 1
            else:
                correction_token = resampled
                break

        stats["accepted"] += n_accept

        if n_accept == K:
            # 全部接受：state 已经是最终 state
            out.extend(draft_tokens)
            t_logits = cur_t_logits
            d_logits = d_logits_full[:, -1, :]
        else:
            # 中途拒绝：直接取 slot n_accept 的 state，不用 rollback+replay
            stats["rejected"] += 1
            out.extend(draft_tokens[:n_accept])
            out.append(correction_token)

            # 恢复到 slot n_accept 的 state
            t_state[0].copy_(t_state_snapshots[n_accept][0])
            t_state[1].copy_(t_state_snapshots[n_accept][1])

            # target forward correction_token 得到新 logits
            nt = torch.tensor([[correction_token]], device=DEVICE, dtype=torch.long)
            t_logits_full, _ = target.forward(nt, t_state, return_hidden_layers=[])
            t_logits = t_logits_full[:, -1, :]
            stats["target_forwards"] += 1

            # draft state rollback + replay [accepted + correction]
            d_state[0].copy_(d_state_backup[0])
            d_state[1].copy_(d_state_backup[1])
            replay_tokens = draft_tokens[:n_accept] + [correction_token]
            replay_tensor = torch.tensor([replay_tokens], device=DEVICE, dtype=torch.long)
            d_logits_full = draft.forward(replay_tensor, d_state)
            d_logits = d_logits_full[:, -1, :]
            stats["draft_forwards"] += 1

        if out[-1] == 0 or len(out) - len(ids) >= n_generate:
            break

    return out, stats


def benchmark_baseline(target, tokenizer, prompt, n_generate=40):
    ids = tokenizer.encode(prompt)
    out = list(ids)
    state = target.zero_state(1)
    ctx = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    logits, _ = target.forward(ctx, state, return_hidden_layers=[])
    forwards = 1
    while len(out) - len(ids) < n_generate:
        next_tok = logits[0, -1].argmax().item()
        out.append(next_tok)
        nt = torch.tensor([[next_tok]], device=DEVICE, dtype=torch.long)
        logits, _ = target.forward(nt, state, return_hidden_layers=[])
        forwards += 1
        if next_tok == 0:
            break
    return out, {"target_forwards": forwards}


def main():
    print("=" * 70)
    print("DSpark → RWKV · Speculative Decoding with Mid-State Preservation")
    print(f"  Target: g1a-2.9B  |  Draft: g1a-0.4B-20250905")
    print(f"  优化：target 逐 token forward + 保存中间 state，拒绝时不用 replay")
    print("=" * 70)

    tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))
    print("加载 target...", flush=True)
    target = RWKV7Target2p9B(G1A_TARGET)
    print("加载 draft...", flush=True)
    draft = RWKV7Draft(DRAFT_WEIGHTS)

    prompts = [
        "什么是人工智能",
        "中国的首都是哪里",
        "水的化学式是什么",
        "地球围绕太阳转吗",
        "一年有多少天",
        "太阳从哪个方向升起",
        "月亮是什么形状",
        "人类有几只手",
        "猫是动物吗",
        "苹果是什么颜色的",
    ]

    # 基线
    print(f"\n[基线: 纯 target 自回归, {len(prompts)} 条]", flush=True)
    t0 = time.time()
    total_base_fwd = 0
    for p in prompts:
        _, stats = benchmark_baseline(target, tokenizer, p, n_generate=40)
        total_base_fwd += stats["target_forwards"]
    base_time = (time.time() - t0) / len(prompts)
    print(f"  平均 {base_time:.2f}s/条, target forwards {total_base_fwd/len(prompts):.1f}", flush=True)

    # mid-state 版本
    K = 4
    alpha = 0.5
    print(f"\n[Mid-State Spec: K={K}, α={alpha}, {len(prompts)} 条]", flush=True)
    t0 = time.time()
    total_acc, total_draft, total_t_fwd = 0, 0, 0
    for p in prompts:
        out, stats = speculative_decode_midstate(draft, target, tokenizer, p, n_generate=40, K=K, alpha=alpha)
        total_acc += stats["accepted"]
        total_draft += stats["total_draft"]
        total_t_fwd += stats["target_forwards"]
    elapsed = time.time() - t0
    ar = total_acc / max(total_draft, 1)
    spec_time = elapsed / len(prompts)
    speedup = base_time / spec_time if spec_time > 0 else 0
    print(f"  接受率: {ar:.2%}", flush=True)
    print(f"  平均 {spec_time:.2f}s/条, target forwards {total_t_fwd/len(prompts):.1f}", flush=True)
    print(f"  加速比: {speedup:.2f}x", flush=True)

    # 对比：原版（rollback+replay）
    print(f"\n[原版 Spec (rollback+replay): K={K}, α={alpha}, {len(prompts)} 条]", flush=True)
    from stage9_batch_spec import speculative_decode_batch
    t0 = time.time()
    total_acc2, total_draft2 = 0, 0
    for p in prompts:
        out, stats = speculative_decode_batch(draft, target, tokenizer, p, n_generate=40, K=K, alpha=alpha)
        total_acc2 += stats["accepted"]
        total_draft2 += stats["total_draft"]
    elapsed2 = time.time() - t0
    ar2 = total_acc2 / max(total_draft2, 1)
    spec_time2 = elapsed2 / len(prompts)
    speedup2 = base_time / spec_time2 if spec_time2 > 0 else 0
    print(f"  接受率: {ar2:.2%}", flush=True)
    print(f"  平均 {spec_time2:.2f}s/条", flush=True)
    print(f"  加速比: {speedup2:.2f}x", flush=True)


if __name__ == "__main__":
    main()
