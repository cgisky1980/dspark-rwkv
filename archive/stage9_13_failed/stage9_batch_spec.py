"""DSpark → RWKV · Speculative Decoding (批量验证模式 + Rejection Sampling)

复现 web-rwkv generate_speculative (Leviathan 2023)：
- 标准 rejection sampling：accept_prob = min(1, p/q)，不是简单 argmax 比较
- 拒绝时从 max(0, p - q) 重采样
- 第一个候选用 Logit Fusion (fused = draft + α * target)
- target 批量验证（Full 模式）
"""
import sys
import time
import json
import torch
import torch.nn.functional as F


def softmax_cpu(logits):
    """softmax 计算"""
    m = logits.max()
    e = torch.exp(logits - m)
    return e / e.sum()


def rejection_sample(candidate, small_probs, large_probs):
    """标准 Speculative Decoding 拒绝采样（参考 Leviathan 2023 / web-rwkv line 73-117）。

    返回 (accept: bool, resampled_token: int)
    - accept=True: 接受 candidate
    - accept=False: 拒绝，返回从 max(0, p-q) 重采样的 token

    关键：accept_prob = min(1, p/q)，不是简单的 argmax 比较！
    """
    idx = candidate
    q = small_probs[idx].item()  # draft 概率
    p = large_probs[idx].item()  # target 概率

    # Greedy 快速接受：候选 == target argmax
    large_argmax = large_probs.argmax().item()
    if large_argmax == candidate:
        return True, candidate

    # 概率接受：accept_prob = min(1, p/q)
    if q <= 0:
        accept_prob = 0.0
    else:
        accept_prob = min(1.0, p / q)

    r = torch.rand(1).item()
    if r < accept_prob:
        return True, candidate
    else:
        # 拒绝：从 max(0, p - q) 的归一化分布重采样
        residual = torch.clamp(large_probs - small_probs, min=0.0)
        s = residual.sum().item()
        if s <= 0:
            return False, large_argmax
        # 从 residual 分布采样
        r2 = torch.rand(1).item() * s
        acc = 0.0
        for i in range(residual.shape[0]):
            acc += residual[i].item()
            if acc >= r2:
                return False, i
        return False, residual.shape[0] - 1


from pathlib import Path
from stage9_target_2p9b import RWKV7Target2p9B, DTYPE, DEVICE
from stage9_logit_fusion_01b import RWKV7Draft, TARGET_WEIGHTS, LAMBADA_FILE
from rwkv_tokenizer import TRIE_TOKENIZER

# 同批次 g1a-2.9B target + g1a-0.4B draft（复现 web-rwkv 82.7% 实验）
G1A_TARGET = Path(r"C:\work\niceui\g1a-2.9B.pth")
DRAFT_WEIGHTS = Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1a-0.4b-20250905-ctx4096.pth")


@torch.no_grad()
def speculative_decode_batch(draft, target, tokenizer, prompt, n_generate=40, K=4, alpha=0.5):
    """批量验证模式 Speculative Decoding（标准 rejection sampling）。

    参考 web-rwkv generate_speculative (Leviathan 2023)：
    1. draft 生成 K 个候选，记录每个候选的 small_probs
    2. 第一个候选用 Logit Fusion 增强（fused = draft + alpha * target）
    3. target 批量 forward K 个候选，得到每个位置的 large_logits
    4. 逐位置 rejection_sample：accept_prob = min(1, p/q)
    5. 拒绝时从 max(0, p-q) 重采样，rollback + replay

    返回：(out_tokens, stats)
    """
    ids = tokenizer.encode(prompt)
    out = list(ids)

    t_state = target.zero_state(1)
    d_state = draft.zero_state(1)

    # forward prompt
    ctx = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    t_logits_full, _ = target.forward(ctx, t_state, return_hidden_layers=[])
    d_logits_full = draft.forward(ctx, d_state)
    t_logits = t_logits_full[:, -1, :]  # [1, V]
    d_logits = d_logits_full[:, -1, :]

    stats = {
        "accepted": 0, "rejected": 0, "total_draft": 0,
        "target_forwards": 1, "target_forward_tokens": len(ids),
        "draft_forwards": 1,
    }

    while len(out) - len(ids) < n_generate:
        # === 阶段 1: draft 生成 K 个候选 ===
        d_state_backup = [s.clone() for s in d_state]
        draft_tokens = []
        small_probs_list = []  # 每个候选位置的 draft probs（用于 rejection sampling）
        cur_d_logits = d_logits

        # 保存 target logits 用于第一个候选融合
        large_logits_for_fusion = t_logits.clone()  # [1, V]

        for i in range(K):
            # 第一个候选用 Logit Fusion（参考 web-rwkv line 1546-1555）
            if i == 0:
                fused_logits = cur_d_logits + alpha * large_logits_for_fusion
                probs = softmax_cpu(fused_logits[0])
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

        # === 阶段 2: target 批量验证 ===
        t_state_backup = [s.clone() for s in t_state]
        draft_tensor = torch.tensor([draft_tokens], device=DEVICE, dtype=torch.long)
        t_logits_full, _ = target.forward(draft_tensor, t_state, return_hidden_layers=[])
        stats["target_forwards"] += 1
        stats["target_forward_tokens"] += K

        # 构造 verify_logits_list:
        # [0] = t_logits（prompt 后，验证候选 0）
        # [k+1] = t_logits_full[0, k]（看到 [t1..t_{k+1}] 后，验证候选 k+1）
        verify_logits_list = [t_logits[0]]
        for k in range(K):
            verify_logits_list.append(t_logits_full[0, k])

        # === 阶段 3: 逐位置 rejection sampling ===
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
            # 全部接受
            out.extend(draft_tokens)
            t_logits = t_logits_full[:, -1, :]
            d_logits = cur_d_logits
        else:
            # 中途拒绝
            stats["rejected"] += 1
            out.extend(draft_tokens[:n_accept])
            out.append(correction_token)

            # target state rollback + replay [accepted + correction]
            t_state[0].copy_(t_state_backup[0])
            t_state[1].copy_(t_state_backup[1])
            replay_tokens = draft_tokens[:n_accept] + [correction_token]
            replay_tensor = torch.tensor([replay_tokens], device=DEVICE, dtype=torch.long)
            t_logits_full_new, _ = target.forward(replay_tensor, t_state, return_hidden_layers=[])
            t_logits = t_logits_full_new[:, -1, :]
            stats["target_forwards"] += 1
            stats["target_forward_tokens"] += len(replay_tokens)

            # draft state rollback + replay
            d_state[0].copy_(d_state_backup[0])
            d_state[1].copy_(d_state_backup[1])
            d_logits_full = draft.forward(replay_tensor, d_state)
            d_logits = d_logits_full[:, -1, :]
            stats["draft_forwards"] += 1

        if out[-1] == 0 or len(out) - len(ids) >= n_generate:
            break

    return out, stats


def benchmark_baseline(target, tokenizer, prompt, n_generate=40):
    """基线：纯 target 自回归"""
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
    print("DSpark → RWKV · Speculative Decoding (批量验证 + Rejection Sampling)")
    print(f"  Target: g1a-2.9B  |  Draft: g1a-0.4B-20250905")
    print(f"  Fusion: 第 0 个候选用 fused = draft + α*target，后续纯 draft")
    print(f"  验证: 标准 rejection sampling (accept_prob = min(1, p/q))")
    print(f"  复现 web-rwkv 82.7% 实验（α=0.5, K=4）")
    print("=" * 70)

    tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))
    print("加载 target...", flush=True)
    target = RWKV7Target2p9B(G1A_TARGET)
    print("加载 draft...", flush=True)
    draft = RWKV7Draft(DRAFT_WEIGHTS)
    print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

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
    base_fwd = total_base_fwd / len(prompts)
    print(f"  平均 {base_time:.2f}s/条, target forwards {base_fwd:.1f}", flush=True)

    # 批量验证不同配置（复现 web-rwkv 82.7% 实验：α=0.5，只融合第 0 个候选）
    configs = [
        (2, 0.5), (2, 1.0),
        (4, 0.5), (4, 1.0),
        (6, 0.5),
        (8, 0.5),
    ]

    print(f"\n{'K':>3} {'α':>4} {'接受率':>8} {'t_fwd':>7} {'t_tok':>7} {'d_fwd':>7} {'加速':>7}")
    for K, alpha in configs:
        total_acc, total_draft, total_t_fwd, total_t_tok, total_d_fwd = 0, 0, 0, 0, 0
        t0 = time.time()
        for p in prompts:
            out, stats = speculative_decode_batch(
                draft, target, tokenizer, p, n_generate=40, K=K, alpha=alpha)
            total_acc += stats["accepted"]
            total_draft += stats["total_draft"]
            total_t_fwd += stats["target_forwards"]
            total_t_tok += stats["target_forward_tokens"]
            total_d_fwd += stats["draft_forwards"]
        elapsed = time.time() - t0
        ar = total_acc / max(total_draft, 1)
        spec_time = elapsed / len(prompts)
        speedup = base_time / spec_time if spec_time > 0 else 0
        print(f"{K:>3} {alpha:>4.1f} {ar:>8.2%} "
              f"{total_t_fwd/len(prompts):>7.1f} "
              f"{total_t_tok/len(prompts):>7.1f} "
              f"{total_d_fwd/len(prompts):>7.1f} "
              f"{speedup:>7.2f}x", flush=True)

    # 示例生成
    print(f"\n[示例生成: K=4, α=0.5]", flush=True)
    for p in prompts[:3]:
        out, _ = speculative_decode_batch(draft, target, tokenizer, p, n_generate=40, K=4, alpha=0.5)
        gen = tokenizer.decode(out[len(tokenizer.encode(p)):])
        print(f"  {p} -> {gen[:80]}", flush=True)


if __name__ == "__main__":
    main()
