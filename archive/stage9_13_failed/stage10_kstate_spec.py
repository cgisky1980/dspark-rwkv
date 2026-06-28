"""DSpark → RWKV · 阶段10: K-state 并发验证投机解码

新方案(推翻旧 stage9 顺序验证):
- 利用 RWKV state 可预填特性,K 个 state 并发准备 + 并发验证
- target 一次 forward [1, K] 即经过 K 个 state,输出 K 个位置的 logits
- draft 用 logits 融合生成第一个候选,提升命中率
- 拒绝时从 S_0 重新 forward [1, j+1](j 个接受 + 修正),无 rollback/replay

核心等价关系(已验证 argmax 一致):
  forward([tok_1, ..., tok_K], S_0) 的 logits[0, i]
  == forward(tok_{i+1}, forward(tok_i, ..., forward(tok_1, S_0))) 的 logits

参数:
- K: 每轮 draft 候选数(默认 4)
- alpha: logits 融合系数,fused = draft_logits + alpha * target_logits(默认 0.5)
"""
import math
import time
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from stage9_target_2p9b import RWKV7Target2p9B, DTYPE, DEVICE
from stage9_logit_fusion_01b import RWKV7Draft, TARGET_WEIGHTS, LAMBADA_FILE
from rwkv_tokenizer import TRIE_TOKENIZER

# 0.4B draft(g1d 系列,0.1B 独立预测能力不足)
DRAFT_WEIGHTS = Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1d-0.4b-20260210-ctx8192.pth")


def clone_state(state):
    """深拷贝 state(forward 是 in-place 更新)"""
    return [s.clone() for s in state]


@torch.no_grad()
def speculative_decode(draft, target, tokenizer, prompt, n_generate=40, K=4, alpha=0.5):
    """K-state 并发验证投机解码

    流程:
    1. forward prompt 初始化 S_0, d_state, t_logits, d_logits
    2. 每轮:
       a. draft 用 logits 融合生成 K 个候选(draft state 累积,备份轮首)
       b. target 一次 forward [1, K] 验证(经过 K 个 state,输出 K 个 logits)
          - t_logits_full[0, i] = S_{i+1} 的预测(验证 tok_{i+2})
       c. 决策:从 tok_1 开始,用 S_0 的 t_logits 验证;tok_{i+1} 用 t_logits_full[0, i-1]
       d. 同步:
          - 全接受:t_state = S_K, t_logits = t_logits_full[0, -1]
          - 拒绝在 j:从 S_0 重新 forward [1, j+1](接受+修正),t_state = S_{j+1}
          - draft:回退 + replay 相同序列

    返回:(生成的 token 列表, 统计信息)
    """
    ids = tokenizer.encode(prompt)
    out = list(ids)

    # 初始化 state
    t_state = target.zero_state(1)
    d_state = draft.zero_state(1)

    # forward prompt
    ctx_tensor = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    t_logits_full, _ = target.forward(ctx_tensor, t_state, return_hidden_layers=[])
    d_logits_full = draft.forward(ctx_tensor, d_state)

    # 取最后位置的 logits(= S_0 的预测)
    t_logits = t_logits_full[:, -1, :]  # [1, V]
    d_logits = d_logits_full[:, -1, :]  # [1, V]

    stats = {"accepted": 0, "rejected": 0, "total_draft": 0,
             "target_forwards": 1, "draft_forwards": 1}

    while len(out) - len(ids) < n_generate:
        # === 阶段 1: draft 用 logits 融合生成 K 个候选 ===
        d_state_backup = clone_state(d_state)  # 备份轮首 draft state

        draft_tokens = []
        # 第 1 个候选:logits 融合(draft + alpha * target)
        fused = d_logits + alpha * t_logits
        tok = fused.argmax(dim=-1).item()
        draft_tokens.append(tok)

        # 候选 2..K:纯 draft logits
        for k in range(1, K):
            next_t = torch.tensor([[tok]], device=DEVICE, dtype=torch.long)
            d_l = draft.forward(next_t, d_state)
            tok = d_l[0, -1].argmax().item()
            draft_tokens.append(tok)
            stats["draft_forwards"] += 1

        # draft 最后一次 forward 的 logits(全接受时用)
        d_logits_after = d_l[:, -1, :]  # [1, V]

        stats["total_draft"] += K

        # === 阶段 2: target 一次 forward [1, K] 验证 ===
        # backup S_0(forward 会 in-place 更新)
        t_state_backup = clone_state(t_state)
        draft_tensor = torch.tensor([draft_tokens], device=DEVICE, dtype=torch.long)  # [1, K]
        t_logits_full, _ = target.forward(draft_tensor, t_state_backup, return_hidden_layers=[])
        # t_logits_full[0, i] = forward(tok_1..tok_{i+1}) 后的预测 = S_{i+1} 的预测
        # t_state_backup 现在是 S_K
        stats["target_forwards"] += 1

        # === 阶段 3: 决策 ===
        accepted = 0
        # 验证 tok_1:用 S_0 的 t_logits
        t_pred = t_logits[0].argmax().item()
        if draft_tokens[0] == t_pred:
            accepted = 1
            out.append(draft_tokens[0])
            stats["accepted"] += 1
            # 验证 tok_2..tok_K:用 t_logits_full[0, i-1]
            for i in range(1, K):
                t_pred = t_logits_full[0, i-1].argmax().item()
                if draft_tokens[i] == t_pred:
                    accepted += 1
                    out.append(draft_tokens[i])
                    stats["accepted"] += 1
                else:
                    # 拒绝,用 t_logits_full[0, i-1] 重采样
                    out.append(t_pred)
                    stats["rejected"] += 1
                    break
        else:
            # 第一个就拒绝
            out.append(t_pred)
            stats["rejected"] += 1

        # === 阶段 4: 同步 state ===
        if accepted == K:
            # 全部接受:t_state_backup 是 S_K(已正确)
            t_state = t_state_backup
            t_logits = t_logits_full[:, -1, :]  # [1, V]
            # draft state 已正确(forward 了 K 个 token)
            d_logits = d_logits_after
        else:
            # 拒绝在位置 accepted(0-indexed,前 accepted 个接受)
            # target:从 S_0(t_state 未被修改)重新 forward [1, accepted+1]
            replay = draft_tokens[:accepted] + [out[-1]]  # 接受的 + 修正
            replay_tensor = torch.tensor([replay], device=DEVICE, dtype=torch.long)
            t_logits_replay, _ = target.forward(replay_tensor, t_state, return_hidden_layers=[])
            t_logits = t_logits_replay[:, -1, :]  # [1, V]
            stats["target_forwards"] += 1

            # draft:回退到轮首 + replay
            d_state = clone_state(d_state_backup)
            d_logits_replay = draft.forward(replay_tensor, d_state)
            d_logits = d_logits_replay[:, -1, :]  # [1, V]
            stats["draft_forwards"] += 1

        if out[-1] == 0 or len(out) - len(ids) >= n_generate:
            break

    return out, stats


def benchmark_baseline(target, tokenizer, prompt, n_generate=40):
    """基线:纯 target 自回归生成"""
    ids = tokenizer.encode(prompt)
    out = list(ids)
    state = target.zero_state(1)
    ctx_tensor = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    logits, _ = target.forward(ctx_tensor, state, return_hidden_layers=[])
    forwards = 1
    while len(out) - len(ids) < n_generate:
        next_tok = logits[0, -1].argmax().item()
        out.append(next_tok)
        next_t = torch.tensor([[next_tok]], device=DEVICE, dtype=torch.long)
        logits, _ = target.forward(next_t, state, return_hidden_layers=[])
        forwards += 1
        if next_tok == 0:
            break
    return out, {"target_forwards": forwards}


def debug_target_equivalence(target, tokenizer, prompt, K=4):
    """调试:对比 target forward [1, K] 和逐 token forward 的 argmax

    等价性测试(0.1B draft)显示 argmax 一致,但 2.9B target 可能有差异。
    若 argmax 不一致 → K-state 方案的根基有问题。
    """
    print(f"\n[调试: target 2.9B forward [1, K] vs 逐 token forward]")
    print(f"  prompt: {prompt}")

    ids = tokenizer.encode(prompt)
    ctx = torch.tensor([ids], device=DEVICE, dtype=torch.long)

    # S_0
    state_0 = target.zero_state(1)
    logits_prompt, _ = target.forward(ctx, state_0, return_hidden_layers=[])
    t_logits_0 = logits_prompt[:, -1, :]  # S_0 的预测

    # 生成 K 个 token(纯 target 逐 token)
    seq_tokens = []
    cur_state = [s.clone() for s in state_0]
    cur_logits = t_logits_0
    for i in range(K):
        tok = cur_logits[0].argmax().item()
        seq_tokens.append(tok)
        next_t = torch.tensor([[tok]], device=DEVICE, dtype=torch.long)
        l, _ = target.forward(next_t, cur_state, return_hidden_layers=[])
        cur_logits = l[:, -1, :]

    print(f"  逐 token 生成的 {K} 个 token: {seq_tokens}")
    print(f"  解码: {tokenizer.decode(seq_tokens)}")

    # 方式 A:一次 forward [1, K]
    state_A = [s.clone() for s in state_0]
    tokens_A = torch.tensor([seq_tokens], device=DEVICE, dtype=torch.long)
    logits_A, _ = target.forward(tokens_A, state_A, return_hidden_layers=[])
    # logits_A[0, i] = forward(seq_tokens[:i+1]) 后的预测 = S_{i+1} 的预测

    # 方式 B:逐 token forward,记录每步 logits
    state_B = [s.clone() for s in state_0]
    logits_B_list = [t_logits_0[0]]  # S_0 的预测
    for i in range(K):
        next_t = torch.tensor([[seq_tokens[i]]], device=DEVICE, dtype=torch.long)
        l, _ = target.forward(next_t, state_B, return_hidden_layers=[])
        logits_B_list.append(l[0, -1, :])  # S_{i+1} 的预测

    # 对比 argmax
    print(f"\n  argmax 对比(预测下一个 token):")
    print(f"  {'位置':<8} {'forward[1,K]':<15} {'逐token':<15} {'一致':<6}")
    all_match = True
    # S_0 预测(方式 B 的 logits_B_list[0])
    pred_A_0 = t_logits_0[0].argmax().item()  # 都一样
    pred_B_0 = logits_B_list[0].argmax().item()
    print(f"  {'S_0':<8} {pred_A_0:<15} {pred_B_0:<15} {'✓' if pred_A_0==pred_B_0 else '✗'}")
    for i in range(K):
        # forward[1,K] 的 S_{i+1} 预测 = logits_A[0, i]
        pred_A = logits_A[0, i].argmax().item()
        # 逐 token 的 S_{i+1} 预测 = logits_B_list[i+1]
        pred_B = logits_B_list[i+1].argmax().item()
        match = pred_A == pred_B
        if not match:
            all_match = False
        print(f"  {'S_'+str(i+1):<8} {pred_A:<15} {pred_B:<15} {'✓' if match else '✗'}")

    print(f"\n  结论: {'✓ argmax 全部一致' if all_match else '✗ argmax 不一致!K-state 方案有问题'}")
    return all_match


def main():
    print("=" * 70)
    print("DSpark → RWKV · 阶段10: K-state 并发验证投机解码")
    print("  新方案:target 一次 forward [1, K] 验证 K 个 token")
    print("  draft 用 logits 融合生成第一个候选")
    print("  0.4B g1d draft + 2.9B target")
    print("=" * 70)

    tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))
    target = RWKV7Target2p9B(TARGET_WEIGHTS)
    draft = RWKV7Draft(DRAFT_WEIGHTS)
    print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # 先调试 target 等价性
    debug_target_equivalence(target, tokenizer, "中国的首都是哪里", K=4)

    # === 阶段 1: 简单中文 prompt ===
    simple_prompts = [
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

    N_SIMPLE = 10
    print(f"\n[阶段 1: 简单中文 prompt, {N_SIMPLE} 条]")
    configs = [
        {"K": 4, "alpha": 2.0},
        {"K": 16, "alpha": 2.0},
    ]

    for cfg in configs:
        K, alpha = cfg["K"], cfg["alpha"]
        print(f"\n--- K={K}, α={alpha} ---")
        total_accepted = 0
        total_draft = 0
        total_forwards = 0
        t0 = time.time()
        for prompt in simple_prompts[:N_SIMPLE]:
            out, stats = speculative_decode(draft, target, tokenizer, prompt,
                                            n_generate=40, K=K, alpha=alpha)
            total_accepted += stats["accepted"]
            total_draft += stats["total_draft"]
            total_forwards += stats["target_forwards"]
        elapsed = time.time() - t0
        accept_rate = total_accepted / max(total_draft, 1)
        avg_forwards = total_forwards / N_SIMPLE
        print(f"  接受率: {accept_rate:.4f} ({total_accepted}/{total_draft})")
        print(f"  平均 target forwards: {avg_forwards:.1f}")
        print(f"  平均每条 {elapsed/N_SIMPLE:.2f}s")

    # === 阶段 2: LAMBADA 基线对比 ===
    texts = []
    with open(LAMBADA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    texts = texts[:30]

    N_BASELINE = 10
    N_SPEC = 10

    # 基线
    print(f"\n[阶段 2 基线: LAMBADA 纯 target, {N_BASELINE} 条]")
    t0 = time.time()
    total_forwards_base = 0
    for text in texts[:N_BASELINE]:
        _, stats = benchmark_baseline(target, tokenizer, text, n_generate=40)
        total_forwards_base += stats["target_forwards"]
    baseline_time = (time.time() - t0) / N_BASELINE
    avg_forwards_base = total_forwards_base / N_BASELINE
    print(f"  平均每条 {baseline_time:.2f}s, target forwards: {avg_forwards_base:.1f}")

    # target sanity check:打印基线生成内容,确认 target 正常
    print(f"\n[target sanity check: 纯 target 生成]")
    for prompt in simple_prompts[:3]:
        out, _ = benchmark_baseline(target, tokenizer, prompt, n_generate=40)
        gen = tokenizer.decode(out[len(tokenizer.encode(prompt)):])
        print(f"  {prompt}")
        print(f"    → {gen[:80]}")

    # Speculative decoding
    for cfg in configs:
        K, alpha = cfg["K"], cfg["alpha"]
        print(f"\n[Speculative K={K}, α={alpha}, {N_SPEC} 条]")
        total_accepted = 0
        total_draft = 0
        total_forwards = 0
        t0 = time.time()
        for text in texts[:N_SPEC]:
            out, stats = speculative_decode(draft, target, tokenizer, text,
                                            n_generate=40, K=K, alpha=alpha)
            total_accepted += stats["accepted"]
            total_draft += stats["total_draft"]
            total_forwards += stats["target_forwards"]
        elapsed = time.time() - t0
        accept_rate = total_accepted / max(total_draft, 1)
        avg_forwards = total_forwards / N_SPEC
        speedup = baseline_time / (elapsed / N_SPEC) if elapsed > 0 else 0
        print(f"  接受率: {accept_rate:.4f} ({total_accepted}/{total_draft})")
        print(f"  平均 target forwards: {avg_forwards:.1f} (vs 基线 {avg_forwards_base:.1f})")
        print(f"  forward 减少: {(1 - avg_forwards/avg_forwards_base)*100:.1f}%")
        print(f"  平均每条 {elapsed/N_SPEC:.2f}s (vs 基线 {baseline_time:.2f}s)")
        print(f"  时间加速比: {speedup:.2f}x")

    # === 示例输出 ===
    print(f"\n[示例输出]")
    K, alpha = 4, 0.5
    for prompt in simple_prompts[:3]:
        out, _ = speculative_decode(draft, target, tokenizer, prompt,
                                    n_generate=40, K=K, alpha=alpha)
        gen = tokenizer.decode(out[len(tokenizer.encode(prompt)):])
        print(f"  {prompt}")
        print(f"    → {gen[:80]}")


if __name__ == "__main__":
    main()
