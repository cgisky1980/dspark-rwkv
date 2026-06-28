"""DSpark → RWKV · 阶段9: Speculative Decoding with Logit Fusion

复现之前 0.4B+2.93B web-rwkv 实验的顺序验证方案（82.7% 接受率）：
1. 小模型生成 K 个候选 token（draft forward 逐个生成）
2. target 用当前 logits 验证候选 0（fusion 生成时的 target logits）
3. 验证通过 → target forward 该 token（Last 模式）→ 得到新 logits
4. 用新 logits 验证候选 1
5. 拒绝就 break，用 target logits 重采样
6. state 始终正确（只处理了接受的 token + 1 个修正 token），无需 snapshot/restore

关键：
- target 顺序 forward 接受的 token（不是批量 forward 所有 draft token）
- fusion 只对第一个候选（有精确的 target logits）
- alpha=0.5（logit 加法：fused = draft + alpha * target）
"""
import math
import time
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from stage9_target_2p9b import RWKV7Target2p9B, DTYPE, DEVICE
from rwkv_tokenizer import TRIE_TOKENIZER
from stage9_logit_fusion_01b import RWKV7Draft, TARGET_WEIGHTS, LAMBADA_FILE

# 0.4B draft（g1d 系列，L=24, C=1024, H=16, N=64）
# 0.1B 独立预测能力为 0，logit fusion 失效；0.4B 有 54.7% 独立接受率
DRAFT_WEIGHTS = Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1d-0.4b-20260210-ctx8192.pth")


@torch.no_grad()
def speculative_decode(draft, target, tokenizer, prompt, n_generate=40, K=4, alpha=0.5):
    """Speculative decoding with Logit Fusion（顺序验证模式）。

    流程（复现 web-rwkv 顺序验证）：
    1. target + draft forward prompt，初始化 state 和 logits
    2. 每轮：
       a. draft 生成 K 个候选（draft state 累积更新，保存候选）
       b. target 逐个验证：
          - 用当前 t_logits 验证候选 0
          - 接受 → target forward 该 token，更新 t_logits → 验证候选 1
          - 拒绝 → 用 t_logits 重采样，break
       c. 同步 draft state：回退到轮首 + replay 接受的 token + 修正 token

    state 管理关键：
    - target state 始终正确（只 forward 接受的 token + 修正 token）
    - draft state 在拒绝时需要回退 + replay（因为 draft forward 了 K 个 token，但只接受了部分）

    返回：(生成的 token 列表, 统计信息)
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

    # 取最后一个位置的 logits [1, V]
    t_logits = t_logits_full[:, -1, :]
    d_logits = d_logits_full[:, -1, :]

    stats = {"accepted": 0, "rejected": 0, "total_draft": 0,
             "target_forwards": 1, "draft_forwards": 1}

    while len(out) - len(ids) < n_generate:
        # === 阶段 1: draft 生成 K 个候选 ===
        # 备份 draft state（用于拒绝时回退）
        d_state_backup = [s.clone() for s in d_state]

        draft_tokens = []
        cur_d_logits = d_logits  # [1, V]

        # 第 1 个候选：fusion（target logits 可用）
        fused_logits = cur_d_logits + alpha * t_logits
        next_tok = fused_logits.argmax(dim=-1).item()
        draft_tokens.append(next_tok)

        # draft forward 第 1 个候选
        next_t = torch.tensor([[next_tok]], device=DEVICE, dtype=torch.long)
        d_logits_full = draft.forward(next_t, d_state)
        cur_d_logits = d_logits_full[:, -1, :]
        stats["draft_forwards"] += 1

        # 候选 1+：纯 draft logits
        for k in range(1, K):
            next_tok = cur_d_logits.argmax(dim=-1).item()
            draft_tokens.append(next_tok)
            next_t = torch.tensor([[next_tok]], device=DEVICE, dtype=torch.long)
            d_logits_full = draft.forward(next_t, d_state)
            cur_d_logits = d_logits_full[:, -1, :]
            stats["draft_forwards"] += 1

        stats["total_draft"] += K

        # === 阶段 2: target 顺序验证（Last 模式）===
        # target 用当前 t_logits 验证候选 0，接受后 forward 该 token 得到新 t_logits
        # 拒绝就用 t_logits 重采样，break
        accepted_count = 0
        for k in range(K):
            t_pred = t_logits[0].argmax().item()

            if draft_tokens[k] == t_pred:
                # 接受：target forward 该 token，更新 t_logits
                out.append(draft_tokens[k])
                accepted_count += 1
                stats["accepted"] += 1
                next_t = torch.tensor([[draft_tokens[k]]], device=DEVICE, dtype=torch.long)
                t_logits_full, _ = target.forward(next_t, t_state, return_hidden_layers=[])
                t_logits = t_logits_full[:, -1, :]
                stats["target_forwards"] += 1
            else:
                # 拒绝：用 t_logits 重采样，break
                out.append(t_pred)
                stats["rejected"] += 1
                # target forward 修正 token，更新 t_logits
                next_t = torch.tensor([[t_pred]], device=DEVICE, dtype=torch.long)
                t_logits_full, _ = target.forward(next_t, t_state, return_hidden_layers=[])
                t_logits = t_logits_full[:, -1, :]
                stats["target_forwards"] += 1
                break

        # === 阶段 3: 同步 draft state ===
        # draft 在阶段 1 forward 了 K 个 token，但 target 只接受了 accepted_count 个 + 可能的修正 token
        # 需要回退 draft state 到轮首，replay 接受的 token + 修正 token（如果有）
        if accepted_count == K:
            # 全部接受，draft state 已正确（forward 了 K 个 token）
            d_logits = cur_d_logits  # [1, V]
        else:
            # 中途拒绝：回退 + replay (accepted_count + 1) 个 token
            # replay 的 token：draft_tokens[:accepted_count] + [out[-1]]（修正的 t_pred）
            replay = draft_tokens[:accepted_count] + [out[-1]]
            replay_tensor = torch.tensor([replay], device=DEVICE, dtype=torch.long)
            # draft 回退 + 重新 forward
            d_state[0].copy_(d_state_backup[0])
            d_state[1].copy_(d_state_backup[1])
            d_logits_full = draft.forward(replay_tensor, d_state)
            d_logits = d_logits_full[:, -1, :]
            stats["draft_forwards"] += 1

        if out[-1] == 0 or len(out) - len(ids) >= n_generate:
            break

    return out, stats


def benchmark_baseline(target, tokenizer, prompt, n_generate=40):
    """基线：纯 target 自回归生成"""
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


def main():
    print("=" * 70)
    print("DSpark → RWKV · Speculative Decoding with Logit Fusion (顺序验证)")
    print("  0.4B draft (Logit Fusion 增强) + 2.9B target 验证")
    print("  复现 web-rwkv 顺序验证方案（82.7% 接受率）")
    print("  fusion: fused = draft + alpha * target (logit 加法)")
    print("=" * 70)

    tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))
    target = RWKV7Target2p9B(TARGET_WEIGHTS)
    draft = RWKV7Draft(DRAFT_WEIGHTS)
    print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # === 阶段 1: 简单中文 prompt（复现之前 82.7% 接受率的场景）===
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
    print(f"\n[阶段 1: 简单中文 prompt, {N_SIMPLE} 条, 复现 82.7% 场景]")
    K, alpha = 2, 0.5
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
    print(f"  K={K}, α={alpha}")
    print(f"  接受率: {accept_rate:.4f} ({total_accepted}/{total_draft})")
    print(f"  平均 target forwards: {avg_forwards:.1f}")
    print(f"  平均每条 {elapsed/N_SIMPLE:.2f}s")
    # 打印第一条的生成结果
    out, _ = speculative_decode(draft, target, tokenizer, simple_prompts[0],
                                n_generate=40, K=K, alpha=alpha)
    gen_text = tokenizer.decode(out[len(tokenizer.encode(simple_prompts[0])):])
    print(f"  示例: {simple_prompts[0]}")
    print(f"  生成: {gen_text[:100]}")

    # === 阶段 2: LAMBADA（难任务对照）===
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
    print(f"\n[阶段 2 基线: LAMBADA 纯 target 自回归, {N_BASELINE} 条]")
    t0 = time.time()
    total_forwards_base = 0
    for text in texts[:N_BASELINE]:
        _, stats = benchmark_baseline(target, tokenizer, text, n_generate=40)
        total_forwards_base += stats["target_forwards"]
    baseline_time = (time.time() - t0) / N_BASELINE
    avg_forwards_base = total_forwards_base / N_BASELINE
    print(f"  平均每条 {baseline_time:.2f}s, target forwards: {avg_forwards_base:.1f}")

    # Speculative decoding
    # alpha 是 logit 加法系数：fused = draft + alpha * target
    # 0.4B 实验最优 alpha=0.5
    configs = [
        {"K": 2, "alpha": 0.5},
        {"K": 4, "alpha": 0.5},
        {"K": 2, "alpha": 1.0},
        {"K": 4, "alpha": 1.0},
    ]

    for cfg in configs:
        K, alpha = cfg["K"], cfg["alpha"]
        print(f"\n[Speculative K={K}, α={alpha}, {N_SPEC} 条]")
        total_accepted = 0
        total_draft = 0
        total_forwards = 0
        total_draft_forwards = 0
        t0 = time.time()
        for text in texts[:N_SPEC]:
            out, stats = speculative_decode(draft, target, tokenizer, text,
                                            n_generate=40, K=K, alpha=alpha)
            total_accepted += stats["accepted"]
            total_draft += stats["total_draft"]
            total_forwards += stats["target_forwards"]
            total_draft_forwards += stats["draft_forwards"]
        elapsed = time.time() - t0
        accept_rate = total_accepted / max(total_draft, 1)
        avg_forwards = total_forwards / N_SPEC
        avg_draft_forwards = total_draft_forwards / N_SPEC
        speedup = baseline_time / (elapsed / N_SPEC) if elapsed > 0 else 0
        print(f"  接受率: {accept_rate:.4f} ({total_accepted}/{total_draft})")
        print(f"  平均 target forwards: {avg_forwards:.1f} (vs 基线 {avg_forwards_base:.1f})")
        print(f"  平均 draft forwards: {avg_draft_forwards:.1f}")
        print(f"  forward 减少: {(1 - avg_forwards/avg_forwards_base)*100:.1f}%")
        print(f"  平均每条 {elapsed/N_SPEC:.2f}s (vs 基线 {baseline_time:.2f}s)")
        print(f"  时间加速比: {speedup:.2f}x")


if __name__ == "__main__":
    main()
