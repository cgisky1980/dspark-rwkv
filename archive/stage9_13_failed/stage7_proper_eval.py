"""DSpark → RWKV · 阶段7 正确评估

修复 v3 报告中的三个硬伤：
1. **过拟合**：训练/验证同一份 stage2_train_5layer.pt → 改用独立测试集
   - 测试集 A：target 自回归新生成 256 条序列（seed 不同）
   - 测试集 B：LAMBADA 真实文本 256 条
2. **不是 draft generation**：原版用 ground truth 喂回 → 改为 draft 自回归
   - 从 anchor 起，draft 每步用自己上一步生成的 token 作输入
3. **top1 命中 ≠ 接受率**：
   - 真实接受率（greedy）：draft_token == target_argmax
   - 真实接受率（采样）：MH 接受 min(1, p_target/p_draft)
   - 加速比 = (1 + n_accepted) / (t_draft_total + t_target_verify)
"""
import json
import math
import time
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================================================================
# 日志（解决 PowerShell buffer 吞输出问题）
# ==================================================================
log_path = Path(__file__).parent / "stage7_proper_eval.log"
logger = logging.getLogger("stage7")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(ch)

def log(msg):
    logger.info(msg)

# ==================================================================
# 导入模型定义（来自 stage2_train_v3.py）和 target
# ==================================================================
from stage2_train_v3 import (
    DSparkDraft, VOCAB, D_DRAFT, N_DRAFT_LAYERS, N_HEADS, CTX,
    TARGET_LAYERS, N_TARGET_LAYERS, D_TARGET, BS, SQRT_E,
)
from stage2_target import RWKV7Target, WEIGHTS as TARGET_WEIGHTS
from rwkv_tokenizer import TRIE_TOKENIZER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK = 4
V3_WEIGHTS = Path(__file__).parent / "weights" / "v3_block4_5000.pth"
VOCAB_FILE = Path(__file__).parent / "rwkv_vocab_v20230424.txt"
LAMBADA_FILE = Path(__file__).parent / "data" / "lambada_test.jsonl"

# 独立测试集大小
N_TEST_SEQS = 256
T_LEN = 32

# ==================================================================
# 模型加载
# ==================================================================
def load_v3_model():
    """加载训练好的 v3 权重。"""
    log(f"[加载 v3 权重] {V3_WEIGHTS}")
    if not V3_WEIGHTS.exists():
        # 尝试从 test/dspark_rwkv/weights 软链
        alt = Path(r"c:\work\niceui\test\dspark_rwkv\weights\v3_block4_5000.pth")
        if alt.exists():
            log(f"  主路径不存在，从 {alt} 软链")
            V3_WEIGHTS.parent.mkdir(exist_ok=True)
            if not V3_WEIGHTS.exists():
                import os
                os.symlink(alt, V3_WEIGHTS)
        else:
            raise FileNotFoundError(f"找不到 v3 权重: {V3_WEIGHTS} 或 {alt}")

    ckpt = torch.load(V3_WEIGHTS, map_location="cpu", weights_only=True)
    log(f"  ckpt keys: {list(ckpt.keys())}")
    log(f"  训练时 pos_rate: {ckpt.get('pos_rate')}")
    log(f"  训练时 avg:      {ckpt.get('avg')}")

    model = DSparkDraft(BLOCK).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  参数量: {n_params/1e6:.1f}M")
    return model


def load_target():
    """加载 RWKV-7 target 模型。"""
    log(f"[加载 target] {TARGET_WEIGHTS}")
    target = RWKV7Target(TARGET_WEIGHTS)
    target.z = {k: v.to(DEVICE) for k, v in target.z.items()}
    log(f"  target 在 GPU")
    return target


# ==================================================================
# 测试集 A：target 自回归生成（独立 seed，未参与训练）
# ==================================================================
@torch.no_grad()
def gen_test_set_autoregressive(target, n_seqs=N_TEST_SEQS, t_len=T_LEN, seed=999):
    """用 target 生成独立测试集（seed 与训练时不同）。

    返回：
      tokens: [N, t_len+1]
      hids:   dict[layer] -> [N, t_len, C]
    """
    log(f"\n[测试集 A] target 自回归生成 (seed={seed}, N={n_seqs}, T={t_len})")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    all_tokens = []
    all_hids = {l: [] for l in TARGET_LAYERS}
    BATCH = 32
    n_done = 0
    while n_done < n_seqs:
        B = min(BATCH, n_seqs - n_done)
        cur = torch.randint(0, 1000, (B, 1), device=DEVICE)
        state = target.zero_state(B, device=DEVICE)
        seq_tokens = [cur]
        seq_hids = {l: [] for l in TARGET_LAYERS}
        for t in range(t_len):
            logits, hids = target.forward(cur, state, return_hidden_layers=TARGET_LAYERS)
            probs = torch.softmax(logits[:, -1] / 0.8, dim=-1)
            nxt = torch.multinomial(probs, 1)
            seq_tokens.append(nxt)
            for i, l in enumerate(TARGET_LAYERS):
                seq_hids[l].append(hids[i][:, -1:, :])
            cur = nxt
        all_tokens.append(torch.cat(seq_tokens, dim=1).cpu())
        for l in TARGET_LAYERS:
            all_hids[l].append(torch.cat(seq_hids[l], dim=1).cpu())
        n_done += B
        if n_done % 64 == 0:
            log(f"  生成 {n_done}/{n_seqs}")

    tokens = torch.cat(all_tokens, dim=0)
    hids = {l: torch.cat(all_hids[l], dim=0) for l in TARGET_LAYERS}
    log(f"  完成: tokens={tokens.shape}, hidden_{TARGET_LAYERS[0]}={hids[TARGET_LAYERS[0]].shape}")
    return tokens, hids


# ==================================================================
# 测试集 B：LAMBADA 真实文本（训练时用前 512 条，测试用后续 256 条）
# ==================================================================
def gen_test_set_lambada(target, tokenizer, n_seqs=N_TEST_SEQS, t_len=T_LEN):
    """用 LAMBADA 后续句子作独立测试集（跳过前 512 条训练用过的）。

    返回：
      tokens: [N, t_len+1]
      hids:   dict[layer] -> [N, t_len, C]
    """
    log(f"\n[测试集 B] LAMBADA 真实文本 (跳过前 512 条训练用, N={n_seqs}, T={t_len})")
    seqs = []
    skipped = 0
    with open(LAMBADA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            ids = tokenizer.encode(text)
            if len(ids) >= t_len + 1:
                if skipped < 512:  # 跳过训练用过的前 512 条
                    skipped += 1
                    continue
                seqs.append(ids[:t_len + 1])
                if len(seqs) >= n_seqs:
                    break
    log(f"  跳过 {skipped} 条训练句，取 {len(seqs)} 条新句")

    all_tokens = []
    all_hids = {l: [] for l in TARGET_LAYERS}
    BATCH = 16  # 真实文本 batch 小一点
    for start in range(0, len(seqs), BATCH):
        end = min(start + BATCH, len(seqs))
        B = end - start
        tokens_batch = torch.tensor(seqs[start:end], dtype=torch.long, device=DEVICE)
        state = target.zero_state(B, device=DEVICE)
        with torch.no_grad():
            _, hids = target.forward(tokens_batch, state, return_hidden_layers=TARGET_LAYERS)
        all_tokens.append(tokens_batch.cpu())
        for i, l in enumerate(TARGET_LAYERS):
            all_hids[l].append(hids[i][:, :t_len, :].cpu())

    tokens = torch.cat(all_tokens, dim=0)
    hids = {l: torch.cat(all_hids[l], dim=0) for l in TARGET_LAYERS}
    log(f"  完成: tokens={tokens.shape}, hidden_{TARGET_LAYERS[0]}={hids[TARGET_LAYERS[0]].shape}")
    return tokens, hids


# ==================================================================
# 正确评估：真实 draft generation + target 验证
# ==================================================================
@torch.no_grad()
def proper_eval(model, target, tokens, hids, test_name):
    """正确评估 v3 的真实接受率。

    流程（每个样本）：
    1. 选 anchor 位置 a（ctx <= a < T - block）
    2. draft 自回归生成 block 个 token：
       - prev[:, 0] = anchor_token
       - draft forward → logits[:, 0]
       - draft_token[:, 0] = argmax(logits[:, 0])
       - prev[:, 1] = draft_token[:, 0]   ← 用 draft 自己的输出，不是 ground truth!
       - draft forward → logits[:, 1]
       - ... 重复 block 次
    3. target 一次 forward [anchor, draft_tokens] → target_logits
    4. 接受率：
       - greedy: draft_token == target_argmax
       - MH 采样: accept_prob = min(1, p_target(draft_token) / p_draft(draft_token))
    """
    log(f"\n{'='*60}")
    log(f"[正确评估] {test_name}")
    log(f"  样本数: {tokens.shape[0]}, 序列长: {tokens.shape[1]}, block: {BLOCK}")
    log(f"{'='*60}")

    N, T1 = tokens.shape
    T = T1 - 1
    max_anchor = T - BLOCK
    assert max_anchor >= CTX, f"序列太短: T={T}, CTX={CTX}, BLOCK={BLOCK}"

    tokens = tokens.to(DEVICE)
    hids = {k: v.to(DEVICE) for k, v in hids.items()}

    # 每个 batch 评估
    n_samples = min(N, 256)  # 评估 256 个样本
    idx_list = torch.randperm(N)[:n_samples].tolist()

    # 收集统计
    greedy_accept = [0] * BLOCK   # greedy 接受数
    mh_accept = [0.0] * BLOCK     # MH 平均接受概率
    n_total = 0

    # 用于加速比计算的时间统计
    t_draft_total = 0.0  # draft 总耗时
    t_target_verify = 0.0  # target 验证耗时

    BATCH_EVAL = 32
    for batch_start in range(0, n_samples, BATCH_EVAL):
        batch_idx = idx_list[batch_start:batch_start + BATCH_EVAL]
        B = len(batch_idx)
        # 随机选 anchor
        anchors = [torch.randint(CTX, max_anchor + 1, (1,)).item() for _ in batch_idx]

        # 准备 ctx_hidden: [B, CTX, D_TARGET * N_TARGET_LAYERS]
        # v3 训练时用 torch.cat(list_of_[CTX, D], dim=-1) → [CTX, D*N_LAYERS]
        ctx_list = []
        for b in range(B):
            i_seq = batch_idx[b]
            a = anchors[b]
            ctx_l = torch.cat([hids[l][i_seq, a-CTX:a] for l in TARGET_LAYERS], dim=-1)  # [CTX, D*N_LAYERS]
            ctx_list.append(ctx_l)
        ctx_hidden = torch.stack(ctx_list, dim=0)  # [B, CTX, D*N_LAYERS]

        anchor_tokens = torch.tensor([tokens[batch_idx[b], anchors[b]] for b in range(B)],
                                      device=DEVICE, dtype=torch.long)

        # === draft generation（自回归，每步用自己上一步输出）===
        # v3 是「并行 trunk + 顺序 head」一次性预测 BLOCK 个 token。
        # 真实 draft generation：逐步生成，每步 prev_tokens 前缀更新为 draft 自己的输出。
        # 实现方式：循环 BLOCK 次，每次把已生成的 draft token 填入 prev_tokens 的下一位置，
        # 然后重新 forward 取第 t 个位置预测。每次 forward 都重算所有位置（低效但正确）。
        draft_tokens = torch.zeros(B, BLOCK, device=DEVICE, dtype=torch.long)
        prev_tokens = torch.zeros(B, BLOCK, device=DEVICE, dtype=torch.long)
        prev_tokens[:, 0] = anchor_tokens

        t0 = time.perf_counter()
        for t in range(BLOCK):
            # forward 整个 block（每次 prev_tokens 前缀更新）
            draft_logits, _ = model(anchor_tokens, ctx_hidden, prev_tokens)
            # 取第 t 个位置的预测
            next_tok = draft_logits[:, t, :].argmax(dim=-1)  # [B]
            draft_tokens[:, t] = next_tok
            # 下一步 prev 用 draft 自己的输出（不是 ground truth！）
            if t + 1 < BLOCK:
                prev_tokens[:, t + 1] = next_tok
        torch.cuda.synchronize()
        t_draft_total += time.perf_counter() - t0

        # === target 验证 ===
        # target forward [anchor, draft_0, draft_1, ..., draft_{BLOCK-1}]
        verify_input = torch.cat([anchor_tokens.unsqueeze(1), draft_tokens], dim=1)  # [B, BLOCK+1]
        state = target.zero_state(B, device=DEVICE)
        t0 = time.perf_counter()
        target_logits, _ = target.forward(verify_input, state, return_hidden_layers=[])
        torch.cuda.synchronize()
        t_target_verify += time.perf_counter() - t0
        # target_logits: [B, BLOCK+1, V]，取位置 1..BLOCK（预测每个 draft token 的位置）
        target_logits_for_draft = target_logits[:, 1:BLOCK+1, :]  # [B, BLOCK, V]
        target_argmax = target_logits_for_draft.argmax(dim=-1)  # [B, BLOCK]

        # === 接受率计算 ===
        # 1. greedy 接受：draft_token == target_argmax
        greedy_match = (draft_tokens == target_argmax)  # [B, BLOCK]
        for t in range(BLOCK):
            greedy_accept[t] += greedy_match[:, t].sum().item()

        # 2. MH 采样接受：accept_prob = min(1, p_target(draft_tok) / p_draft(draft_tok))
        #    这里用 target 和 draft 的 softmax 概率
        draft_logits_for_prob, _ = model(anchor_tokens, ctx_hidden, prev_tokens)
        draft_probs = F.softmax(draft_logits_for_prob, dim=-1)  # [B, BLOCK, V]
        target_probs = F.softmax(target_logits_for_draft, dim=-1)  # [B, BLOCK, V]
        # 取 draft_token 对应的概率
        draft_tok_idx = draft_tokens.unsqueeze(-1)  # [B, BLOCK, 1]
        p_draft = draft_probs.gather(-1, draft_tok_idx).squeeze(-1)  # [B, BLOCK]
        p_target = target_probs.gather(-1, draft_tok_idx).squeeze(-1)  # [B, BLOCK]
        # MH 接受概率
        mh_ratio = (p_target / (p_draft + 1e-10)).clamp(max=1.0)  # [B, BLOCK]
        for t in range(BLOCK):
            mh_accept[t] += mh_ratio[:, t].sum().item()

        n_total += B

    # 汇总
    greedy_rate = [greedy_accept[t] / n_total for t in range(BLOCK)]
    mh_rate = [mh_accept[t] / n_total for t in range(BLOCK)]
    avg_greedy = sum(greedy_rate) / BLOCK
    avg_mh = sum(mh_rate) / BLOCK

    # 期望接受长度 E[len] = sum_{t>=1} P(至少接受 t 个) = sum_{t=1}^{BLOCK} prod_{i<t} p_i
    # 注意：DSpark 中遇错就回滚到第一个错位置，所以 E[len] = sum_t P(>= t 个被接受)
    expected_len = 0.0
    cum_p = 1.0
    for t in range(BLOCK):
        cum_p *= greedy_rate[t]  # P(位置 t 被接受)
        expected_len += cum_p    # P(至少接受 t+1 个)

    # 加速比估算（每个 sample）：
    # - baseline: target 串行 forward E[len] 个 token（实际生成 E[len] 个 token）
    # - spec_decoding: draft 跑 BLOCK 次 forward + target 验证 1 次 forward（BLOCK+1 个位置）
    # 加速比 = (E[len] * target_single_step) / (t_draft + t_target_verify)
    # 注意：t_draft 是 BLOCK 次 draft forward 的总耗时（每次重算整个 block，是 v3 评估的低效实现，真实部署会优化）
    avg_t_draft = t_draft_total / n_total
    avg_t_target = t_target_verify / n_total
    target_single_step = avg_t_target / (BLOCK + 1)  # target forward BLOCK+1 个位置
    real_speedup = (expected_len * target_single_step) / (avg_t_draft + avg_t_target) if (avg_t_draft + avg_t_target) > 0 else 0

    log(f"\n  --- 结果 ---")
    log(f"  样本数: {n_total}")
    log(f"  greedy 各位置接受率: {[f'{r:.4f}' for r in greedy_rate]}")
    log(f"  greedy 平均接受率:   {avg_greedy:.4f}")
    log(f"  MH 各位置接受概率:   {[f'{r:.4f}' for r in mh_rate]}")
    log(f"  MH 平均接受概率:     {avg_mh:.4f}")
    log(f"  期望接受长度 E[len]: {expected_len:.4f} / {BLOCK}")
    log(f"  平均 draft 耗时:     {avg_t_draft*1000:.2f} ms/sample")
    log(f"  平均 target 验证耗时: {avg_t_target*1000:.2f} ms/sample")
    log(f"  target 单步耗时:     {target_single_step*1000:.2f} ms/step")
    log(f"  估算加速比:          {real_speedup:.3f}x")

    return {
        "test_name": test_name,
        "n_samples": n_total,
        "greedy_rate": greedy_rate,
        "avg_greedy": avg_greedy,
        "mh_rate": mh_rate,
        "avg_mh": avg_mh,
        "expected_len": expected_len,
        "avg_t_draft_ms": avg_t_draft * 1000,
        "avg_t_target_ms": avg_t_target * 1000,
        "target_single_step_ms": target_single_step * 1000,
        "speedup": real_speedup,
    }


# ==================================================================
# 主流程
# ==================================================================
def main():
    log(f"=" * 60)
    log(f"DSpark → RWKV · 阶段7 正确评估")
    log(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"DEVICE={DEVICE}  BLOCK={BLOCK}")
    log(f"=" * 60)
    log(f"\n修复 v3 报告的三个硬伤：")
    log(f"  1. 训练=验证过拟合 → 独立测试集（不同 seed / LAMBADA 后 256 条）")
    log(f"  2. 不是 draft generation → draft 自回归（每步用自己输出）")
    log(f"  3. top1 命中≠接受率 → target 验证 + greedy + MH 双指标")

    # 1. 加载模型
    model = load_v3_model()
    target = load_target()
    tokenizer = TRIE_TOKENIZER(str(VOCAB_FILE))

    # 2. 测试集 A：target 自回归（独立 seed）
    tokens_a, hids_a = gen_test_set_autoregressive(target, n_seqs=N_TEST_SEQS, seed=999)
    result_a = proper_eval(model, target, tokens_a, hids_a, "测试集 A: target 自回归 (seed=999)")

    # 3. 测试集 B：LAMBADA 真实文本（跳过训练用过的前 512 条）
    tokens_b, hids_b = gen_test_set_lambada(target, tokenizer, n_seqs=N_TEST_SEQS)
    result_b = proper_eval(model, target, tokens_b, hids_b, "测试集 B: LAMBADA 真实文本")

    # 4. 汇总对比
    log(f"\n{'='*60}")
    log(f"汇总对比")
    log(f"{'='*60}")
    log(f"\n{'测试集':<35} {'greedy avg':<12} {'MH avg':<12} {'E[len]':<10} {'加速比':<8}")
    for r in [result_a, result_b]:
        log(f"{r['test_name']:<35} {r['avg_greedy']:<12.4f} {r['avg_mh']:<12.4f} {r['expected_len']:<10.4f} {r['speedup']:<8.3f}x")

    log(f"\n--- v3 原报告（错误评估）对照 ---")
    log(f"  训练集 top1 命中: 0.9795（训练=验证，teacher forcing）")
    log(f"  → 与本评估的 greedy 接受率差异即为「过拟合 + 评估偏差」的暴露")

    log(f"\n日志已保存: {log_path}")


if __name__ == "__main__":
    main()
