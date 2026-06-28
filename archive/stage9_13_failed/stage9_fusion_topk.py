"""DSpark → RWKV · 阶段9: Logit Fusion 正确评估（top-k 命中率）

更合理的评估：
- target 的 top-5 作为"合理答案集"
- 看 draft 独立 top-1 是否落在 target top-5（draft 命中率）
- 看融合 top-1 是否落在 target top-5（融合命中率）
- 如果融合命中率 > draft 命中率，说明 target 提升 draft
"""
import math
import time
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from stage9_target_2p9b import RWKV7Target2p9B, DTYPE, DEVICE
from rwkv_tokenizer import TRIE_TOKENIZER
from stage9_logit_fusion_01b import RWKV7Draft, TARGET_WEIGHTS, DRAFT_WEIGHTS, LAMBADA_FILE


def eval_topk_hit(draft, target, tokenizer, texts, alphas, ctx_len=64, gen_len=20, k=5):
    """评估 top-k 命中率。

    对每个位置：
    - target top-k 作为合理答案集
    - draft top-1 是否在 target top-k（draft_hit）
    - 融合 top-1 是否在 target top-k（fusion_hit）
    """
    results = {a: {"hit": 0, "total": 0} for a in alphas}
    draft_hit = {"hit": 0, "total": 0}

    for text_idx, text in enumerate(texts):
        ids = tokenizer.encode(text)
        if len(ids) < ctx_len + gen_len + 10:
            continue
        ctx = ids[:ctx_len]
        gt_next = ids[ctx_len:ctx_len + gen_len]

        ctx_tensor = torch.tensor([ctx], device=DEVICE, dtype=torch.long)
        t_state = target.zero_state(1)
        d_state = draft.zero_state(1)
        t_logits, _ = target.forward(ctx_tensor, t_state, return_hidden_layers=[])
        d_logits = draft.forward(ctx_tensor, d_state)

        for t in range(gen_len):
            t_logit = t_logits[:, -1, :]
            d_logit = d_logits[:, -1, :]

            t_topk = t_logit.topk(k).indices[0]  # target top-k
            d_pred = d_logit.argmax(dim=-1).item()

            draft_hit["total"] += 1
            if d_pred in t_topk.tolist():
                draft_hit["hit"] += 1

            for a in alphas:
                t_prob = F.softmax(t_logit, dim=-1)
                d_prob = F.softmax(d_logit, dim=-1)
                fused = a * d_prob + (1 - a) * t_prob
                f_pred = fused.argmax(dim=-1).item()
                results[a]["total"] += 1
                if f_pred in t_topk.tolist():
                    results[a]["hit"] += 1

            next_tok = gt_next[t]
            next_t = torch.tensor([[next_tok]], device=DEVICE, dtype=torch.long)
            t_logits, _ = target.forward(next_t, t_state, return_hidden_layers=[])
            d_logits = draft.forward(next_t, d_state)

        if (text_idx + 1) % 20 == 0:
            print(f"  处理 {text_idx+1}/{len(texts)}")

    draft_rate = draft_hit["hit"] / max(draft_hit["total"], 1)
    fusion_rates = {a: results[a]["hit"] / max(results[a]["total"], 1) for a in alphas}
    return draft_rate, fusion_rates


def main():
    print("=" * 70)
    print("DSpark → RWKV · Logit Fusion top-k 命中率评估")
    print("  target top-5 作为合理答案集")
    print("=" * 70)

    tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))
    target = RWKV7Target2p9B(TARGET_WEIGHTS)
    draft = RWKV7Draft(DRAFT_WEIGHTS)
    print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    texts = []
    with open(LAMBADA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    texts = texts[:100]
    print(f"\n加载 {len(texts)} 条 LAMBADA")

    alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    print(f"开始评估 (ctx=64, gen=20, k=5)...")
    t0 = time.time()
    draft_rate, fusion_rates = eval_topk_hit(draft, target, tokenizer, texts, alphas, ctx_len=64, gen_len=20, k=5)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"结果: top-5 命中率 (融合 top-1 是否落在 target top-5)")
    print(f"{'-'*60}")
    print(f"{'α':>6}  {'hit_rate':>12}  {'vs draft':>12}  说明")
    print(f"{'-'*60}")
    for a in alphas:
        rate = fusion_rates[a]
        diff = rate - draft_rate
        note = ""
        if a == 0.0: note = "纯 target (应 100%)"
        elif a == 1.0: note = f"纯 draft (基线 {draft_rate:.4f})"
        else: note = f"{'提升' if diff > 0 else '下降'} {diff:+.4f}"
        print(f"{a:>6.2f}  {rate:>12.4f}  {diff:>+12.4f}  {note}")
    print(f"\n纯 draft top-5 命中率: {draft_rate:.4f}")
    print(f"耗时: {elapsed:.1f}s")
    print(f"\n解读:")
    print(f"  - 如果 α>0 时命中率 > draft_rate, 说明 fusion 提升了 draft")
    print(f"  - α=0 应为 100% (target top-1 必在自己的 top-5)")
    print(f"  - α=1 为纯 draft 基线")


if __name__ == "__main__":
    main()
