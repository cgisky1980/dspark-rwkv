"""诊断：0.4B draft + 2.9B target 的 fusion 一致性
检查 fusion 后 argmax 是否与 target argmax 一致
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from stage9_target_2p9b import RWKV7Target2p9B, DTYPE, DEVICE
from rwkv_tokenizer import TRIE_TOKENIZER
from stage9_logit_fusion_01b import RWKV7Draft, TARGET_WEIGHTS, LAMBADA_FILE
import json

DRAFT_WEIGHTS = Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1d-0.4b-20260210-ctx8192.pth")

tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))
target = RWKV7Target2p9B(TARGET_WEIGHTS)
draft = RWKV7Draft(DRAFT_WEIGHTS)

# 加载 1 条 LAMBADA
texts = []
with open(LAMBADA_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            texts.append(json.loads(line)["text"])
text = texts[0]
print(f"Prompt: {text[:80]}...")

ids = tokenizer.encode(text)
ctx = ids[:64]
ctx_tensor = torch.tensor([ctx], device=DEVICE, dtype=torch.long)

# forward prompt
t_state = target.zero_state(1)
d_state = draft.zero_state(1)
t_logits_full, _ = target.forward(ctx_tensor, t_state, return_hidden_layers=[])
d_logits_full = draft.forward(ctx_tensor, d_state)

t_logits = t_logits_full[:, -1, :]  # [1, V]
d_logits = d_logits_full[:, -1, :]

# 纯 target / 纯 draft / fusion 的 top-1
t_pred = t_logits[0].argmax().item()
d_pred = d_logits[0].argmax().item()
print(f"\n位置 0（prompt 后）:")
print(f"  target top-1: {t_pred} ({tokenizer.decode([t_pred])!r})")
print(f"  draft  top-1: {d_pred} ({tokenizer.decode([d_pred])!r})")
print(f"  一致？ {t_pred == d_pred}")

# 不同 alpha 的 fusion argmax
for alpha in [0.5, 1.0, 2.0, 5.0]:
    fused = d_logits + alpha * t_logits
    f_pred = fused[0].argmax().item()
    print(f"  fusion α={alpha}: {f_pred} ({tokenizer.decode([f_pred])!r})  一致？ {f_pred == t_pred}")

# 用 ground truth 验证
gt = ids[64]
print(f"\nGround truth (ids[64]): {gt} ({tokenizer.decode([gt])!r})")
print(f"  target 正确？ {t_pred == gt}")
print(f"  draft  正确？ {d_pred == gt}")

# 统计 16 个位置的一致率
print(f"\n=== 16 位置一致率 ===")
t_state = target.zero_state(1)
d_state = draft.zero_state(1)
t_logits_full, _ = target.forward(ctx_tensor, t_state, return_hidden_layers=[])
d_logits_full = draft.forward(ctx_tensor, d_state)
t_logits = t_logits_full[:, -1, :]
d_logits = d_logits_full[:, -1, :]

match_d, match_f05, match_f1, match_f2 = 0, 0, 0, 0
total = 0
for i in range(16):
    t_pred = t_logits[0].argmax().item()
    d_pred = d_logits[0].argmax().item()
    f05 = (d_logits + 0.5 * t_logits)[0].argmax().item()
    f1 = (d_logits + 1.0 * t_logits)[0].argmax().item()
    f2 = (d_logits + 2.0 * t_logits)[0].argmax().item()
    total += 1
    if d_pred == t_pred: match_d += 1
    if f05 == t_pred: match_f05 += 1
    if f1 == t_pred: match_f1 += 1
    if f2 == t_pred: match_f2 += 1
    # 用 ground truth 推进
    gt = ids[64 + i]
    next_t = torch.tensor([[gt]], device=DEVICE, dtype=torch.long)
    t_logits_full, _ = target.forward(next_t, t_state, return_hidden_layers=[])
    d_logits_full = draft.forward(next_t, d_state)
    t_logits = t_logits_full[:, -1, :]
    d_logits = d_logits_full[:, -1, :]

print(f"  draft vs target: {match_d}/{total} = {match_d/total:.2%}")
print(f"  fusion α=0.5 vs target: {match_f05}/{total} = {match_f05/total:.2%}")
print(f"  fusion α=1.0 vs target: {match_f1}/{total} = {match_f1/total:.2%}")
print(f"  fusion α=2.0 vs target: {match_f2}/{total} = {match_f2/total:.2%}")
