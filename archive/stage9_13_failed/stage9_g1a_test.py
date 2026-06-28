"""测试同批次 g1a-2.9B target + g1d-0.4B draft 的接受率"""
import sys
import time
import torch
from pathlib import Path
from stage9_target_2p9b import RWKV7Target2p9B, DTYPE, DEVICE
from stage9_logit_fusion_01b import RWKV7Draft
from stage9_speculative import speculative_decode, benchmark_baseline
from rwkv_tokenizer import TRIE_TOKENIZER

sys.stdout.reconfigure(line_buffering=True)

G1A_TARGET = Path(r"C:\work\niceui\g1a-2.9B.pth")
DRAFT_WEIGHTS = Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1d-0.4b-20260210-ctx8192.pth")

tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))
print("加载 g1a-2.9B target...", flush=True)
target = RWKV7Target2p9B(G1A_TARGET)
print("加载 g1d-0.4B draft...", flush=True)
draft = RWKV7Draft(DRAFT_WEIGHTS)

# 先测 target 纯生成
prompt = "什么是人工智能"
ids = tokenizer.encode(prompt)
state = target.zero_state(1)
ctx = torch.tensor([ids], device=DEVICE, dtype=torch.long)
logits, _ = target.forward(ctx, state, return_hidden_layers=[])
out = list(ids)
for _ in range(30):
    nxt = logits[0, -1].argmax().item()
    out.append(nxt)
    if nxt == 0:
        break
    nt = torch.tensor([[nxt]], device=DEVICE, dtype=torch.long)
    logits, _ = target.forward(nt, state, return_hidden_layers=[])
gen = tokenizer.decode(out[len(ids):])
print(f"\ng1a-2.9B 生成: {gen[:100]}", flush=True)

prompts = [
    "什么是人工智能",
    "中国的首都是哪里",
    "水的化学式是什么",
    "地球围绕太阳转吗",
    "一年有多少天",
]

# 基线
t0 = time.time()
for p in prompts:
    benchmark_baseline(target, tokenizer, p, n_generate=40)
base_time = (time.time() - t0) / len(prompts)
print(f"\n基线: {base_time:.2f}s/条", flush=True)

# 测试不同配置
configs = [
    (2, 0.5), (2, 1.0), (2, 2.0),
    (4, 0.5), (4, 1.0), (4, 2.0),
]

print(f"\n{'K':>3} {'α':>4} {'接受率':>10} {'fwd':>6} {'加速':>7}")
for K, alpha in configs:
    total_acc, total_draft, total_fwd = 0, 0, 0
    t0 = time.time()
    for p in prompts:
        out, stats = speculative_decode(draft, target, tokenizer, p, n_generate=40, K=K, alpha=alpha)
        total_acc += stats["accepted"]
        total_draft += stats["total_draft"]
        total_fwd += stats["target_forwards"]
    elapsed = time.time() - t0
    ar = total_acc / max(total_draft, 1)
    spec_time = elapsed / len(prompts)
    speedup = base_time / spec_time if spec_time > 0 else 0
    print(f"{K:>3} {alpha:>4.1f} {ar:>10.2%} {total_fwd/len(prompts):>6.1f} {speedup:>7.2f}x", flush=True)
