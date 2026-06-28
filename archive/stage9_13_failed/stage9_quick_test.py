"""快速测试投机解码接受率（只测简单 prompt）"""
import sys
import time
import torch
from pathlib import Path
from stage9_target_2p9b import RWKV7Target2p9B, DTYPE, DEVICE
from stage9_logit_fusion_01b import RWKV7Draft, TARGET_WEIGHTS
from stage9_speculative import speculative_decode, benchmark_baseline
from rwkv_tokenizer import TRIE_TOKENIZER

sys.stdout.reconfigure(line_buffering=True)

tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))
print("加载 target...", flush=True)
target = RWKV7Target2p9B(TARGET_WEIGHTS)
print("加载 draft...", flush=True)
draft = RWKV7Draft(Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1d-0.4b-20260210-ctx8192.pth"))

prompts = [
    "什么是人工智能",
    "中国的首都是哪里",
    "水的化学式是什么",
    "地球围绕太阳转吗",
    "一年有多少天",
]

K, alpha = 2, 0.5
print(f"\n=== K={K}, α={alpha} ===", flush=True)
total_acc, total_draft, total_fwd = 0, 0, 0
t0 = time.time()
for p in prompts:
    out, stats = speculative_decode(draft, target, tokenizer, p, n_generate=40, K=K, alpha=alpha)
    total_acc += stats["accepted"]
    total_draft += stats["total_draft"]
    total_fwd += stats["target_forwards"]
    gen = tokenizer.decode(out[len(tokenizer.encode(p)):])
    print(f"  {p} -> {gen[:60]}", flush=True)
elapsed = time.time() - t0
print(f"接受率: {total_acc}/{total_draft} = {total_acc/max(total_draft,1):.2%}", flush=True)
print(f"平均 target forwards: {total_fwd/len(prompts):.1f} (基线 40)", flush=True)
print(f"耗时: {elapsed:.1f}s", flush=True)

# 基线时间
print("\n=== 基线 ===", flush=True)
t0 = time.time()
for p in prompts:
    benchmark_baseline(target, tokenizer, p, n_generate=40)
base_time = (time.time() - t0) / len(prompts)
print(f"基线平均: {base_time:.2f}s/条", flush=True)
print(f"投机平均: {elapsed/len(prompts):.2f}s/条", flush=True)
print(f"加速比: {base_time/(elapsed/len(prompts)):.2f}x", flush=True)
