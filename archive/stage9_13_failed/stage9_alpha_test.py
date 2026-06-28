"""测试不同 alpha 和 K 的组合"""
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
print("加载 models...", flush=True)
target = RWKV7Target2p9B(TARGET_WEIGHTS)
draft = RWKV7Draft(Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1d-0.4b-20260210-ctx8192.pth"))

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
    (2, 0.1), (2, 0.5), (2, 1.0), (2, 2.0),
    (4, 0.5), (4, 1.0), (4, 2.0),
]

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
    print(f"K={K} α={alpha:.1f}: 接受率={ar:.2%} ({total_acc}/{total_draft}) "
          f"fwd={total_fwd/len(prompts):.1f} 加速={speedup:.2f}x", flush=True)
