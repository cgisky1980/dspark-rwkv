"""测试 batch verification: batch=K T=K vs batch=1 T=K

核心问题：RWKV 的 WKV for loop 每步只处理 [B, H, N, N]。
- batch=1 T=K: 每步处理 [1, H, N, N]，GPU 利用率低，但总计算量小
- batch=K T=K: 每步处理 [K, H, N, N]，GPU 利用率高，但总计算量大 K 倍

如果 GPU 没打满，batch=K 可能和 batch=1 一样快。
"""
import time
import torch
from pathlib import Path
from stage9_target_2p9b import RWKV7Target2p9B, DTYPE, DEVICE
from stage9_logit_fusion_01b import RWKV7Draft
from rwkv_tokenizer import TRIE_TOKENIZER

G1A_TARGET = Path(r"C:\work\niceui\g1a-2.9B.pth")
DRAFT_WEIGHTS = Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1a-0.4b-20250905-ctx4096.pth")

tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))
print("加载 target...", flush=True)
target = RWKV7Target2p9B(G1A_TARGET)

ids = tokenizer.encode("什么是人工智能")
ctx = torch.tensor([ids], device=DEVICE, dtype=torch.long)

# warmup
state = target.zero_state(1)
target.forward(ctx, state, return_hidden_layers=[])
torch.cuda.synchronize()

# === 对比 batch=1 T=K vs batch=K T=K ===
print("\n=== target: batch=1 T=K vs batch=K T=K ===", flush=True)
print(f"{'K':>3} {'B=1 T=K':>12} {'B=K T=K':>12} {'ratio':>8}", flush=True)

for K in [2, 4, 8, 16]:
    tokens = ids[:K]
    # batch=1 T=K
    state = target.zero_state(1)
    seq = torch.tensor([tokens], device=DEVICE, dtype=torch.long)
    target.forward(seq, state, return_hidden_layers=[])
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        state = target.zero_state(1)
        target.forward(seq, state, return_hidden_layers=[])
    torch.cuda.synchronize()
    b1_time = (time.time() - t0) / 5

    # batch=K T=K (所有 slot 跑相同序列)
    state = target.zero_state(K)
    seqs = torch.tensor([tokens] * K, device=DEVICE, dtype=torch.long)
    target.forward(seqs, state, return_hidden_layers=[])
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        state = target.zero_state(K)
        target.forward(seqs, state, return_hidden_layers=[])
    torch.cuda.synchronize()
    bk_time = (time.time() - t0) / 5

    print(f"{K:>3} {b1_time*1000:>10.1f}ms {bk_time*1000:>10.1f}ms {bk_time/b1_time:>7.2f}x", flush=True)

# === 同样测 draft ===
print("\n加载 draft...", flush=True)
draft = RWKV7Draft(DRAFT_WEIGHTS)
state = draft.zero_state(1)
draft.forward(ctx, state)
torch.cuda.synchronize()

print("\n=== draft: batch=1 T=K vs batch=K T=K ===", flush=True)
print(f"{'K':>3} {'B=1 T=K':>12} {'B=K T=K':>12} {'ratio':>8}", flush=True)

for K in [2, 4, 8, 16]:
    tokens = ids[:K]
    state = draft.zero_state(1)
    seq = torch.tensor([tokens], device=DEVICE, dtype=torch.long)
    draft.forward(seq, state)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        state = draft.zero_state(1)
        draft.forward(seq, state)
    torch.cuda.synchronize()
    b1_time = (time.time() - t0) / 5

    state = draft.zero_state(K)
    seqs = torch.tensor([tokens] * K, device=DEVICE, dtype=torch.long)
    draft.forward(seqs, state)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        state = draft.zero_state(K)
        draft.forward(seqs, state)
    torch.cuda.synchronize()
    bk_time = (time.time() - t0) / 5

    print(f"{K:>3} {b1_time*1000:>10.1f}ms {bk_time*1000:>10.1f}ms {bk_time/b1_time:>7.2f}x", flush=True)
