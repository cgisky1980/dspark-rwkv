"""测试 CUDA stream 并行：draft 和 target 能否真正并行执行

对比：
1. 串行: target.forward + draft.forward
2. 并行: target 在 stream1, draft 在 stream2
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
print("加载 draft...", flush=True)
draft = RWKV7Draft(DRAFT_WEIGHTS)

ids = tokenizer.encode("什么是人工智能")
ctx = torch.tensor([ids], device=DEVICE, dtype=torch.long)

# warmup
t_state = target.zero_state(1)
d_state = draft.zero_state(1)
target.forward(ctx, t_state, return_hidden_layers=[])
draft.forward(ctx, d_state)
torch.cuda.synchronize()

# === 1. 串行：target T=4 + draft T=4 ===
print("\n=== 串行: target T=4 + draft T=4 ===", flush=True)
N = 5
t0 = time.time()
for _ in range(N):
    ts = target.zero_state(1)
    ds = draft.zero_state(1)
    target.forward(ctx, ts, return_hidden_layers=[])
    draft.forward(ctx, ds)
torch.cuda.synchronize()
serial_time = (time.time() - t0) / N
print(f"  串行: {serial_time*1000:.1f}ms", flush=True)

# === 2. 并行：target 在 stream1, draft 在 stream2 ===
print("\n=== 并行: target stream1 + draft stream2 ===", flush=True)
target_stream = torch.cuda.Stream()
draft_stream = torch.cuda.Stream()

t0 = time.time()
for _ in range(N):
    ts = target.zero_state(1)
    ds = draft.zero_state(1)
    with torch.cuda.stream(target_stream):
        target.forward(ctx, ts, return_hidden_layers=[])
    with torch.cuda.stream(draft_stream):
        draft.forward(ctx, ds)
    torch.cuda.synchronize()
parallel_time = (time.time() - t0) / N
print(f"  并行: {parallel_time*1000:.1f}ms", flush=True)
print(f"  加速: {serial_time/parallel_time:.2f}x", flush=True)

# === 3. draft 生成 4 候选（串行 K 次 T=1）+ target 验证（1 次 T=4）===
print("\n=== 场景: draft 4×T=1 + target T=4 ===", flush=True)
# 串行
t0 = time.time()
for _ in range(N):
    ds = draft.zero_state(1)
    draft.forward(ctx, ds)
    for i in range(4):
        tok = torch.tensor([[ids[0]]], device=DEVICE, dtype=torch.long)
        draft.forward(tok, ds)
    ts = target.zero_state(1)
    target.forward(ctx, ts, return_hidden_layers=[])
    draft_tokens = torch.tensor([[ids[0]]*4], device=DEVICE, dtype=torch.long)
    target.forward(draft_tokens, ts, return_hidden_layers=[])
torch.cuda.synchronize()
serial_full = (time.time() - t0) / N
print(f"  串行: {serial_full*1000:.1f}ms", flush=True)

# 并行：draft 生成 4 候选 + target prompt forward 同时跑
t0 = time.time()
for _ in range(N):
    ts = target.zero_state(1)
    ds = draft.zero_state(1)
    # target prompt forward 在 target_stream
    with torch.cuda.stream(target_stream):
        target.forward(ctx, ts, return_hidden_layers=[])
    # draft prompt + 4 候选 在 draft_stream
    with torch.cuda.stream(draft_stream):
        draft.forward(ctx, ds)
        for i in range(4):
            tok = torch.tensor([[ids[0]]], device=DEVICE, dtype=torch.long)
            draft.forward(tok, ds)
    torch.cuda.synchronize()
    # target 验证（依赖 draft 结果，串行）
    draft_tokens = torch.tensor([[ids[0]]*4], device=DEVICE, dtype=torch.long)
    target.forward(draft_tokens, ts, return_hidden_layers=[])
torch.cuda.synchronize()
parallel_full = (time.time() - t0) / N
print(f"  并行(prompt+draft||target): {parallel_full*1000:.1f}ms", flush=True)
print(f"  加速: {serial_full/parallel_full:.2f}x", flush=True)
