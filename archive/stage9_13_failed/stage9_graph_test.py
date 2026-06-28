"""测试 CUDA graph 加速 draft/target forward

CUDA graph 把整个 forward 录制成一个 graph，replay 时直接执行 GPU kernel，
消除所有 Python 调用开销（360 次 F.linear × 0.1ms = 36ms）。
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

# ============ 1. draft T=1: 普通 vs CUDA graph ============
print("\n=== draft T=1: 普通 vs CUDA graph ===", flush=True)
d_state = draft.zero_state(1)
tok = torch.tensor([[ids[0]]], device=DEVICE, dtype=torch.long)
# warmup
for _ in range(3):
    draft.forward(tok, d_state)
torch.cuda.synchronize()

# 普通
t0 = time.time()
for _ in range(20):
    draft.forward(tok, d_state)
torch.cuda.synchronize()
normal_time = (time.time() - t0) / 20
print(f"  普通: {normal_time*1000:.1f}ms", flush=True)

# CUDA graph
d_state_g = draft.zero_state(1)
tok_g = tok.clone()
# warmup
for _ in range(3):
    draft.forward(tok_g, d_state_g)
torch.cuda.synchronize()

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    draft.forward(tok_g, d_state_g)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(20):
    g.replay()
torch.cuda.synchronize()
graph_time = (time.time() - t0) / 20
print(f"  CUDA graph: {graph_time*1000:.1f}ms", flush=True)
print(f"  加速: {normal_time/graph_time:.2f}x", flush=True)

# ============ 2. target T=1: 普通 vs CUDA graph ============
print("\n=== target T=1: 普通 vs CUDA graph ===", flush=True)
t_state = target.zero_state(1)
# warmup
for _ in range(3):
    target.forward(tok, t_state, return_hidden_layers=[])
torch.cuda.synchronize()

t0 = time.time()
for _ in range(20):
    target.forward(tok, t_state, return_hidden_layers=[])
torch.cuda.synchronize()
normal_time = (time.time() - t0) / 20
print(f"  普通: {normal_time*1000:.1f}ms", flush=True)

t_state_g = target.zero_state(1)
tok_g2 = tok.clone()
for _ in range(3):
    target.forward(tok_g2, t_state_g, return_hidden_layers=[])
torch.cuda.synchronize()

g2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g2):
    target.forward(tok_g2, t_state_g, return_hidden_layers=[])
torch.cuda.synchronize()

t0 = time.time()
for _ in range(20):
    g2.replay()
torch.cuda.synchronize()
graph_time = (time.time() - t0) / 20
print(f"  CUDA graph: {graph_time*1000:.1f}ms", flush=True)
print(f"  加速: {normal_time/graph_time:.2f}x", flush=True)

# ============ 3. target T=4: 普通 vs CUDA graph ============
print("\n=== target T=4: 普通 vs CUDA graph ===", flush=True)
seq4 = torch.tensor([ids[:4]], device=DEVICE, dtype=torch.long)
t_state = target.zero_state(1)
for _ in range(3):
    target.forward(seq4, t_state, return_hidden_layers=[])
torch.cuda.synchronize()

t0 = time.time()
for _ in range(20):
    target.forward(seq4, t_state, return_hidden_layers=[])
torch.cuda.synchronize()
normal_time = (time.time() - t0) / 20
print(f"  普通: {normal_time*1000:.1f}ms", flush=True)

t_state_g = target.zero_state(1)
seq4_g = seq4.clone()
for _ in range(3):
    target.forward(seq4_g, t_state_g, return_hidden_layers=[])
torch.cuda.synchronize()

g3 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g3):
    target.forward(seq4_g, t_state_g, return_hidden_layers=[])
torch.cuda.synchronize()

t0 = time.time()
for _ in range(20):
    g3.replay()
torch.cuda.synchronize()
graph_time = (time.time() - t0) / 20
print(f"  CUDA graph: {graph_time*1000:.1f}ms", flush=True)
print(f"  加速: {normal_time/graph_time:.2f}x", flush=True)
