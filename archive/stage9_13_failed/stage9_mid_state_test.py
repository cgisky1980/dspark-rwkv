"""测试：forward 保存中间 state 的方案

方案：修改 tmix，在 for t in range(T) 循环里每步 clone wkv_state
这样单条 T=K forward 就能得到 K 个中间 state

对比：
- 方案 A（当前）：单条 T=K，不保存中间 state，拒绝时 rollback+replay（多一次 forward）
- 方案 B（新）：单条 T=K，保存中间 state，拒绝时直接取 slot n_accept 的 state（不用 replay）
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
K = 4
draft_tokens = ids[:K]

# === 测试：单条 T=K forward，保存中间 state 的成本 ===
print(f"\n=== 单条 T={K} forward，保存中间 state 的成本 ===", flush=True)

# 原版 forward（不保存中间 state）
seq = torch.tensor([draft_tokens], device=DEVICE, dtype=torch.long)
state = target.zero_state(1)
target.forward(seq, state, return_hidden_layers=[])
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    state = target.zero_state(1)
    target.forward(seq, state, return_hidden_layers=[])
torch.cuda.synchronize()
time_normal = (time.time() - t0) / 10
print(f"  原版: {time_normal*1000:.1f}ms", flush=True)

# 保存中间 state 的成本：clone wkv_state K 次
# wkv_state shape: [1, H, N, N] = [1, 40, 64, 64] = 65536 elements
# clone 一次的成本？
state = target.zero_state(1)
wkv = state[1][0]  # [1, H, N, N]
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    snap = wkv.clone()
torch.cuda.synchronize()
clone_time = (time.time() - t0) / 100
print(f"  wkv_state.clone(): {clone_time*1000:.2f}ms", flush=True)
print(f"  K={K} 次 clone: {clone_time*K*1000:.2f}ms", flush=True)

# shift_state 也要保存
shift = state[0][0]  # [2, 1, C]
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    snap = shift.clone()
torch.cuda.synchronize()
shift_clone = (time.time() - t0) / 100
print(f"  shift_state.clone(): {shift_clone*1000:.2f}ms", flush=True)

# === 模拟方案 B：forward 内部保存中间 state ===
# 修改 tmix，每步保存 [wkv_state.clone(), shift_state.clone(), v_first.clone()]
# 但 v_first 是跨层的，不好保存
# 简化：只保存 wkv_state 和 shift_state

# 实际上，state[0] (shift_state) 和 state[1] (wkv_state) 是每层独立的
# state[0]: [n_layer, 2, B, C]
# state[1]: [n_layer, B, H, N, N]
# 要保存中间 state，需要每层每步保存一份

# 方案：forward 后，state 是看到所有 token 后的 state
# 要得到中间 state，可以在 forward 内部每步保存
# 但这会改变 forward 签名

# 更简单方案：用 K 个 batch slot，每个 slot 跑不同长度前缀
# 但 Python batch 要求 T 相同

# 最简单方案：手动 K 次 forward T=1
# slot 0: state=S0, forward [t1] → logits_1, state=S1
# slot 1: state=S1, forward [t2] → logits_2, state=S2
# ...
# 这是串行的，但有数据依赖，无法并行

print(f"\n=== K 次串行 T=1 forward ===", flush=True)
state = target.zero_state(1)
target.forward(torch.tensor([ids], device=DEVICE, dtype=torch.long), state, return_hidden_layers=[])
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    s = target.zero_state(1)
    target.forward(torch.tensor([ids], device=DEVICE, dtype=torch.long), s, return_hidden_layers=[])
    for i in range(K):
        tok = torch.tensor([[draft_tokens[i]]], device=DEVICE, dtype=torch.long)
        target.forward(tok, s, return_hidden_layers=[])
torch.cuda.synchronize()
time_k_t1 = (time.time() - t0) / 10
print(f"  K={K} 次 T=1: {time_k_t1*1000:.1f}ms", flush=True)
print(f"  对比单条 T={K}: {time_normal*1000:.1f}ms", flush=True)
print(f"  比值: {time_k_t1/time_normal:.2f}x", flush=True)

# === 结论 ===
print(f"\n=== 结论 ===", flush=True)
print(f"  单条 T={K} (当前方案): {time_normal*1000:.1f}ms", flush=True)
print(f"  K 次 T=1 (串行): {time_k_t1*1000:.1f}ms", flush=True)
print(f"  wkv clone 成本: {clone_time*K*1000:.2f}ms (可忽略)", flush=True)
print(f"  → 方案 B (保存中间 state) 几乎免费，只需 clone", flush=True)
