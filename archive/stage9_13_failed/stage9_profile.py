"""Profiling: 定位 speculative decoding 的时间瓶颈

测各部分耗时：
1. target forward (T tokens) — WKV for-loop 串行
2. draft forward (1 token) — 单步
3. state clone/backup 开销
4. rejection sampling 开销
"""
import time
import torch
import torch.nn.functional as F
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

prompt = "什么是人工智能"
ids = tokenizer.encode(prompt)

# ============ 1. target forward 不同 T 的耗时 ============
print("\n=== target forward 耗时 (不同 T) ===", flush=True)
for T in [1, 2, 4, 8, 16, 32]:
    state = target.zero_state(1)
    ctx = torch.tensor([ids[:T]], device=DEVICE, dtype=torch.long)
    # warmup
    target.forward(ctx, state, return_hidden_layers=[])
    torch.cuda.synchronize()

    state = target.zero_state(1)
    t0 = time.time()
    for _ in range(5):
        target.forward(ctx, state, return_hidden_layers=[])
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / 5
    print(f"  T={T:>3}: {elapsed*1000:.1f}ms ({elapsed/T*1000:.1f}ms/token)", flush=True)

# ============ 2. draft forward 不同 T 的耗时 ============
print("\n=== draft forward 耗时 (不同 T) ===", flush=True)
for T in [1, 2, 4, 8, 16, 32]:
    state = draft.zero_state(1)
    ctx = torch.tensor([ids[:T]], device=DEVICE, dtype=torch.long)
    # warmup
    draft.forward(ctx, state)
    torch.cuda.synchronize()

    state = draft.zero_state(1)
    t0 = time.time()
    for _ in range(5):
        draft.forward(ctx, state)
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / 5
    print(f"  T={T:>3}: {elapsed*1000:.1f}ms ({elapsed/T*1000:.1f}ms/token)", flush=True)

# ============ 3. state clone 开销 ============
print("\n=== state clone 开销 ===", flush=True)
t_state = target.zero_state(1)
d_state = draft.zero_state(1)
t0 = time.time()
for _ in range(20):
    backup = [s.clone() for s in t_state]
torch.cuda.synchronize()
print(f"  target state clone: {(time.time()-t0)/20*1000:.1f}ms", flush=True)

t0 = time.time()
for _ in range(20):
    backup = [s.clone() for s in d_state]
torch.cuda.synchronize()
print(f"  draft state clone: {(time.time()-t0)/20*1000:.1f}ms", flush=True)

# ============ 4. 模拟一轮 speculative decoding ============
print("\n=== 模拟一轮 spec decode (K=4) ===", flush=True)
K = 4
# prompt forward
t_state = target.zero_state(1)
d_state = draft.zero_state(1)
ctx = torch.tensor([ids], device=DEVICE, dtype=torch.long)
target.forward(ctx, t_state, return_hidden_layers=[])
draft.forward(ctx, d_state)

# draft 生成 K 个候选
torch.cuda.synchronize()
t0 = time.time()
d_backup = [s.clone() for s in d_state]
for i in range(K):
    tok = torch.tensor([[ids[0]]], device=DEVICE, dtype=torch.long)
    draft.forward(tok, d_state)
torch.cuda.synchronize()
draft_gen_time = time.time() - t0

# target 批量验证
torch.cuda.synchronize()
t0 = time.time()
t_backup = [s.clone() for s in t_state]
draft_tokens = torch.tensor([[ids[0]]*K], device=DEVICE, dtype=torch.long)
target.forward(draft_tokens, t_state, return_hidden_layers=[])
torch.cuda.synchronize()
target_verify_time = time.time() - t0

# replay (rollback + forward)
torch.cuda.synchronize()
t0 = time.time()
t_state[0].copy_(t_backup[0])
t_state[1].copy_(t_backup[1])
replay = torch.tensor([[ids[0]]*2], device=DEVICE, dtype=torch.long)
target.forward(replay, t_state, return_hidden_layers=[])
torch.cuda.synchronize()
replay_time = time.time() - t0

print(f"  draft 生成 {K} 候选: {draft_gen_time*1000:.1f}ms", flush=True)
print(f"  target 验证 {K} 候选: {target_verify_time*1000:.1f}ms", flush=True)
print(f"  target rollback+replay: {replay_time*1000:.1f}ms", flush=True)
print(f"  总计: {(draft_gen_time+target_verify_time+replay_time)*1000:.1f}ms", flush=True)

# ============ 5. 基线对比: 纯 target 自回归 4 个 token ============
print("\n=== 基线: 纯 target 4 个 token 自回归 ===", flush=True)
t_state = target.zero_state(1)
ctx = torch.tensor([ids], device=DEVICE, dtype=torch.long)
target.forward(ctx, t_state, return_hidden_layers=[])
torch.cuda.synchronize()
t0 = time.time()
for i in range(4):
    tok = torch.tensor([[ids[0]]], device=DEVICE, dtype=torch.long)
    target.forward(tok, t_state, return_hidden_layers=[])
torch.cuda.synchronize()
print(f"  4× 单 token forward: {(time.time()-t0)*1000:.1f}ms", flush=True)
