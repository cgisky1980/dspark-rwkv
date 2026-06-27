"""阶段2a：验证 target forward 速度，决定数据生成策略。

target 是 RWKV-7 0.1B（L=12 C=768 H=12 N=64 V=65536）。
测试不同 batch/seq_len 下的 forward 时间。
"""
import time
import torch
from stage2_target import RWKV7Target, WEIGHTS

target = RWKV7Target(WEIGHTS)
target.z = {k: v.to("cuda") for k, v in target.z.items()}
print("target 已搬到 GPU")

# 测试不同配置的 forward 时间
configs = [
    (32, 32),   # batch=32, T=32
    (64, 32),   # batch=64, T=32
    (32, 64),   # batch=32, T=64
]

for B, T in configs:
    tokens = torch.randint(0, 1000, (B, T), device="cuda")
    state = target.zero_state(B, device="cuda")
    # warmup
    with torch.no_grad():
        _ = target.forward(tokens, state, return_hidden_layers=[0, 6, 11])
    torch.cuda.synchronize()
    # 计时
    t0 = time.time()
    n_iter = 3
    with torch.no_grad():
        for _ in range(n_iter):
            state = target.zero_state(B, device="cuda")
            logits, hids = target.forward(tokens, state, return_hidden_layers=[0, 6, 11])
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n_iter
    print(f"B={B:3d} T={T:3d}: {dt*1000:.1f} ms/forward, logits={logits.shape}, hidden={len(hids)}x{hids[0].shape}")

# 测试自回归生成速度（逐 token）
print("\n自回归生成 32 token（B=32）:")
B = 32
cur = torch.randint(0, 1000, (B, 1), device="cuda")
state = target.zero_state(B, device="cuda")
t0 = time.time()
with torch.no_grad():
    for _ in range(32):
        logits, _ = target.forward(cur, state, return_hidden_layers=[])
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        cur = nxt
torch.cuda.synchronize()
dt = time.time() - t0
print(f"  {dt:.2f}s for 32 steps, {dt/32*1000:.1f} ms/step")
