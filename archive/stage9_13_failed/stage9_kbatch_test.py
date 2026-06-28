"""测试 K 个 batch slot 并行验证 vs 单条 T=K

方案对比：
- 方案 A（当前）：单条 T=K，一次 forward，WKV for-loop 内部串行 K 步
- 方案 B（用户建议）：K 个 batch slot，slot i 跑 [t1..ti]（长度 i），共用初始 state
  - slot 0: T=1 跑 [t1]
  - slot 1: T=2 跑 [t1, t2]
  - slot 2: T=3 跑 [t1, t2, t3]
  - slot 3: T=4 跑 [t1, t2, t3, t4]
  - 一次 forward，所有 slot 并行

关键问题：
1. 方案 B 是否比方案 A 快？（GPU 并行度）
2. 方案 B 是否能直接得到 K 个 state？（不用 rollback）
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

K = 4
draft_tokens = ids[:K]

# === 方案 A: 单条 T=K ===
print(f"\n=== 方案 A: 单条 T={K} ===", flush=True)
state_a = target.zero_state(1)
seq_a = torch.tensor([draft_tokens], device=DEVICE, dtype=torch.long)
# warmup
target.forward(seq_a, state_a, return_hidden_layers=[])
torch.cuda.synchronize()

t0 = time.time()
for _ in range(10):
    state_a = target.zero_state(1)
    target.forward(seq_a, state_a, return_hidden_layers=[])
torch.cuda.synchronize()
time_a = (time.time() - t0) / 10
print(f"  耗时: {time_a*1000:.1f}ms", flush=True)

# === 方案 B: K 个 batch slot, slot i 跑 [t1..ti] ===
print(f"\n=== 方案 B: {K} 个 batch slot, slot i 跑 [t1..t{{i+1}}] ===", flush=True)

# 构造 batch: 每个 slot 跑不同长度的前缀
# 但 forward 要求所有 slot T 相同，所以用 padding
# slot 0: [t1, pad, pad, pad] T=4 实际只看 t1
# slot 1: [t1, t2, pad, pad] T=4 实际看 t1,t2
# 但 RWKV 会处理 pad token，state 会错
# 正确方案：每个 slot T=i，不能用 batch
# 或者：slot i 跑 [t1..ti]，所有 slot T=K，但用 attention mask？
# RWKV 是 RNN，没有 attention mask，只能用不同 T

# 实际方案：每个 slot 单独 forward T=i
# 但这就不是 batch 了

# 或者：所有 slot 都 T=K，但 slot i 只取前 i 个 token 的结果
# state 会是看到全部 K 个 token 后的 state，不是看到 i 个 token 的 state
# 这不行

# 正确方案 B：slot i 跑 [t1..ti] T=i
# 不能用 batch（T 不同）
# 只能串行 K 次 forward
# 但每次 T 不同：T=1, T=2, T=3, T=4
print("  方案 B 实现：每个 slot 单独 forward T=i", flush=True)

t0 = time.time()
for _ in range(10):
    for i in range(K):
        state_b = target.zero_state(1)
        seq_b = torch.tensor([draft_tokens[:i+1]], device=DEVICE, dtype=torch.long)
        target.forward(seq_b, state_b, return_hidden_layers=[])
torch.cuda.synchronize()
time_b = (time.time() - t0) / 10
print(f"  耗时: {time_b*1000:.1f}ms", flush=True)

# === 方案 C: 所有 slot T=K, 但每个 slot 从相同 state 开始 ===
# 这就是 batch=K T=K, 所有 slot 跑相同序列
# 但每个 slot 的 state 是独立的（从相同初始 state 开始）
print(f"\n=== 方案 C: batch={K} T={K}, 所有 slot 跑相同序列 ===", flush=True)
state_c = target.zero_state(K)
seqs_c = torch.tensor([draft_tokens] * K, device=DEVICE, dtype=torch.long)
target.forward(seqs_c, state_c, return_hidden_layers=[])
torch.cuda.synchronize()

t0 = time.time()
for _ in range(10):
    state_c = target.zero_state(K)
    target.forward(seqs_c, state_c, return_hidden_layers=[])
torch.cuda.synchronize()
time_c = (time.time() - t0) / 10
print(f"  耗时: {time_c*1000:.1f}ms", flush=True)

# === 方案 D: 每个槽跑不同前缀，但 T=K, 用 padding + mask ===
# RWKV 是 RNN, pad token 会影响 state
# 除非用 "stop" token, 否则不行
# 但如果我们只关心 logits[0, i-1]（看到前 i 个 token 后的预测），可以用 padding
# state 会错，但 logits 可能对
# 不行，state 错了后续就全错了

# === 结论 ===
print(f"\n=== 结论 ===", flush=True)
print(f"  方案 A (单条 T=K): {time_a*1000:.1f}ms", flush=True)
print(f"  方案 B (K 次串行 T=i): {time_b*1000:.1f}ms", flush=True)
print(f"  方案 C (batch=K T=K): {time_c*1000:.1f}ms", flush=True)
print(f"  A vs C: {time_a/time_c:.2f}x", flush=True)

# 关键问题：方案 A 的 state 是看到 [t1..tK] 后的 state
# 方案 B 的 slot i 的 state 是看到 [t1..ti] 后的 state
# 方案 C 的每个 slot 的 state 都是看到 [t1..tK] 后的 state（相同输入）
# 所以方案 A 和 C 都不能直接得到 slot i 的 state

# 但方案 A 的 forward 内部，每个 token 位置都计算了 state
# 如果能让 forward 返回中间 state，就解决了
print(f"\n=== 关键：让 forward 返回中间 state ===", flush=True)

# 测试：修改 forward 返回每个 token 位置的 wkv_state
state_d = target.zero_state(1)
seq_d = torch.tensor([draft_tokens], device=DEVICE, dtype=torch.long)

# 手动模拟 forward 内部 state 变化
# 先看 tmix 的 wkv_state 是否在 forward 后保留中间值
print("  forward 后 state 是看到全部 token 后的 state", flush=True)
print("  要得到中间 state, 需要 forward 内部保存每步的 state", flush=True)
