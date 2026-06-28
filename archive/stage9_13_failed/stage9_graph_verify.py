"""验证 CUDA graph 能否处理动态 input（tokens 内容变化，shape 不变）

关键：CUDA graph 用 tensor 的内存地址，replay 时如果 tensor 内容变了（copy_），
graph 会用新内容。验证 forward 结果是否正确。
"""
import torch
from pathlib import Path
from stage9_target_2p9b import RWKV7Target2p9B, DTYPE, DEVICE
from stage9_logit_fusion_01b import RWKV7Draft
from rwkv_tokenizer import TRIE_TOKENIZER

G1A_TARGET = Path(r"C:\work\niceui\g1a-2.9B.pth")
DRAFT_WEIGHTS = Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1a-0.4b-20250905-ctx4096.pth")

tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))
print("加载 draft...", flush=True)
draft = RWKV7Draft(DRAFT_WEIGHTS)

ids = tokenizer.encode("什么是人工智能")

# 准备 buffer
tok_buf = torch.tensor([[ids[0]]], device=DEVICE, dtype=torch.long)
d_state = draft.zero_state(1)

# warmup
for _ in range(3):
    draft.forward(tok_buf, d_state)
torch.cuda.synchronize()

# 录制 graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out_g = draft.forward(tok_buf, d_state)
torch.cuda.synchronize()

# 测试 1: 用相同 token，对比 graph vs 普通 forward
print("=== 测试 1: 相同 token，graph vs 普通 ===", flush=True)
d_state_normal = draft.zero_state(1)
tok_normal = torch.tensor([[ids[0]]], device=DEVICE, dtype=torch.long)
out_normal = draft.forward(tok_normal, d_state_normal)
torch.cuda.synchronize()

# graph replay（state 已被上面的 forward 更新，需要重置）
# 注意：graph 录制时 state 已经被更新了，replay 会从当前 state 继续
# 为了对比，需要重置 state
d_state[0].zero_()
d_state[1].zero_()
g.replay()
torch.cuda.synchronize()

print(f"  普通 out[0,0,:5]: {out_normal[0,0,:5].tolist()}", flush=True)
print(f"  graph out[0,0,:5]: {out_g[0,0,:5].tolist()}", flush=True)
print(f"  差异: {(out_normal - out_g).abs().max().item():.6f}", flush=True)

# 测试 2: 改变 tok_buf 内容，replay，看是否用新 token
print("\n=== 测试 2: 改变 tok_buf 内容 ===", flush=True)
d_state_normal2 = draft.zero_state(1)
tok_normal2 = torch.tensor([[ids[1]]], device=DEVICE, dtype=torch.long)
out_normal2 = draft.forward(tok_normal2, d_state_normal2)
torch.cuda.synchronize()

# graph: 改变 tok_buf 内容
tok_buf.copy_(torch.tensor([[ids[1]]], device=DEVICE, dtype=torch.long))
d_state[0].zero_()
d_state[1].zero_()
g.replay()
torch.cuda.synchronize()

print(f"  普通 out[0,0,:5]: {out_normal2[0,0,:5].tolist()}", flush=True)
print(f"  graph out[0,0,:5]: {out_g[0,0,:5].tolist()}", flush=True)
print(f"  差异: {(out_normal2 - out_g).abs().max().item():.6f}", flush=True)

# 测试 3: 连续 replay（模拟 draft 生成 K 个候选）
print("\n=== 测试 3: 连续 replay 生成 4 个候选 ===", flush=True)
# 普通：连续 4 次 forward
d_state_n = draft.zero_state(1)
ctx = torch.tensor([ids], device=DEVICE, dtype=torch.long)
draft.forward(ctx, d_state_n)  # prompt forward
normal_tokens = []
normal_logits = draft.forward(torch.tensor([[ids[0]]], device=DEVICE, dtype=torch.long), d_state_n)
for i in range(4):
    tok = normal_logits[0, -1].argmax().item()
    normal_tokens.append(tok)
    normal_logits = draft.forward(torch.tensor([[tok]], device=DEVICE, dtype=torch.long), d_state_n)
torch.cuda.synchronize()
print(f"  普通生成: {normal_tokens}", flush=True)

# graph：连续 replay 4 次
d_state[0].zero_()
d_state[1].zero_()
draft.forward(ctx, d_state)  # prompt forward（普通）
graph_tokens = []
# 第一次用 ids[0] 作为 input
tok_buf.copy_(torch.tensor([[ids[0]]], device=DEVICE, dtype=torch.long))
g.replay()
torch.cuda.synchronize()
for i in range(4):
    tok = out_g[0, -1].argmax().item()
    graph_tokens.append(tok)
    tok_buf.copy_(torch.tensor([[tok]], device=DEVICE, dtype=torch.long))
    g.replay()
    torch.cuda.synchronize()
print(f"  graph生成: {graph_tokens}", flush=True)
print(f"  一致: {normal_tokens == graph_tokens}", flush=True)
