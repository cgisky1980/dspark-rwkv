"""验证 target 实现正确性：用真实文本生成，看是否通顺。"""
import torch
from stage2_target import RWKV7Target, WEIGHTS

# RWKV vocab 是 rwkv_vocab_v20230424，token 0 是 '\x00'，常见 token：
# 0=\x00, 1=' ' (空格常见), 测试用英文 token id
# 先用随机 token 看 logits 分布是否合理

target = RWKV7Target(WEIGHTS)

# 测试1：随机 token 前向，看 logits 分布
torch.manual_seed(0)
tokens = torch.randint(0, 1000, (1, 16))  # 用前 1000 token（常见 ASCII）
state = target.zero_state(1)
logits, hids = target.forward(tokens, state, return_hidden_layers=[0, 6, 11])
print(f"输入 tokens: {tokens[0].tolist()}")
print(f"logits shape: {logits.shape}")
print(f"hidden 层数: {len(hids)}, 每层 shape: {hids[0].shape}")
print(f"hidden[0] 均值/方差: {hids[0].mean():.4f}/{hids[0].std():.4f}")
print(f"hidden[1] 均值/方差: {hids[1].mean():.4f}/{hids[1].std():.4f}")
print(f"hidden[2] 均值/方差: {hids[2].mean():.4f}/{hids[2].std():.4f}")

# 测试2：自回归生成 20 token，看是否生成合理序列
print("\n自回归生成:")
cur = torch.tensor([[1]], dtype=torch.long)  # token 1 通常是空格或常见字符
state = target.zero_state(1)
gen = [1]
for _ in range(20):
    logits, _ = target.forward(cur, state, return_hidden_layers=[])
    nxt = logits[0, -1].argmax().item()
    gen.append(nxt)
    cur = torch.tensor([[nxt]], dtype=torch.long)
print(f"生成 token ids: {gen}")

# 测试3：logits 的 entropy（应该是有限的，不是 uniform）
logits_last = logits[0, -1]
probs = torch.softmax(logits_last, dim=-1)
entropy = -(probs * (probs + 1e-12).log()).sum()
print(f"最后一步 logits entropy: {entropy:.4f} (uniform={torch.log(torch.tensor(65536.0)):.4f})")
print(f"top-5 prob: {probs.topk(5).values.tolist()}")
print("验证完成")
