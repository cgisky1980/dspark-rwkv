"""对比我们的实现和 albatross CUDA 实现的输出"""
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# 添加 albatross 路径
sys.path.insert(0, str(Path(__file__).parent / "albatross_ref"))

DEVICE = "cuda"
DTYPE = torch.float16

# 加载权重
WEIGHTS = Path(r"C:\work\niceui\rwkv7-g1h_preview4533-2.9b-20260623-ctx8192.pth.pth")
z = torch.load(WEIGHTS, map_location="cpu", weights_only=True)

# 检查 r_k
r_k_raw = z["blocks.0.att.r_k"].squeeze()
H, N = r_k_raw.shape
print(f"H={H}, N={N}")

# albatross 的权重处理（line 119-120）
keys = list(z.keys())
for k in keys:
    v = z[k].squeeze()
    if 'att.g1' in k or 'att.g2' in k or 'att.a1' in k or 'att.a2' in k or 'att.w1' in k or 'att.w2' in k or 'att.v1' in k or 'att.v2' in k or 'ffn.value.weight' in k:
        v = v.t()
    if k.endswith("att.r_k"):
        v = v.flatten()
    z[k] = v.to(DTYPE).contiguous()

# emb ln0
z["emb.weight"] = F.layer_norm(z["emb.weight"], (H*N,), z["blocks.0.ln0.weight"], z["blocks.0.ln0.bias"])
z = {k: v.to(DEVICE) for k, v in z.items()}

# 测试输入
prompt = "What is"
from rwkv_tokenizer import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))
ids = tokenizer.encode(prompt)
print(f"ids: {ids}")

# 我们的纯 PyTorch 实现
sys.path.insert(0, str(Path(__file__).parent))
from stage9_target_2p9b import RWKV7Target2p9B, group_norm, lerp, layer_norm

target = RWKV7Target2p9B(WEIGHTS)
state_ours = target.zero_state(1)
ctx = torch.tensor([ids], device=DEVICE, dtype=torch.long)
logits_ours, _ = target.forward(ctx, state_ours, return_hidden_layers=[])
print(f"\n我们的 logits shape: {logits_ours.shape}")
print(f"  top-5: {logits_ours[0, -1].topk(5).indices.tolist()}")
print(f"  解码: {tokenizer.decode(logits_ours[0, -1].topk(5).indices.tolist())}")

# 尝试加载 albatross（需要 CUDA op）
print("\n尝试加载 albatross CUDA op...")
try:
    from rwkv7 import RWKV_x070, RWKV_x070_TMix_seq, RWKV7_SEQ_OP
    print("albatross 加载成功！")

    # 用 albatross 的 TMix_seq 测试第 0 层
    state_alba = [
        torch.zeros(32, 2, 1, H*N, dtype=DTYPE, device=DEVICE),
        torch.zeros(32, 1, H, N, N, dtype=DTYPE, device=DEVICE),
        torch.zeros(1, dtype=torch.int32, device=DEVICE),
    ]

    x = z["emb.weight"][ctx]  # [1, T, C]
    print(f"x shape: {x.shape}")

    # albatross 的 TMix_seq
    i = 0
    p = f"blocks.{i}."
    att = p + "att."
    xx = F.layer_norm(x, (H*N,), z[p+"ln1.weight"], z[p+"ln1.bias"])
    v_first = torch.zeros_like(xx)

    xx_out, v_first_out = RWKV_x070_TMix_seq(
        i, H, N, xx[0], state_alba[0][i], v_first[0], state_alba[1][i],
        z[att+"x_r"], z[att+"x_w"], z[att+"x_k"], z[att+"x_v"], z[att+"x_a"], z[att+"x_g"],
        z[att+"w0"], z[att+"w1"], z[att+"w2"], z[att+"a0"], z[att+"a1"], z[att+"a2"],
        z[att+"v0"], z[att+"v1"], z[att+"v2"], z[att+"g1"], z[att+"g2"],
        z[att+"k_k"], z[att+"k_a"], z[att+"r_k"],
        z[att+"receptance.weight"], z[att+"key.weight"], z[att+"value.weight"], z[att+"output.weight"],
        z[att+"ln_x.weight"], z[att+"ln_x.bias"], state_alba[2]
    )
    print(f"albatross 第0层输出 shape: {xx_out.shape}")
    print(f"  前5个值: {xx_out[0, :5].tolist()}")

    # 我们的第 0 层
    state_ours2 = target.zero_state(1)
    x2 = z["emb.weight"][ctx]
    xx2 = layer_norm(x2, z[p+"ln1.weight"], z[p+"ln1.bias"])
    v_first2 = torch.zeros_like(xx2)
    xx2_out, v_first2_out = target.tmix(0, xx2, state_ours2[0][0], v_first2, state_ours2[1][0], att)
    print(f"\n我们的第0层输出 shape: {xx2_out.shape}")
    print(f"  前5个值: {xx2_out[0, 0, :5].tolist()}")

    # 差异
    diff = (xx_out - xx2_out[0]).abs().mean()
    print(f"\n平均差异: {diff.item():.6f}")

except Exception as e:
    print(f"albatross 加载失败: {e}")
    import traceback
    traceback.print_exc()
