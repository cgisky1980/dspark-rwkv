"""检查 ffn.value.weight 的转置"""
import torch
from pathlib import Path

path = Path(r"C:\work\niceui\rwkv7-g1h_preview4533-2.9b-20260623-ctx8192.pth.pth")
z = torch.load(path, map_location="cpu", weights_only=True)

k = "blocks.0.ffn.value.weight"
print(f"原始 {k}: {list(z[k].shape)}")
print(f"  'ffn.value.weight' in k: {'ffn.value.weight' in k}")
print(f"  转置后: {list(z[k].t().shape)}")

# 检查 ffn.key.weight
k2 = "blocks.0.ffn.key.weight"
print(f"原始 {k2}: {list(z[k2].shape)}")

# att 的
for ak in ["blocks.0.att.receptance.weight", "blocks.0.att.key.weight",
           "blocks.0.att.output.weight", "blocks.0.att.w1", "blocks.0.att.g1"]:
    print(f"原始 {ak}: {list(z[ak].shape)}")
