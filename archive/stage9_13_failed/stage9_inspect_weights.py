"""对比 0.1B 和 0.4B 权重结构，找出 g1d 加载问题"""
import torch
from pathlib import Path

PATHS = {
    "0.1B": Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-0.1b.pth"),
    "0.4B g1d": Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1d-0.4b-20260210-ctx8192.pth"),
    "0.4B g1a": Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1a-0.4b-20250905-ctx4096.pth"),
    "2.9B target": Path(r"C:\work\niceui\rwkv7-g1h_preview4533-2.9b-20260623-ctx8192.pth.pth"),
}

for name, path in PATHS.items():
    if not path.exists():
        print(f"{name}: 文件不存在 {path}")
        continue
    print(f"\n=== {name} ({path.name}) ===")
    z = torch.load(path, map_location="cpu", weights_only=True)
    keys = list(z.keys())
    print(f"总 keys: {len(keys)}")

    # 基本结构
    max_layer = -1
    for k in keys:
        if k.startswith("blocks."):
            max_layer = max(max_layer, int(k.split(".")[1]))
    print(f"层数: {max_layer + 1}")

    # 关键参数 shape
    for target_key in ["emb.weight", "head.weight",
                        "blocks.0.att.r_k", "blocks.0.att.receptance.weight",
                        "blocks.0.att.key.weight", "blocks.0.att.output.weight",
                        "blocks.0.att.w1.weight", "blocks.0.att.a1.weight",
                        "blocks.0.att.v1.weight", "blocks.0.att.g1.weight",
                        "blocks.0.ffn.key.weight", "blocks.0.ffn.value.weight",
                        "blocks.0.ln1.weight", "blocks.0.ln0.weight",
                        "blocks.0.att.x_r"]:
        if target_key in z:
            t = z[target_key]
            print(f"  {target_key}: {list(t.shape)} {t.dtype}")
        else:
            print(f"  {target_key}: NOT FOUND")

    # 检查 key 命名差异（看是否有 g1d 特有的 key）
    g1d_specific = [k for k in keys if "g1d" in k.lower() or "adapt" in k.lower() or "control" in k.lower()]
    if g1d_specific:
        print(f"  g1d 特有 keys: {g1d_specific[:5]}")

    # 检查是否有 blocks.0.att.g0 这样的新参数
    att_keys = [k for k in keys if k.startswith("blocks.0.att.")]
    print(f"  blocks.0.att.* keys ({len(att_keys)}):")
    for k in sorted(att_keys):
        print(f"    {k}: {list(z[k].shape)}")
