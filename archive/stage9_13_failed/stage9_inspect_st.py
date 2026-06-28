"""检查 .st 文件格式"""
from pathlib import Path
from safetensors import safe_open

ST_PATH = Path(r"c:\work\niceui\参考\rwkv7-state-fusion\models\rwkv7-g1a-0.4b-20250905-ctx4096.st")

with safe_open(str(ST_PATH), framework="pt") as f:
    keys = list(f.keys())
    print(f"Total keys: {len(keys)}")
    print("First 10:")
    for k in keys[:10]:
        t = f.get_tensor(k)
        print(f"  {k}: {t.shape} {t.dtype}")
    print("Last 5:")
    for k in keys[-5:]:
        t = f.get_tensor(k)
        print(f"  {k}: {t.shape} {t.dtype}")

    # 检查关键 key
    print("\n关键 key 检查:")
    for target in ["emb.weight", "blocks.0.att.r_k", "blocks.0.att.receptance.weight",
                   "blocks.0.ln1.weight", "blocks.0.ln0.weight", "head.weight"]:
        if target in keys:
            t = f.get_tensor(target)
            print(f"  {target}: {t.shape} {t.dtype}")
        else:
            print(f"  {target}: NOT FOUND")

    # 检查最大 layer
    max_layer = -1
    for k in keys:
        if k.startswith("blocks."):
            max_layer = max(max_layer, int(k.split(".")[1]))
    print(f"\n最大 layer: {max_layer} (总 {max_layer+1} 层)")
