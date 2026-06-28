"""转换 .st → .pth（保持 key 和 shape 不变）"""
from pathlib import Path
import torch
from safetensors import safe_open

ST_PATH = Path(r"c:\work\niceui\参考\rwkv7-state-fusion\models\rwkv7-g1a-0.4b-20250905-ctx4096.st")
PTH_PATH = Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1a-0.4b-20250905-ctx4096.pth")

print(f"转换: {ST_PATH}")
print(f"  -> {PTH_PATH}")

with safe_open(str(ST_PATH), framework="pt") as f:
    keys = list(f.keys())
    state_dict = {}
    for k in keys:
        state_dict[k] = f.get_tensor(k).clone()

torch.save(state_dict, PTH_PATH)
print(f"完成: {len(state_dict)} 个 key")
print(f"文件大小: {PTH_PATH.stat().st_size / 1e6:.1f} MB")
