"""探查 2.9B target 权重结构。

不加载到 GPU，只在 CPU 上读 keys 和 shape，确定架构参数。
"""
import torch
from pathlib import Path

WEIGHTS = Path(r"c:\work\niceui\rwkv7-g1h_preview4533-2.9b-20260623-ctx8192.pth.pth")

print(f"加载权重 keys: {WEIGHTS}")
# 只加载到 CPU，weights_only 取决于 torch 版本
try:
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
except Exception as e:
    print(f"weights_only=True 失败: {e}")
    print("尝试 weights_only=False...")
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=False)

print(f"\n类型: {type(state)}")
if isinstance(state, dict):
    print(f"顶层 keys: {list(state.keys())[:20]}")
    # 如果是嵌套的（含 'config' 或 '_config'）
    for k in ['config', '_config', 'args', 'model_args']:
        if k in state:
            print(f"\n{k}: {state[k]}")

    # 找实际的 weight dict
    if 'state_dict' in state:
        sd = state['state_dict']
    elif 'model' in state:
        sd = state['model']
    else:
        sd = state

    print(f"\n权重 keys 数: {len(sd)}")
    print(f"\n前 30 个 key + shape:")
    for i, (k, v) in enumerate(sd.items()):
        if i >= 30:
            break
        if hasattr(v, 'shape'):
            print(f"  {k}: {tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")

    # 找关键架构参数
    print(f"\n=== 架构推断 ===")
    import re
    layer_ids = set()
    for k in sd.keys():
        m = re.match(r'(blocks|layers)\.(\d+)\.', k)
        if m:
            layer_ids.add(int(m.group(2)))
    if layer_ids:
        print(f"  层数: {max(layer_ids) + 1} (id 0..{max(layer_ids)})")

    # 找 head 和 FFN 维度
    for k, v in sd.items():
        if hasattr(v, 'shape'):
            # emb.weight 或 head.weight: [vocab, d_model]
            if (k == 'emb.weight' or k == 'head.weight') and v.dim() == 2:
                print(f"  {k}: vocab={v.shape[0]}, d_model={v.shape[1]}")
            # block 0 的 FFN：key.weight, receptance.weight, value.weight
            if k == 'blocks.0.att.key.weight':
                # 在新版 RWKV-7 中 att.key 可能是 data_alt
                print(f"  {k}: {tuple(v.shape)} (att 内部投影)")
            if k == 'blocks.0.ffn.key.weight':
                print(f"  {k}: {tuple(v.shape)} (FFN key, 推断 ffn_dim)")
            if k == 'blocks.0.ffn.value.weight':
                print(f"  {k}: {tuple(v.shape)} (FFN value)")
            if k == 'blocks.0.att.g1.weight':
                print(f"  {k}: {tuple(v.shape)} (g1)")
            if k == 'blocks.0.att.g2.weight':
                print(f"  {k}: {tuple(v.shape)} (g2)")

    # 找 hidden_dim（从 ln0 或 att 的 key）
    for k, v in sd.items():
        if 'ln0' in k or 'ln_x' in k or 'att.ln' in k or 'att.r_k' in k:
            if hasattr(v, 'shape') and v.dim() == 1:
                print(f"  {k}: shape={tuple(v.shape)} (推断 hidden_dim)")
                break
