"""阶段8：生成大数据集（train 8192 + val 1024，独立 seed）。

修复 v3 过拟合问题：
- v3: 512 条序列，训练=验证，625 epoch 严重过拟合
- stage8: 8192 条训练 + 1024 条独立验证（不同 seed），

数据格式与 stage2_gen_data_5layer.py 完全一致：
- tokens: [N, T_LEN+1] = [8192, 33]
- hidden_l: [N, T_LEN, C] = [8192, 32, 768]（5 层）
"""
import torch
from pathlib import Path
from stage2_target import RWKV7Target, WEIGHTS

T_LEN = 32
TARGET_LAYERS = [0, 3, 6, 9, 11]
BATCH = 64

TRAIN_SEQS = 8192
VAL_SEQS = 1024
TRAIN_OUT = Path(__file__).parent / "data" / "stage8_train.pt"
VAL_OUT = Path(__file__).parent / "data" / "stage8_val.pt"


@torch.no_grad()
def gen_split(target, n_seqs, seed, out_path, name):
    """用 target 自回归生成 n_seqs 条序列。"""
    print(f"\n[{name}] 生成 {n_seqs} 条 (seed={seed})")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    all_tokens = []
    all_hids = {l: [] for l in TARGET_LAYERS}
    n_done = 0
    while n_done < n_seqs:
        B = min(BATCH, n_seqs - n_done)
        cur = torch.randint(0, 1000, (B, 1), device="cuda")
        state = target.zero_state(B, device="cuda")
        seq_tokens = [cur]
        seq_hids = {l: [] for l in TARGET_LAYERS}
        for t in range(T_LEN):
            logits, hids = target.forward(cur, state, return_hidden_layers=TARGET_LAYERS)
            probs = torch.softmax(logits[:, -1] / 0.8, dim=-1)
            nxt = torch.multinomial(probs, 1)
            seq_tokens.append(nxt)
            for i, l in enumerate(TARGET_LAYERS):
                seq_hids[l].append(hids[i][:, -1:, :])
            cur = nxt
        all_tokens.append(torch.cat(seq_tokens, dim=1).cpu())
        for l in TARGET_LAYERS:
            all_hids[l].append(torch.cat(seq_hids[l], dim=1).cpu())
        n_done += B
        if n_done % 1024 == 0 or n_done == n_seqs:
            print(f"  {name}: {n_done}/{n_seqs}")

    tokens = torch.cat(all_tokens, dim=0)
    data = {"tokens": tokens}
    for l in TARGET_LAYERS:
        data[f"hidden_{l}"] = torch.cat(all_hids[l], dim=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  保存: {out_path} ({size_mb:.1f} MB)")
    print(f"  tokens: {tokens.shape}, hidden_{TARGET_LAYERS[0]}: {data[f'hidden_{TARGET_LAYERS[0]}'].shape}")


def main():
    print(f"=== 阶段8: 生成大数据集 ===")
    target = RWKV7Target(WEIGHTS)
    target.z = {k: v.to("cuda") for k, v in target.z.items()}
    print(f"target 在 GPU")

    gen_split(target, TRAIN_SEQS, seed=42, out_path=TRAIN_OUT, name="训练集")
    gen_split(target, VAL_SEQS, seed=999, out_path=VAL_OUT, name="验证集")

    print(f"\n=== 完成 ===")
    print(f"训练集: {TRAIN_OUT}")
    print(f"验证集: {VAL_OUT}")


if __name__ == "__main__":
    main()
