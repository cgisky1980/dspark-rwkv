"""阶段2：生成 5 层 target hidden 的训练数据（layer 0/3/6/9/11）。

修复数据泄露：生成后按 80/10/10 划分 train/val/test，保存三个独立文件。
Target: RWKV-7 2.9B（GPU fp16）。
"""
import torch
from pathlib import Path
from stage2_target import RWKV7Target, WEIGHTS

N_SEQ = 2048
T_LEN = 32
TARGET_LAYERS = [0, 3, 6, 9, 11]  # 5 层
BATCH = 16  # 2.9B 模型显存较大，batch 降到 16
SPLIT = (0.8, 0.1, 0.1)  # train / val / test
SEED = 42
DATA_DIR = Path(__file__).parent / "data"


def main():
    target = RWKV7Target(WEIGHTS)
    print("target 已在 GPU")

    all_tokens = []
    all_hids = {l: [] for l in TARGET_LAYERS}
    n_done = 0
    while n_done < N_SEQ:
        B = min(BATCH, N_SEQ - n_done)
        cur = torch.randint(0, 1000, (B, 1), device="cuda")
        state = target.zero_state(B, device="cuda")
        seq_tokens = [cur]
        seq_hids = {l: [] for l in TARGET_LAYERS}
        with torch.no_grad():
            for t in range(T_LEN):
                logits, hids = target.forward(cur, state, return_hidden_layers=TARGET_LAYERS)
                probs = torch.softmax(logits[:, -1] / 0.8, dim=-1)
                nxt = torch.multinomial(probs, 1)
                seq_tokens.append(nxt)
                for i, l in enumerate(TARGET_LAYERS):
                    seq_hids[l].append(hids[i][:, -1:, :])
                cur = nxt
        tokens_cat = torch.cat(seq_tokens, dim=1)
        all_tokens.append(tokens_cat.cpu())
        for l in TARGET_LAYERS:
            all_hids[l].append(torch.cat(seq_hids[l], dim=1).cpu())
        n_done += B
        print(f"  完成 {n_done}/{N_SEQ}")

    all_tokens = torch.cat(all_tokens, dim=0)  # [N_SEQ, T_LEN+1]
    all_hids_cat = {l: torch.cat(all_hids[l], dim=0) for l in TARGET_LAYERS}  # [N_SEQ, T_LEN, C]

    # 固定 seed 划分 train/val/test，保证可复现
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(N_SEQ, generator=g)
    n_train = int(N_SEQ * SPLIT[0])
    n_val = int(N_SEQ * SPLIT[1])
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        data = {"tokens": all_tokens[idx]}
        for l in TARGET_LAYERS:
            data[f"hidden_{l}"] = all_hids_cat[l][idx]
        out = DATA_DIR / f"stage2_{name}_5layer.pt"
        torch.save(data, out)
        print(f"保存 {name}: {len(idx)} 条 -> {out}")
        print(f"  tokens: {data['tokens'].shape}")
        for l in TARGET_LAYERS:
            print(f"  hidden_{l}: {data[f'hidden_{l}'].shape}")


if __name__ == "__main__":
    main()
