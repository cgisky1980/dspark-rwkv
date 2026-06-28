"""阶段2a：用 target 自回归生成训练数据并预计算 hidden。

数据格式：
- 用 RWKV-7 0.1B 自回归生成 N_SEQ 个 token 序列（每个 T_LEN token）
- 预计算 target 在每位置的 3 层 hidden（layer 0/6/11）
- 存到 data/stage2_train.pt：{tokens, hidden_layer_0/6/11}

训练时从这些序列采 anchor + block，不需要再跑 target forward。
logits 太大（4GB）不存，训练用 CE loss（label=target token）+ 置信度 BCE，不用 L1 loss。
"""
import torch
from pathlib import Path
from stage2_target import RWKV7Target, WEIGHTS

N_SEQ = 512      # 序列数
T_LEN = 32       # 每序列长度
TARGET_LAYERS = [0, 6, 11]  # 首中尾三层
BATCH = 64       # 生成时 batch
OUT = Path(__file__).parent / "data" / "stage2_train.pt"

def main():
    target = RWKV7Target(WEIGHTS)
    target.z = {k: v.to("cuda") for k, v in target.z.items()}
    print("target 已搬到 GPU")

    all_tokens = []
    all_hids = {l: [] for l in TARGET_LAYERS}

    n_done = 0
    while n_done < N_SEQ:
        B = min(BATCH, N_SEQ - n_done)
        # 随机起始 token（用前 1000 常见 token）
        cur = torch.randint(0, 1000, (B, 1), device="cuda")
        state = target.zero_state(B, device="cuda")
        seq_tokens = [cur]
        seq_hids = {l: [] for l in TARGET_LAYERS}
        # 自回归生成 T_LEN 个 token，记录每步 hidden
        with torch.no_grad():
            for t in range(T_LEN):
                logits, hids = target.forward(cur, state, return_hidden_layers=TARGET_LAYERS)
                # 采样下一个 token（temperature 采样，让分布多样）
                probs = torch.softmax(logits[:, -1] / 0.8, dim=-1)
                nxt = torch.multinomial(probs, 1)  # [B,1]
                seq_tokens.append(nxt)
                for i, l in enumerate(TARGET_LAYERS):
                    seq_hids[l].append(hids[i][:, -1:, :])  # [B,1,C]
                cur = nxt
                if (t + 1) % 16 == 0:
                    print(f"  seq batch {n_done}-{n_done+B}, step {t+1}/{T_LEN}")
        # 拼接：tokens [B, T_LEN+1]（含起始 token），hidden [B, T_LEN, C]
        tokens_cat = torch.cat(seq_tokens, dim=1)  # [B, T_LEN+1]
        all_tokens.append(tokens_cat.cpu())
        for l in TARGET_LAYERS:
            h_cat = torch.cat(seq_hids[l], dim=1)  # [B, T_LEN, C]
            all_hids[l].append(h_cat.cpu())
        n_done += B
        print(f"  完成 {n_done}/{N_SEQ} 序列")

    # 拼接所有
    all_tokens = torch.cat(all_tokens, dim=0)  # [N_SEQ, T_LEN+1]
    data = {"tokens": all_tokens}
    for l in TARGET_LAYERS:
        data[f"hidden_{l}"] = torch.cat(all_hids[l], dim=0)  # [N_SEQ, T_LEN, C]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, OUT)
    print(f"\n保存到 {OUT}")
    print(f"tokens: {all_tokens.shape}")
    for l in TARGET_LAYERS:
        print(f"hidden_{l}: {data[f'hidden_{l}'].shape}")
    # 简单统计
    print(f"\ntoken 分布: min={all_tokens.min().item()}, max={all_tokens.max().item()}")
    print(f"hidden_0 均值/方差: {data['hidden_0'].mean():.4f}/{data['hidden_0'].std():.4f}")
    print(f"hidden_6 均值/方差: {data['hidden_6'].mean():.4f}/{data['hidden_6'].std():.4f}")
    print(f"hidden_11 均值/方差: {data['hidden_11'].mean():.4f}/{data['hidden_11'].std():.4f}")

if __name__ == "__main__":
    main()
