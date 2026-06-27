"""阶段2 优化B：生成 5 层 target hidden 的训练数据（layer 0/3/6/9/11）。"""
import torch
from pathlib import Path
from stage2_target import RWKV7Target, WEIGHTS

N_SEQ = 512
T_LEN = 32
TARGET_LAYERS = [0, 3, 6, 9, 11]  # 5 层
BATCH = 64
OUT = Path(__file__).parent / "data" / "stage2_train_5layer.pt"

def main():
    target = RWKV7Target(WEIGHTS)
    target.z = {k: v.to("cuda") for k, v in target.z.items()}
    print("target 已搬到 GPU")

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

    all_tokens = torch.cat(all_tokens, dim=0)
    data = {"tokens": all_tokens}
    for l in TARGET_LAYERS:
        data[f"hidden_{l}"] = torch.cat(all_hids[l], dim=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, OUT)
    print(f"\n保存到 {OUT}")
    print(f"tokens: {all_tokens.shape}")
    for l in TARGET_LAYERS:
        print(f"hidden_{l}: {data[f'hidden_{l}'].shape}")

if __name__ == "__main__":
    main()
