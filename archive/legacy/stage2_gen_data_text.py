"""阶段2 对比实验：用真实文本（LAMBADA）生成训练数据。

与 stage2_gen_data_5layer.py 对比：
- 原版：target 自回归生成（target 喜欢的分布）
- 本版：真实英文文本（LAMBADA 测试集，5153 句）

数据格式与 stage2_gen_data_5layer.py 完全一致：
- tokens: [N_SEQ, T_LEN+1] = [512, 33]
- hidden_l: [N_SEQ, T_LEN, C] = [512, 32, 768]（5 层）

仅数据来源不同，可直接喂给 stage2_train_v3.py 对比接受率。
"""
import json
import torch
from pathlib import Path
from stage2_target import RWKV7Target, WEIGHTS
from rwkv_tokenizer import TRIE_TOKENIZER

N_SEQ = 512
T_LEN = 32  # 与原版一致
TARGET_LAYERS = [0, 3, 6, 9, 11]
BATCH = 32   # 真实文本 forward 一次较重，batch 小一点
VOCAB_FILE = Path(__file__).parent / "rwkv_vocab_v20230424.txt"
LAMBADA_FILE = Path(__file__).parent / "data" / "lambada_test.jsonl"
OUT = Path(__file__).parent / "data" / "stage2_train_text.pt"


def load_lambada_sequences(tokenizer):
    """从 LAMBADA 读取并 tokenize，返回长度 >= T_LEN+1 的序列列表。"""
    seqs = []
    with open(LAMBADA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            ids = tokenizer.encode(text)
            # 取前 T_LEN+1 个 token（与原版 tokens shape 一致）
            if len(ids) >= T_LEN + 1:
                seqs.append(ids[:T_LEN + 1])
            if len(seqs) >= N_SEQ * 2:  # 多收集一些便于筛选
                break
    return seqs


def main():
    print("=== 真实文本数据生成（LAMBADA）===")
    tokenizer = TRIE_TOKENIZER(str(VOCAB_FILE))
    print(f"tokenizer 加载完成: {VOCAB_FILE.name}")

    seqs = load_lambada_sequences(tokenizer)
    print(f"LAMBADA 中长度 >= {T_LEN+1} 的句子: {len(seqs)}")
    if len(seqs) < N_SEQ:
        raise RuntimeError(f"可用句子不足 {N_SEQ}，实际只有 {len(seqs)}")
    seqs = seqs[:N_SEQ]
    print(f"取前 {N_SEQ} 条作为训练数据")

    # 打印前 3 条样本预览
    for i in range(3):
        txt = tokenizer.decode(seqs[i], utf8_errors="replace")
        print(f"  样本{i}: {txt[:80]}...")

    # 加载 target
    target = RWKV7Target(WEIGHTS)
    target.z = {k: v.to("cuda") for k, v in target.z.items()}
    print("target 已搬到 GPU")

    all_tokens = []
    all_hids = {l: [] for l in TARGET_LAYERS}

    for start in range(0, N_SEQ, BATCH):
        end = min(start + BATCH, N_SEQ)
        B = end - start
        # 一次 forward 整个序列（T_LEN+1 个 token），记录每位置 hidden
        tokens_batch = torch.tensor(seqs[start:end], dtype=torch.long, device="cuda")  # [B, T_LEN+1]
        state = target.zero_state(B, device="cuda")
        with torch.no_grad():
            logits, hids = target.forward(tokens_batch, state, return_hidden_layers=TARGET_LAYERS)
        # tokens 存完整 T_LEN+1，hidden 存前 T_LEN 个位置（对齐 stage2_gen_data_5layer）
        all_tokens.append(tokens_batch.cpu())
        for i, l in enumerate(TARGET_LAYERS):
            # hids[i]: [B, T_LEN+1, C]，取前 T_LEN 个
            all_hids[l].append(hids[i][:, :T_LEN, :].cpu())
        print(f"  完成 {end}/{N_SEQ}")

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
    for l in TARGET_LAYERS:
        print(f"hidden_{l} 均值/方差: {data[f'hidden_{l}'].mean():.4f}/{data[f'hidden_{l}'].std():.4f}")
    print("\n=== 数据生成完成，可用 stage2_train_v3_text.py 训练对比 ===")


if __name__ == "__main__":
    main()
