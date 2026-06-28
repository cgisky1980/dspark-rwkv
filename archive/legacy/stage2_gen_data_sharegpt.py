"""阶段2 ShareGPT 真实文本数据生成。

流程：
1. 从 HuggingFace 加载 ShareGPT 中英文对话数据
2. 转成 RWKV 格式: "User: xxxxx\n\nAssistant: xxxx\n\n"
3. tokenize 全部文本
4. 从 token 流随机采样 anchor 位置（anchor 是真实文本 token）
5. 2.9B target 从 anchor 自回归生成 32 个 token + 记录 5 层 hidden
6. 保存 tokens [N, 33] + hidden [N, 32, C]，按 80/10/10 划分 train/val/test
"""
import json
import torch
from pathlib import Path
from datasets import load_dataset
from stage2_target import RWKV7Target, WEIGHTS
from rwkv_tokenizer import TRIE_TOKENIZER

N_SEQ = 4096
T_LEN = 32
TARGET_LAYERS = [0, 3, 6, 9, 11]  # 5 层
BATCH = 16
SPLIT = (0.8, 0.1, 0.1)
SEED = 42
DATA_DIR = Path(__file__).parent / "data"
VOCAB_FILE = Path(__file__).parent / "rwkv_vocab_v20230424.txt"


def load_sharegpt_texts():
    """从 HuggingFace 加载 ShareGPT 中英文对话，转成 RWKV 格式文本列表。"""
    print("加载 ShareGPT 数据集...")
    # 中英文混合 ShareGPT
    ds = load_dataset("shareAI/ShareGPT-Chinese-English-90k", split="train")
    print(f"  原始样本数: {len(ds)}")
    texts = []
    for row in ds:
        # 字段可能是 'conversation' 或 'messages'，尝试多种
        conv = row.get("conversation") or row.get("messages") or row.get("items")
        if not conv:
            continue
        # 拼接成 RWKV 格式: "User: xxxxx\n\nAssistant: xxxx\n\n"
        parts = []
        for turn in conv:
            # turn 可能是 dict {human/assistant} 或 {role/content}
            if isinstance(turn, dict):
                human = turn.get("human") or turn.get("user") or turn.get("content")
                assistant = turn.get("assistant") or turn.get("bot") or turn.get("output")
                if human:
                    parts.append(f"User: {human}\n\n")
                if assistant:
                    parts.append(f"Assistant: {assistant}\n\n")
            elif isinstance(turn, (list, tuple)) and len(turn) == 2:
                role, content = turn
                if role in ("human", "user"):
                    parts.append(f"User: {content}\n\n")
                elif role in ("assistant", "bot", "gpt"):
                    parts.append(f"Assistant: {content}\n\n")
        if parts:
            texts.append("".join(parts))
    print(f"  有效对话: {len(texts)}")
    return texts


def build_token_pool(texts, tokenizer):
    """将所有文本 tokenize 后拼接成一个长 token 流。"""
    all_tokens = []
    for t in texts:
        toks = tokenizer.encode(t)
        all_tokens.extend(toks)
    return all_tokens


def main():
    tokenizer = TRIE_TOKENIZER(str(VOCAB_FILE))
    texts = load_sharegpt_texts()
    token_pool = build_token_pool(texts, tokenizer)
    print(f"token 池总长: {len(token_pool)}")

    target = RWKV7Target(WEIGHTS)
    print("target 已在 GPU")

    # 固定 seed 采样 anchor 位置
    g = torch.Generator().manual_seed(SEED)
    # anchor 从 token_pool 中均匀采样（避免开头/结尾）
    anchor_idxs = torch.randint(0, len(token_pool) - T_LEN - 1, (N_SEQ,), generator=g).tolist()

    all_tokens = []
    all_hids = {l: [] for l in TARGET_LAYERS}
    n_done = 0
    while n_done < N_SEQ:
        B = min(BATCH, N_SEQ - n_done)
        # 取真实文本 anchor
        cur = torch.tensor([[token_pool[anchor_idxs[n_done + i]]] for i in range(B)],
                           dtype=torch.long, device="cuda")
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
        tokens_cat = torch.cat(seq_tokens, dim=1)  # [B, T_LEN+1]
        all_tokens.append(tokens_cat.cpu())
        for l in TARGET_LAYERS:
            all_hids[l].append(torch.cat(seq_hids[l], dim=1).cpu())  # [B, T_LEN, C]
        n_done += B
        if n_done % 256 == 0 or n_done == N_SEQ:
            print(f"  完成 {n_done}/{N_SEQ}")

    all_tokens = torch.cat(all_tokens, dim=0)  # [N_SEQ, T_LEN+1]
    all_hids_cat = {l: torch.cat(all_hids[l], dim=0) for l in TARGET_LAYERS}  # [N_SEQ, T_LEN, C]

    # 划分 train/val/test
    g2 = torch.Generator().manual_seed