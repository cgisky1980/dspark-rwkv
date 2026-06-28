"""阶段2 v3：分文件数据生成（fp16，不量化不合并）。

- 数据源: mlabonne/open-perfectblend（DSpark 论文同源）
- 对话格式: User: xxxxx\n\nAssistant: xxxxxxx\n\n
- anchor 切片: 从 'Assistant: ' 之后开始
- 分文件: 每个文件 1W 条 fp16，独立存储，训练时分文件读
- 共 10W 条 = 10 个文件
"""
import time
import torch
from pathlib import Path
from stage2_target import RWKV7Target, WEIGHTS
from rwkv_tokenizer import TRIE_TOKENIZER

DATA_DIR = Path(__file__).parent / "data"
VOCAB_FILE = Path(__file__).parent / "rwkv_vocab_v20230424.txt"
TARGET_LAYERS = [0, 3, 6, 9, 11]  # 5 层
T_LEN = 32
CHUNK_SIZE = 10000   # 每文件 1W 条
N_CHUNKS = 10         # 共 10 个文件 = 10W 条
BATCH = 128
SEED = 42


def load_conversations(n_target):
    """加载 open-perfectblend，返回结构化对话 List[List[(role, content)]]。"""
    from datasets import load_dataset
    print(f"下载 open-perfectblend (取 {n_target} 条对话)...")
    ds = load_dataset("mlabonne/open-perfectblend", split="train", streaming=True)
    conversations = []
    for item in ds:
        convs = item.get("conversations", [])
        if not convs:
            continue
        turns = []
        for c in convs:
            role = c.get("from", "")
            value = c.get("value", "")
            if role == "human":
                turns.append(("user", value))
            elif role == "gpt":
                turns.append(("assistant", value))
        if turns:
            conversations.append(turns)
        if len(conversations) >= n_target:
            break
    print(f"加载对话: {len(conversations)} 条")
    return conversations


def build_token_pool(conversations, tokenizer):
    """合并所有对话为单一 token 序列，记录每个 anchor 起始位置。

    对话内容中的 \\n\\n 统一替换为 \\n（单换行），但对话之间的分隔符保持 \\n\\n。
    """
    all_tokens = []
    anchor_starts = []
    for turns in conversations:
        for role, content in turns:
            # 仅替换 content 内部的 \n\n 为 \n
            content = content.replace("\r\n", "\n").replace("\n\n", "\n")
            if role == "user":
                # 对话之间分隔符保持 \n\n
                tokens = tokenizer.encode(f"User: {content}\n\n")
                all_tokens.extend(tokens)
            else:  # assistant —— "Assistant: " 之后即 response
                all_tokens.extend(tokenizer.encode("Assistant: "))
                anchor_pos = len(all_tokens)  # response 第一个 token
                anchor_starts.append(anchor_pos)
                # response 末尾也保持 \n\n 作为下一轮分隔
                all_tokens.extend(tokenizer.encode(content + "\n\n"))
    return all_tokens, anchor_starts


def sample_blocks(all_tokens, anchor_starts, n_seq, t_len, seed):
    """从 anchor 起始位置切片为 [anchor, t1, ..., t_K] blocks。"""
    torch.manual_seed(seed)
    block_len = t_len + 1
    available = [p for p in anchor_starts if p + block_len <= len(all_tokens)]
    print(f"可用 anchor (response >= {block_len} tokens): {len(available)}")
    if len(available) < n_seq:
        reps = (n_seq // len(available)) + 1
        available = available * reps
        print(f"  anchor 不足，重复 {reps} 次")
    indices = torch.randperm(len(available))[:n_seq]
    blocks = [all_tokens[available[i]:available[i] + block_len] for i in indices]
    return torch.tensor(blocks, dtype=torch.long)


def forward_and_save(target, tokens_chunk, chunk_idx):
    """对一个 chunk（1W 条）forward 并保存为 fp16 .pt 文件。"""
    N = tokens_chunk.shape[0]
    print(f"\n[chunk {chunk_idx+1}/{N_CHUNKS}] forward {N} 条...")
    t0 = time.time()
    # 准备 fp16 结果容器
    hids_fp16 = {l: torch.empty(N, T_LEN, target.C, dtype=torch.float16) for l in TARGET_LAYERS}

    n_done = 0
    while n_done < N:
        B = min(BATCH, N - n_done)
        tokens = tokens_chunk[n_done:n_done + B].to("cuda")
        state = target.zero_state(B, device="cuda")
        with torch.no_grad():
            logits, hids = target.forward(tokens, state, return_hidden_layers=TARGET_LAYERS)
        for i, l in enumerate(TARGET_LAYERS):
            # hids[i]: [B, T+1, C]，取前 T_LEN（target 预测用的 hidden）
            x = hids[i][:, :T_LEN, :].to(torch.float16).cpu()
            hids_fp16[l][n_done:n_done + B] = x
        del logits, hids, tokens, state
        n_done += B
        if n_done % (BATCH * 20) == 0 or n_done == N:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta = (N - n_done) / rate
            print(f"  完成 {n_done}/{N}  速率 {rate:.0f} 条/秒  ETA {eta:.0f}s")

    # 保存 chunk 文件
    data = {"tokens": tokens_chunk}
    for l in TARGET_LAYERS:
        data[f"hidden_{l}"] = hids_fp16[l]
    out = DATA_DIR / f"chunk_{chunk_idx:02d}.pt"
    torch.save(data, out)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"  保存 -> {out.name} ({size_mb:.1f} MB)  耗时 {time.time()-t0:.1f}s")


def main():
    t0 = time.time()
    print("加载 tokenizer...")
    tokenizer = TRIE_TOKENIZER(str(VOCAB_FILE))

    # 加载足够对话以切出 N_CHUNKS * CHUNK_SIZE 条序列
    # 估算: 50000 条对话通常可切出 10W+ anchor
    n_conv_estimate = 50000
    conversations = load_conversations(n_conv_estimate)
    all_tokens, anchor_starts = build_token_pool(conversations, tokenizer)
    print(f"总 token: {len(all_tokens)}, anchor 数: {len(anchor_starts)}")

    print("加载 target 模型...")
    target = RWKV7Target(WEIGHTS)
    print(f"target: L={target.n_layer} C={target.C} V={target.V}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 分 chunk 生成：每个 chunk 用不同 seed 采样不同的序列
    total_seq = CHUNK_SIZE * N_CHUNKS
    # 一次性采样所有 chunk 的 token blocks（token 很小，无内存压力）
    torch.manual_seed(SEED)
    block_len = T_LEN + 1
    available = [p for p in anchor_starts if p + block_len <= len(all_tokens)]
    print(f"可用 anchor: {len(available)}, 需要: {total_seq}")
    reps = (total_seq // len(available)) + 1 if len(available) < total_seq else 1
    available = available * reps
    perm = torch.randperm(len(available))[:total_seq]
    print(f"采样 {len(perm)} 条序列，分 {N_CHUNKS} 个 chunk 写入")

    for ci in range(N_CHUNKS):
        sel = perm[ci * CHUNK_SIZE:(ci + 1) * CHUNK_SIZE]
        chunk_blocks = torch.stack([
            torch.tensor(all_tokens[available[i]:available[i] + block_len], dtype=torch.long)
            for i in sel
        ])
        forward_and_save(target, chunk_blocks, ci)

    print(f"\n全部完成: 总耗时 {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} 分钟)")
    print(f"输出目录: {DATA_DIR}")
    print(f"共 {N_CHUNKS} 个 chunk 文件，每个 {CHUNK_SIZE} 条 fp16")


if __name__ == "__main__":
    main()
