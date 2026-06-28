"""DSpark → RWKV · 阶段9：用 2.9B target 生成训练数据（LAMBADA + WikiText）

- target: rwkv7-g1h_preview4533-2.9b（L=32, C=2560, H=40, N=64, V=65536）
- 数据: LAMBADA 5153 + WikiText 27615 = 32768 条真实文本
- 5 层 hidden: [0, 8, 16, 24, 31]（从 32 层均匀取 5 层）
- T_LEN = 32（token 序列长度）
- BATCH = 64（并发 forward）
- 输出: data/stage9_train.pt（~27GB，FP16）
"""
import json
import time
import torch
from pathlib import Path
from stage9_target_2p9b import RWKV7Target2p9B, DTYPE, DEVICE
from rwkv_tokenizer import TRIE_TOKENIZER

# ==================================================================
# 配置
# ==================================================================
T_LEN = 32           # token 序列长度（不含 anchor 位置的下一个 token）
TARGET_LAYERS = [0, 8, 16, 24, 31]  # 从 32 层均匀取 5 层
N_SEQ = 32768        # 总序列数
BATCH = 64           # 并发 forward batch size

VOCAB_FILE = Path(__file__).parent / "rwkv_vocab_v20230424.txt"
LAMBADA_FILE = Path(__file__).parent / "data" / "lambada_test.jsonl"
WIKITEXT_FILE = Path(__file__).parent / "data" / "wikitext_sentences.jsonl"
OUT = Path(__file__).parent / "data" / "stage9_train.pt"


def load_sequences(tokenizer):
    """从 LAMBADA + WikiText 读取并 tokenize，返回长度 >= T_LEN+1 的序列列表。"""
    seqs = []
    # LAMBADA（5153 条）
    with open(LAMBADA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            ids = tokenizer.encode(text)
            if len(ids) >= T_LEN + 1:
                seqs.append(ids[:T_LEN + 1])
    print(f"LAMBADA: {len(seqs)} 条 (长度 >= {T_LEN+1})")

    # WikiText（27615 条）
    n_before = len(seqs)
    with open(WIKITEXT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            ids = tokenizer.encode(text)
            if len(ids) >= T_LEN + 1:
                seqs.append(ids[:T_LEN + 1])
    print(f"WikiText: {len(seqs) - n_before} 条 (长度 >= {T_LEN+1})")
    print(f"总计: {len(seqs)} 条")

    if len(seqs) < N_SEQ:
        # 如果不够，重复采样补充
        print(f"警告: 只有 {len(seqs)} 条，不足 {N_SEQ}，将重复采样")
        import random
        random.seed(42)
        extra = [random.choice(seqs) for _ in range(N_SEQ - len(seqs))]
        seqs.extend(extra)
        print(f"补充后: {len(seqs)} 条")

    return seqs[:N_SEQ]


def main():
    print("=== 阶段9: 生成 2.9B target 训练数据 ===")
    print(f"target layers: {TARGET_LAYERS}")
    print(f"T_LEN={T_LEN}, N_SEQ={N_SEQ}, BATCH={BATCH}")

    # 1. tokenize 文本
    tokenizer = TRIE_TOKENIZER(str(VOCAB_FILE))
    print(f"tokenizer 加载完成")
    seqs = load_sequences(tokenizer)
    print(f"取前 {N_SEQ} 条")

    # 打印前 3 条预览
    for i in range(3):
        txt = tokenizer.decode(seqs[i], utf8_errors="replace")
        print(f"  样本{i}: {txt[:80]}...")

    # 2. 加载 target
    print(f"\n加载 2.9B target...")
    target = RWKV7Target2p9B()
    print(f"GPU 显存（权重）: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # 3. 预分配 CPU tensor（流式填充，避免 list 累积内存碎片）
    C = target.C
    tokens_buf = torch.zeros(N_SEQ, T_LEN + 1, dtype=torch.long)  # 8.6 MB
    hidden_bufs = {
        l: torch.zeros(N_SEQ, T_LEN, C, dtype=DTYPE)  # 每个 5.37 GB
        for l in TARGET_LAYERS
    }
    print(f"\n预分配 CPU tensor:")
    print(f"  tokens: {tokens_buf.shape} ({tokens_buf.numel()*8/1e6:.1f} MB)")
    for l in TARGET_LAYERS:
        sz = hidden_bufs[l].numel() * 2 / 1e9
        print(f"  hidden_{l}: {hidden_bufs[l].shape} ({sz:.2f} GB)")
    total_gb = tokens_buf.numel()*8/1e9 + sum(hidden_bufs[l].numel()*2 for l in TARGET_LAYERS)/1e9
    print(f"  总计: {total_gb:.2f} GB")

    # 4. batch forward
    print(f"\n开始 forward ({N_SEQ // BATCH} batches)...")
    t0 = time.time()
    n_done = 0
    for start in range(0, N_SEQ, BATCH):
        end = min(start + BATCH, N_SEQ)
        B = end - start
        # 准备 batch tokens [B, T_LEN+1]
        batch_tokens = torch.tensor([seqs[i] for i in range(start, end)], dtype=torch.long, device=DEVICE)
        state = target.zero_state(B)
        with torch.no_grad():
            logits, hids = target.forward(batch_tokens, state, return_hidden_layers=TARGET_LAYERS)
        # 保存 tokens（int64）
        tokens_buf[start:end] = batch_tokens.cpu()
        # 保存 hidden（FP16，取前 T_LEN 个位置）
        for i, l in enumerate(TARGET_LAYERS):
            hidden_bufs[l][start:end] = hids[i][:, :T_LEN, :].cpu()

        n_done += B
        if n_done % 1024 == 0 or n_done == N_SEQ:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta = (N_SEQ - n_done) / rate if rate > 0 else 0
            gpu_gb = torch.cuda.memory_allocated() / 1e9
            print(f"  {n_done}/{N_SEQ}  rate={rate:.1f}/s  ETA={eta:.0f}s  GPU={gpu_gb:.2f}GB")

        # 释放 GPU 缓存
        del logits, hids, state, batch_tokens
        if n_done % 4096 == 0:
            torch.cuda.empty_cache()

    # 5. 保存
    print(f"\n保存到 {OUT}...")
    data = {"tokens": tokens_buf}
    for l in TARGET_LAYERS:
        data[f"hidden_{l}"] = hidden_bufs[l]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, OUT)
    sz_gb = OUT.stat().st_size / 1e9
    print(f"保存完成: {OUT} ({sz_gb:.2f} GB)")

    # 6. 验证
    print(f"\n=== 验证 ===")
    print(f"tokens: {tokens_buf.shape} dtype={tokens_buf.dtype}")
    print(f"  min={tokens_buf.min().item()}, max={tokens_buf.max().item()}")
    for l in TARGET_LAYERS:
        h = hidden_bufs[l]
        print(f"hidden_{l}: {h.shape} dtype={h.dtype} mean={h.float().mean():.4f} std={h.float().std():.4f}")

    elapsed = time.time() - t0
    print(f"\n=== 完成 === 总耗时 {elapsed:.1f}s")


if __name__ == "__main__":
    main()
