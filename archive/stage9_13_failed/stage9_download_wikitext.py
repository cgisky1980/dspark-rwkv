"""阶段9：下载 WikiText-103-raw，切分成句子，保存为 JSONL。

补充 LAMBADA（5153 条）到 32768 条，需要 27615 条 WikiText。
输出格式与 LAMBADA 一致：每行 {"text": "..."}。
"""
import json
from pathlib import Path
from datasets import load_dataset

OUT = Path(__file__).parent / "data" / "wikitext_sentences.jsonl"
NEEDED = 27615  # 32768 - 5153(LAMBADA)
MIN_LEN = 50    # 最短字符数（太短的句子没意义）
MAX_LEN = 500   # 最长字符数（太长的句子截断或跳过）


def main():
    print("=== 下载 WikiText-103-raw-v1 (train) ===")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)

    n = 0
    skipped = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for item in ds:
            text = item["text"].strip()
            # 跳过空行、标题行（= = ...）、过短/过长
            if not text or text.startswith("=") or len(text) < MIN_LEN or len(text) > MAX_LEN:
                skipped += 1
                continue
            # 按 . ! ? 切分成句子，保留标点
            for sent in split_sentences(text):
                s = sent.strip()
                if MIN_LEN <= len(s) <= MAX_LEN:
                    f.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")
                    n += 1
                    if n >= NEEDED:
                        break
            if n >= NEEDED:
                break

    print(f"保存: {OUT}")
    print(f"句子数: {n} (跳过 {skipped} 行)")
    # 验证
    with open(OUT, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"文件行数: {len(lines)}")
    print(f"前 2 行: {lines[0].strip()[:80]}... / {lines[1].strip()[:80]}...")


def split_sentences(text):
    """简单按 . ! ? 切分，保留标点。"""
    import re
    parts = re.split(r'(?<=[.!?])\s+', text)
    return parts


if __name__ == "__main__":
    main()
