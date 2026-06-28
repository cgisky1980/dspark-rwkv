"""测试 target 的纯生成（验证 target 是否正常）"""
import torch
from pathlib import Path
from rwkv_tokenizer import TRIE_TOKENIZER
from stage9_target_2p9b import RWKV7Target2p9B

tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))
target = RWKV7Target2p9B(Path(r"C:\work\niceui\g1a-2.9B.pth"))

prompt = "什么是人工智能"
ids = tokenizer.encode(prompt)
state = target.zero_state(1)
ctx = torch.tensor([ids], device="cuda", dtype=torch.long)
logits, _ = target.forward(ctx, state, return_hidden_layers=[])
out = list(ids)
for _ in range(30):
    next_tok = logits[0, -1].argmax().item()
    out.append(next_tok)
    if next_tok == 0:
        break
    next_t = torch.tensor([[next_tok]], device="cuda", dtype=torch.long)
    logits, _ = target.forward(next_t, state, return_hidden_layers=[])
gen = tokenizer.decode(out[len(ids):])
print(f"Target 生成: {gen[:200]}")
