"""测试 0.1B 和 0.4B g1d 的纯 draft 生成（无 fusion）"""
import torch
import torch.nn.functional as F
from pathlib import Path
from rwkv_tokenizer import TRIE_TOKENIZER
from stage9_logit_fusion_01b import RWKV7Draft

tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))

prompt = "什么是人工智能"
ids = tokenizer.encode(prompt)
print(f"Prompt: {prompt}")
print(f"Token ids: {ids[:20]}...")

for name, path in [
    ("0.4B g1a", Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1a-0.4b-20250905-ctx4096.pth")),
    ("0.4B g1d", Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1d-0.4b-20260210-ctx8192.pth")),
]:
    print(f"\n=== {name} 纯生成 ===")
    draft = RWKV7Draft(path)
    state = draft.zero_state(1)
    ctx = torch.tensor([ids], device="cuda", dtype=torch.long)
    logits = draft.forward(ctx, state)
    out = list(ids)
    for _ in range(30):
        next_tok = logits[0, -1].argmax().item()
        out.append(next_tok)
        if next_tok == 0:
            break
        next_t = torch.tensor([[next_tok]], device="cuda", dtype=torch.long)
        logits = draft.forward(next_t, state)
    gen = tokenizer.decode(out[len(ids):])
    print(f"  生成: {gen[:150]}")
    print(f"  tokens: {out[len(ids):len(ids)+10]}")
