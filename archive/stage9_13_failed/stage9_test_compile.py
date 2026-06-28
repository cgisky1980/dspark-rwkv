"""测试 torch.compile 是否可用 + 对比编译前后 forward 耗时"""
import time
import torch
from pathlib import Path
from stage9_target_2p9b import RWKV7Target2p9B, DTYPE, DEVICE
from stage9_logit_fusion_01b import RWKV7Draft
from rwkv_tokenizer import TRIE_TOKENIZER

print(f"torch version: {torch.__version__}")
print(f"torch.compile available: {hasattr(torch, 'compile')}")
print(f"CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0)}")

G1A_TARGET = Path(r"C:\work\niceui\g1a-2.9B.pth")
DRAFT_WEIGHTS = Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1a-0.4b-20250905-ctx4096.pth")
tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))
ids = tokenizer.encode("什么是人工智能")

# ============ 1. 不编译基线 ============
print("\n=== 不编译 (基线) ===", flush=True)
target = RWKV7Target2p9B(G1A_TARGET)
draft = RWKV7Draft(DRAFT_WEIGHTS)

for name, model, is_t in [("target", target, True), ("draft", draft, False)]:
    for T in [1, 4, 16]:
        state = model.zero_state(1)
        ctx = torch.tensor([ids[:T]], device=DEVICE, dtype=torch.long)
        # warmup
        if is_t:
            model.forward(ctx, state, return_hidden_layers=[])
        else:
            model.forward(ctx, state)
        torch.cuda.synchronize()
        # bench
        state = model.zero_state(1)
        t0 = time.time()
        for _ in range(5):
            if is_t:
                model.forward(ctx, state, return_hidden_layers=[])
            else:
                model.forward(ctx, state)
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / 5
        print(f"  {name} T={T:>3}: {elapsed*1000:.1f}ms ({elapsed/T*1000:.1f}ms/tok)", flush=True)

# 释放模型
del target, draft
torch.cuda.empty_cache()

# ============ 2. torch.compile 测试 ============
print("\n=== torch.compile (reduce-overhead) ===", flush=True)
try:
    target = RWKV7Target2p9B(G1A_TARGET)
    draft = RWKV7Draft(DRAFT_WEIGHTS)

    # 尝试编译
    target_compiled = torch.compile(target, mode="reduce-overhead")
    draft_compiled = torch.compile(draft, mode="reduce-overhead")
    print("  compile 成功，开始 warmup...", flush=True)

    # warmup (编译会在第一次调用时触发，可能很慢)
    for name, model, is_t in [("target", target_compiled, True), ("draft", draft_compiled, False)]:
        for T in [1, 4, 16]:
            state = model.zero_state(1)
            ctx = torch.tensor([ids[:T]], device=DEVICE, dtype=torch.long)
            print(f"  warmup {name} T={T}...", flush=True)
            t0 = time.time()
            if is_t:
                model.forward(ctx, state, return_hidden_layers=[])
            else:
                model.forward(ctx, state)
            torch.cuda.synchronize()
            print(f"    warmup 耗时 {(time.time()-t0):.1f}s", flush=True)

            state = model.zero_state(1)
            t0 = time.time()
            for _ in range(5):
                if is_t:
                    model.forward(ctx, state, return_hidden_layers=[])
                else:
                    model.forward(ctx, state)
            torch.cuda.synchronize()
            elapsed = (time.time() - t0) / 5
            print(f"    {name} T={T:>3}: {elapsed*1000:.1f}ms ({elapsed/T*1000:.1f}ms/tok)", flush=True)

except Exception as e:
    print(f"  compile 失败: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
