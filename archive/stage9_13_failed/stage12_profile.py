"""stage12 速度瓶颈 profiling

精确测量投机解码每个阶段的时间:
1. target forward [1, T] 单次时间(T=1, 4, 16)
2. draft forward [1, T] 单次时间(T=1, 4, 16)
3. 一轮投机解码的时间分解(draft 生成 + target 验证 + replay)
4. kernel launch 开销
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from stage11_cuda_target import RWKV7Target2p9BCuda
from stage9_logit_fusion_01b import TARGET_WEIGHTS, LAMBADA_FILE, DEVICE
from rwkv_tokenizer import TRIE_TOKENIZER

DRAFT_WEIGHTS = Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1d-0.4b-20260210-ctx8192.pth")


def cuda_sync():
    torch.cuda.synchronize()


def bench_forward(model, state_fn, T: int, warmup: int = 5, repeat: int = 20):
    """测量 forward [1, T] 单次时间(ms)"""
    state = state_fn(1)
    tokens = torch.zeros((1, T), device=DEVICE, dtype=torch.long)
    # warmup
    for _ in range(warmup):
        s = state_fn(1)
        model.forward(tokens, s, [])
    cuda_sync()
    # bench
    s = state_fn(1)
    t0 = time.perf_counter()
    for _ in range(repeat):
        model.forward(tokens, s, [])
    cuda_sync()
    elapsed = (time.perf_counter() - t0) / repeat * 1000
    return elapsed


def profile_speculative_round(draft, target, t_state, d_state, t_logits, d_logits, K=4, alpha=2.0, n_rounds=50):
    """profiling 一轮投机解码的各阶段时间(ms)"""
    # 预热
    for _ in range(3):
        _run_one_round(draft, target, t_state, d_state, t_logits, d_logits, K, alpha)
    cuda_sync()

    # 分阶段计时
    times = {
        "draft_gen": 0,      # draft 生成 K 候选
        "target_verify": 0,  # target forward [1, K] 验证
        "decision": 0,        # argmax + 比较
        "replay_target": 0,   # 拒绝时 target replay
        "replay_draft": 0,    # 拒绝时 draft replay
        "clone_state": 0,     # state clone
        "total": 0,
    }
    rounds_with_reject = 0

    t_total = time.perf_counter()
    for _ in range(n_rounds):
        # === 阶段 1: draft 生成 K 个候选 ===
        cuda_sync()
        t0 = time.perf_counter()
        d_state_backup = [s.clone() for s in d_state]
        draft_tokens = []
        fused = d_logits + alpha * t_logits
        tok = fused.argmax(dim=-1).item()
        draft_tokens.append(tok)
        for k in range(1, K):
            next_t = torch.tensor([[tok]], device=DEVICE, dtype=torch.long)
            d_l, _ = draft.forward(next_t, d_state, [])
            tok = d_l[0, -1].argmax().item()
            draft_tokens.append(tok)
        cuda_sync()
        times["draft_gen"] += (time.perf_counter() - t0)

        # === 阶段 2: target forward [1, K] ===
        cuda_sync()
        t0 = time.perf_counter()
        t_state_backup = [s.clone() for s in t_state]
        draft_tensor = torch.tensor([draft_tokens], device=DEVICE, dtype=torch.long)
        t_logits_full, _ = target.forward(draft_tensor, t_state_backup, [])
        cuda_sync()
        times["target_verify"] += (time.perf_counter() - t0)
        times["clone_state"] += 0  # clone 已计入

        # === 阶段 3: 决策 ===
        t0 = time.perf_counter()
        accepted = 0
        t_pred = t_logits[0].argmax().item()
        if draft_tokens[0] == t_pred:
            accepted = 1
            for i in range(1, K):
                t_pred = t_logits_full[0, i-1].argmax().item()
                if draft_tokens[i] == t_pred:
                    accepted += 1
                else:
                    break
        cuda_sync()
        times["decision"] += (time.perf_counter() - t0)

        # === 阶段 4: replay(如果拒绝)===
        if accepted < K:
            rounds_with_reject += 1
            replay = draft_tokens[:accepted] + [t_pred]
            replay_tensor = torch.tensor([replay], device=DEVICE, dtype=torch.long)

            cuda_sync()
            t0 = time.perf_counter()
            t_logits_replay, _ = target.forward(replay_tensor, t_state, [])
            cuda_sync()
            times["replay_target"] += (time.perf_counter() - t0)

            t0 = time.perf_counter()
            d_state = [s.clone() for s in d_state_backup]
            d_logits_replay, _ = draft.forward(replay_tensor, d_state, [])
            cuda_sync()
            times["replay_draft"] += (time.perf_counter() - t0)
            t_logits = t_logits_replay[:, -1, :]
            d_logits = d_logits_replay[:, -1, :]
        else:
            t_state = t_state_backup
            t_logits = t_logits_full[:, -1, :]
            d_logits = d_l[:, -1, :]

    times["total"] = (time.perf_counter() - t_total) * 1000 / n_rounds
    for k in times:
        if k != "total":
            times[k] = times[k] * 1000 / n_rounds
    times["reject_rate"] = rounds_with_reject / n_rounds
    return times


def _run_one_round(draft, target, t_state, d_state, t_logits, d_logits, K, alpha):
    """跑一轮,用于预热"""
    d_state_backup = [s.clone() for s in d_state]
    draft_tokens = []
    fused = d_logits + alpha * t_logits
    tok = fused.argmax(dim=-1).item()
    draft_tokens.append(tok)
    for k in range(1, K):
        next_t = torch.tensor([[tok]], device=DEVICE, dtype=torch.long)
        d_l, _ = draft.forward(next_t, d_state, [])
        tok = d_l[0, -1].argmax().item()
        draft_tokens.append(tok)
    t_state_backup = [s.clone() for s in t_state]
    draft_tensor = torch.tensor([draft_tokens], device=DEVICE, dtype=torch.long)
    t_logits_full, _ = target.forward(draft_tensor, t_state_backup, [])
    accepted = 0
    t_pred = t_logits[0].argmax().item()
    if draft_tokens[0] == t_pred:
        accepted = 1
        for i in range(1, K):
            t_pred = t_logits_full[0, i-1].argmax().item()
            if draft_tokens[i] == t_pred:
                accepted += 1
            else:
                break
    return accepted


def main():
    print("=" * 70)
    print("stage12 速度瓶颈 profiling")
    print("=" * 70)

    tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))
    print("\n加载 target 2.9B...")
    target = RWKV7Target2p9BCuda(TARGET_WEIGHTS)
    print("加载 draft 0.4B...")
    draft = RWKV7Target2p9BCuda(DRAFT_WEIGHTS)

    # === 1. 单次 forward 时间 ===
    print("\n" + "=" * 70)
    print("[1] 单次 forward 时间(ms)")
    print("=" * 70)
    print(f"{'模型':<12} {'T=1':<12} {'T=4':<12} {'T=16':<12}")
    for name, m, sf in [("target 2.9B", target, target.zero_state), ("draft 0.4B", draft, draft.zero_state)]:
        t1 = bench_forward(m, sf, 1)
        t4 = bench_forward(m, sf, 4)
        t16 = bench_forward(m, sf, 16)
        print(f"{name:<12} {t1:<12.2f} {t4:<12.2f} {t16:<12.2f}")

    # === 2. 一轮投机解码的时间分解 ===
    print("\n" + "=" * 70)
    print("[2] 一轮投机解码的时间分解(ms) K=4, α=2.0")
    print("=" * 70)

    prompt = "中国的首都是哪里"
    ids = tokenizer.encode(prompt)
    ctx = torch.tensor([ids], device=DEVICE, dtype=torch.long)

    t_state = target.zero_state(1)
    d_state = draft.zero_state(1)
    t_logits_full, _ = target.forward(ctx, t_state, [])
    d_logits_full, _ = draft.forward(ctx, d_state, [])
    t_logits = t_logits_full[:, -1, :]
    d_logits = d_logits_full[:, -1, :]

    times = profile_speculative_round(draft, target, t_state, d_state, t_logits, d_logits, K=4, alpha=2.0, n_rounds=50)
    total = times["total"]
    print(f"{'阶段':<20} {'时间(ms)':<12} {'占比':<10}")
    print("-" * 42)
    for k in ["draft_gen", "target_verify", "decision", "replay_target", "replay_draft"]:
        pct = times[k] / total * 100
        print(f"{k:<20} {times[k]:<12.2f} {pct:<10.1f}%")
    print("-" * 42)
    print(f"{'total':<20} {total:<12.2f}")
    print(f"拒绝率: {times['reject_rate']:.1%}")

    # === 3. 理论 vs 实际 ===
    print("\n" + "=" * 70)
    print("[3] 理论 vs 实际加速比分析")
    print("=" * 70)
    t_target_1 = bench_forward(target, target.zero_state, 1, repeat=30)
    t_draft_1 = bench_forward(draft, draft.zero_state, 1, repeat=30)
    t_target_4 = bench_forward(target, target.zero_state, 4, repeat=30)
    print(f"target forward [1,1]: {t_target_1:.2f}ms")
    print(f"target forward [1,4]: {t_target_4:.2f}ms")
    print(f"draft forward [1,1]:  {t_draft_1:.2f}ms")
    print(f"")
    print(f"一轮投机(K=4):")
    print(f"  draft 生成 3 个候选: 3 × {t_draft_1:.2f} = {3*t_draft_1:.2f}ms")
    print(f"  target 验证: {t_target_4:.2f}ms")
    print(f"  理论总计(无拒绝): {3*t_draft_1 + t_target_4:.2f}ms")
    print(f"  实际总计(含拒绝): {total:.2f}ms")
    print(f"  replay 开销占比: {(times['replay_target']+times['replay_draft'])/total*100:.1f}%")
    print(f"")
    print(f"纯 target 生成 2 token(接受率 50%): 2 × {t_target_1:.2f} = {2*t_target_1:.2f}ms")
    print(f"  → 加速比: {2*t_target_1/total:.2f}x")
    print(f"纯 target 生成 3 token(接受率 75%): 3 × {t_target_1:.2f} = {3*t_target_1:.2f}ms")
    print(f"  → 加速比: {3*t_target_1/total:.2f}x")


if __name__ == "__main__":
    main()
