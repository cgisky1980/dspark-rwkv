"""DSpark → RWKV 复现 · 阶段4并发实测

修正 stage4_e2e.py 的两个问题：
1. 公式 bug：t_round = t_draft + (vl+1)*t_target 错把验证当成 vl 次独立 forward
   实际 target 一次 forward [B, T=vl] 就能验证 vl 个 token
2. 只测 batch=1：DSpark 加速依赖并发批处理，必须测 batch=N

加速比公式（N 个并发请求批处理）：
- 纯 target：K × T_target(N, T=1)
- DSpark：(K/exp_tok) × (T_draft(N) + T_target(N, T=vl))
- 加速比 = exp_tok × T_target(N,1) / (T_draft(N) + T_target(N,vl))
"""
import time
import torch
import torch.nn.functional as F
from stage2_target import RWKV7Target, WEIGHTS
from stage2_train_v3 import (
    DSparkDraft, VOCAB, D_DRAFT, CTX, TARGET_LAYERS, load_data, sample_batch, SQRT_E
)

DEVICE = "cuda"
BLOCK = 4
ACCEPT = [0.934, 0.998, 0.998, 0.988]  # v3 接受率


def bench_target_batch(target, batch, T, n_steps=30):
    """测 target forward [batch, T] 的时间（含 state 维护）。

    模拟自回归：每步 forward T 个 token，更新 state。
    batch=N, T=1 → 纯 target 并发基线
    batch=N, T=vl → DSpark 验证
    """
    state = target.zero_state(batch, device=DEVICE)
    tokens = torch.randint(0, VOCAB, (batch, T), device=DEVICE)
    return_hids = TARGET_LAYERS if T > 1 else []
    # warmup
    with torch.no_grad():
        for _ in range(5):
            target.forward(tokens, state, return_hidden_layers=return_hids)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_steps):
            target.forward(tokens, state, return_hidden_layers=return_hids)
    torch.cuda.synchronize()
    return (time.time() - t0) / n_steps


def bench_draft_batch(draft, batch, n_steps=50):
    """测 draft forward [batch] 的时间（生成一个 block）。"""
    anchor = torch.randint(0, VOCAB, (batch,), device=DEVICE)
    ctx_hidden = torch.randn(batch, CTX, 768 * len(TARGET_LAYERS), device=DEVICE)
    prev = torch.zeros(batch, BLOCK, dtype=torch.long, device=DEVICE)
    prev[:, 0] = anchor
    with torch.no_grad():
        for _ in range(5):
            draft(anchor, ctx_hidden, prev)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_steps):
            draft(anchor, ctx_hidden, prev)
    torch.cuda.synchronize()
    return (time.time() - t0) / n_steps


def expected_tokens(accept, vl):
    """给定 verify_len，算每轮期望接受 token 数。"""
    exp = 0.0
    prob_all = 1.0
    for t in range(vl):
        exp += prob_all * accept[t]
        prob_all *= accept[t]
    exp += 1.0  # 拒绝后 target 补 1 个
    return exp


def main():
    print("=" * 70)
    print("阶段4并发实测：batch=N 下 target/draft 时间 + 加速比")
    print("=" * 70)

    target = RWKV7Target(WEIGHTS)
    target.z = {k: v.to(DEVICE) for k, v in target.z.items()}

    # 训练 draft（1000 步快速收敛，只为测时间）
    print("\n训练 draft 1000 步...")
    tokens, hids_dict = load_data()
    tokens = tokens.to(DEVICE)
    hids_dict = {k: v.to(DEVICE) for k, v in hids_dict.items()}
    draft = DSparkDraft(BLOCK).to(DEVICE)
    opt = torch.optim.Adam(draft.parameters(), lr=1e-4)
    for step in range(1000):
        anc, ch, prev, tgt = sample_batch(tokens, hids_dict, 64, BLOCK, CTX)
        dl, conf = draft(anc, ch, prev)
        loss = F.cross_entropy(dl.reshape(-1, VOCAB), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        opt.step()
    draft.eval()
    print("draft 训练完成\n")

    batches = [1, 2, 4, 8, 16, 32]
    verify_lens = [1, 2, 3, 4]

    # 1. 测 target 时间矩阵
    print("=" * 70)
    print("target forward 时间矩阵 (ms)")
    print("=" * 70)
    print(f"{'batch\\T':<10} {'T=1':<12} {'T=2':<12} {'T=3':<12} {'T=4':<12}")
    t_target = {}  # t_target[batch][T]
    for b in batches:
        row = [f"batch={b:<4}"]
        t_target[b] = {}
        for T in [1, 2, 3, 4]:
            dt = bench_target_batch(target, b, T)
            t_target[b][T] = dt
            row.append(f"{dt*1000:.2f}")
        print(f"{row[0]:<10} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12}")

    # 2. 测 draft 时间
    print("\n" + "=" * 70)
    print("draft forward 时间 (ms)")
    print("=" * 70)
    t_draft = {}
    for b in batches:
        dt = bench_draft_batch(draft, b)
        t_draft[b] = dt
        print(f"  batch={b:<4}  draft={dt*1000:.2f} ms  draft/target(T=1)={dt/t_target[b][1]:.3f}")

    # 3. 算并发加速比
    print("\n" + "=" * 70)
    print("并发加速比（纯 target 基线 vs DSpark）")
    print("=" * 70)
    print(f"接受率: {ACCEPT}")
    print()
    for vl in verify_lens:
        exp_tok = expected_tokens(ACCEPT, vl)
        print(f"\n--- verify_len={vl} (每轮期望 {exp_tok:.3f} token) ---")
        print(f"{'batch':<8} {'T_target(1)ms':<14} {'T_target(vl)ms':<14} {'T_draftms':<10} {'DSpark tok/s':<14} {'纯target tok/s':<16} {'加速比'}")
        for b in batches:
            tt1 = t_target[b][1]
            ttvl = t_target[b][vl]
            td = t_draft[b]
            # DSpark: 每轮 = draft + target(vl)，出 exp_tok token
            dspark_tp = exp_tok / (td + ttvl)
            # 纯 target: 每步 = target(1)，出 1 token
            target_tp = 1.0 / tt1
            speedup = dspark_tp / target_tp
            print(f"{b:<8} {tt1*1000:<14.2f} {ttvl*1000:<14.2f} {td*1000:<10.2f} {dspark_tp:<14.1f} {target_tp:<16.1f} {speedup:.2f}x")

    # 4. 最优配置汇总
    print("\n" + "=" * 70)
    print("最优配置汇总")
    print("=" * 70)
    best = (0, 0, 0, 0)  # (speedup, batch, vl, exp_tok)
    for vl in verify_lens:
        exp_tok = expected_tokens(ACCEPT, vl)
        for b in batches:
            tt1 = t_target[b][1]
            ttvl = t_target[b][vl]
            td = t_draft[b]
            dspark_tp = exp_tok / (td + ttvl)
            target_tp = 1.0 / tt1
            sp = dspark_tp / target_tp
            if sp > best[0]:
                best = (sp, b, vl, exp_tok)
    print(f"最优: batch={best[1]}, verify_len={best[2]}, 加速比={best[0]:.2f}x")
    print(f"  (每轮期望 {best[3]:.3f} token)")


if __name__ == "__main__":
    main()
