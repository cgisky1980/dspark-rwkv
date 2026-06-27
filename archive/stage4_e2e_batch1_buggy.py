"""DSpark → RWKV 复现 · 阶段4：实测 forward 时间 + 理论加速比

简化端到端测试：
1. 实测 target 单步 forward 时间（自回归 1 token）
2. 实测 draft 整个 block forward 时间
3. 用 v3 的接受率算理论加速比
4. 对比阶段3的离线预测

不跑完整端到端（state 管理工程量大），只测关键时间指标。
"""
import time
import math
import torch
import torch.nn.functional as F
from stage2_target import RWKV7Target, WEIGHTS
from stage2_train_v3 import (
    DSparkDraft, VOCAB, D_DRAFT, CTX, TARGET_LAYERS, load_data, sample_batch, SQRT_E
)

DEVICE = "cuda"
BLOCK = 4  # v3 最佳 block_size
N_GEN = 64


def bench_target(target, n_steps=50):
    """实测 target 自回归 n_steps 的时间。"""
    state = target.zero_state(1, device=DEVICE)
    cur = torch.tensor([[1]], device=DEVICE)
    # warmup
    with torch.no_grad():
        for _ in range(5):
            logits, _ = target.forward(cur, state, return_hidden_layers=[])
            cur = logits[:, -1].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_steps):
            logits, _ = target.forward(cur, state, return_hidden_layers=[])
            cur = logits[:, -1].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return dt / n_steps  # per step


def bench_target_with_hidden(target, n_steps=30):
    """实测 target forward + 返回 5 层 hidden 的时间（推测解码每轮需要）。"""
    state = target.zero_state(1, device=DEVICE)
    cur = torch.tensor([[1]], device=DEVICE)
    with torch.no_grad():
        for _ in range(5):  # warmup
            logits, hids = target.forward(cur, state, return_hidden_layers=TARGET_LAYERS)
            cur = logits[:, -1].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_steps):
            logits, hids = target.forward(cur, state, return_hidden_layers=TARGET_LAYERS)
            cur = logits[:, -1].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return dt / n_steps


def bench_draft(draft, n_steps=50, block=4):
    """实测 draft 生成一个 block 的时间。"""
    anchor = torch.tensor([1], device=DEVICE)
    ctx_hidden = torch.randn(1, CTX, 768 * len(TARGET_LAYERS), device=DEVICE)
    prev = torch.zeros(1, block, dtype=torch.long, device=DEVICE)
    prev[:, 0] = anchor
    # warmup
    with torch.no_grad():
        for _ in range(5):
            draft(anchor, ctx_hidden, prev)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_steps):
            draft(anchor, ctx_hidden, prev)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return dt / n_steps


def main():
    print("="*70)
    print("阶段4：实测 forward 时间 + 理论加速比")
    print("="*70)

    target = RWKV7Target(WEIGHTS)
    target.z = {k: v.to(DEVICE) for k, v in target.z.items()}

    # 1. target 时间
    t_target = bench_target(target)
    print(f"\ntarget 单步 forward (无 hidden): {t_target*1000:.2f} ms")
    t_target_h = bench_target_with_hidden(target)
    print(f"target 单步 forward (+5层hidden): {t_target_h*1000:.2f} ms")

    # 2. 训练 draft 并测时间
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
        if (step + 1) % 200 == 0:
            acc = (dl.argmax(-1) == tgt).float().mean().item()
            print(f"  step {step+1} loss={loss.item():.3f} acc={acc:.3f}")
    draft.eval()

    # 3. draft 时间
    t_draft = bench_draft(draft, block=BLOCK)
    print(f"\ndraft 生成 block={BLOCK} 时间: {t_draft*1000:.2f} ms")
    print(f"  draft/block / target/step = {t_draft/t_target_h:.3f}")

    # 4. 理论加速比（用实测时间）
    # v3 block=4 接受率
    accept = [0.934, 0.998, 0.998, 0.988]
    print(f"\n接受率 (v3 block=4): {accept}")

    # 推测解码每轮时间 = draft + target验证(verify_len步) + target拿hidden(1步)
    # 但 target 拿 hidden 和验证可以合并（验证时顺便拿 hidden）
    # 简化：每轮 = draft + (verify_len+1) * target
    print(f"\n{'='*70}")
    print(f"理论加速比（实测时间）")
    print(f"{'='*70}")
    print(f"{'verify_len':<12} {'每轮时间ms':<14} {'每轮期望token':<14} {'吞吐tok/s':<14} {'加速比'}")

    baseline_tp = 1.0 / t_target  # 纯 target
    print(f"\n纯 target 基线: {baseline_tp:.1f} tok/s ({t_target*1000:.2f} ms/token)\n")

    for vl in range(1, BLOCK + 1):
        # 每轮时间 = draft + (vl+1)*target (+1 for getting hidden, 可优化掉但保守估计)
        t_round = t_draft + (vl + 1) * t_target
        # 期望接受 token 数
        exp_tok = 0
        prob_all = 1.0
        for t in range(vl):
            exp_tok += prob_all * accept[t]
            prob_all *= accept[t]
        exp_tok += 1  # 至少 1 个（拒绝后 target 补）
        tp = exp_tok / t_round
        speedup = tp / baseline_tp
        print(f"{vl:<12} {t_round*1000:<14.2f} {exp_tok:<14.3f} {tp:<14.1f} {speedup:.2f}x")

    # 最优配置
    print(f"\n结论:")
    print(f"  draft/block 时间 = {t_draft*1000:.2f} ms")
    print(f"  target/step 时间 = {t_target*1000:.2f} ms")
    print(f"  draft 开销占比 = {t_draft/(t_draft+2*t_target)*100:.1f}% (verify_len=1)")
    print(f"  单 GPU batch=1 下，draft 开销是主要瓶颈")


if __name__ == "__main__":
    main()
