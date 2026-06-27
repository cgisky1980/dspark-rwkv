"""DSpark → RWKV 复现 · 阶段3：离线调度评估

基于 v2 训练好的模型（置信度 + 接受率），离线评估"如果按置信度截断"的吞吐收益。
不接真实推理引擎，只算理论吞吐曲线。

模型：DSpark 论文的硬件感知前缀调度器简化版
- 给定每个请求的置信度序列，决定验证多长
- 目标：最大化全局吞吐（token/s）

简化假设：
- 单 GPU，batch=1（最坏情况，对应 web-rwkv 场景）
- target verify 1 token 耗时 T_verify
- draft 生成整个 block 耗时 T_draft（一次性并行）
- 接受率从训练数据统计
"""
import torch
import numpy as np
from pathlib import Path
import json

# v2 训练的 RWKV-7 模型在各位置的接受率（从阶段2v2结果）
RWKV_V2_ACCEPT = [0.551, 0.822, 0.799, 0.633]  # block=4
GRU_V2_ACCEPT = [0.445, 0.590, 0.582, 0.523]

# target/draft 耗时比（从 benchmark：target forward 369ms, draft 应该快很多）
# 假设 draft 1 个 block 耗时 = 0.3 × target 1 token 耗时（draft 小得多）
# target verify k 个 token 耗时 = k × T_verify（串行 RWKV state 推进）
T_VERIFY = 1.0   # 归一化
T_DRAFT = 0.3    # draft 生成整个 block


def simulate_speculative(accept_rates, block_size, strategy="fixed", conf_threshold=0.5):
    """模拟推测解码的吞吐。
    accept_rates: 各位置接受率
    strategy:
      - "fixed": 固定验证 block_size 个 token（基线，无调度）
      - "confidence": 按置信度阈值截断（DSpark 调度器简化版）
    返回：平均每步生成的 token 数（吞吐指标）
    """
    n_sim = 10000
    rng = np.random.default_rng(42)
    total_tokens = 0
    total_time = 0
    for _ in range(n_sim):
        # draft 生成整个 block（并行，固定耗时）
        total_time += T_DRAFT
        # 决定验证多少个
        if strategy == "fixed":
            verify_len = block_size
        else:  # confidence
            # 简化：从位置0开始，遇到接受率<阈值就截断
            verify_len = block_size
            for t in range(block_size):
                if accept_rates[t] < conf_threshold:
                    verify_len = t if t > 0 else 1
                    break
        # target 验证 verify_len 个 token（串行）
        total_time += verify_len * T_VERIFY
        # 统计接受多少个（简化：按接受率伯努利采样）
        n_accepted = 0
        for t in range(verify_len):
            if rng.random() < accept_rates[t]:
                n_accepted += 1
            else:
                break  # 首个拒绝后停止
        total_tokens += n_accepted + 1  # +1 是至少接受1个（拒绝后回退到 target）
    return total_tokens / total_time


def simulate_optimal_schedule(accept_rates, block_size):
    """最优调度：穷举所有截断点，找吞吐最高的。"""
    best_throughput = 0
    best_len = 1
    for verify_len in range(1, block_size + 1):
        # 验证 verify_len 个的期望接受数
        exp_accept = 0
        prob_all_accept = 1.0
        for t in range(verify_len):
            exp_accept += prob_all_accept * accept_rates[t]
            prob_all_accept *= accept_rates[t]
        # 吞吐 = 期望 token / 时间
        time = T_DRAFT + verify_len * T_VERIFY
        throughput = (exp_accept + 1) / time  # +1 保证至少1个
        if throughput > best_throughput:
            best_throughput = throughput
            best_len = verify_len
    return best_throughput, best_len


def main():
    print("="*70)
    print("阶段3：离线调度评估")
    print("="*70)
    print(f"假设: T_verify={T_VERIFY}, T_draft={T_DRAFT} (draft/block = {T_DRAFT/T_VERIFY:.1f}x verify)")
    print()

    results = {}
    for name, accept in [("GRU v2", GRU_V2_ACCEPT), ("RWKV-7 v2", RWKV_V2_ACCEPT)]:
        print(f"\n--- {name} (accept={accept}) ---")
        block = len(accept)
        # 基线：固定验证全 block
        tp_fixed = simulate_speculative(accept, block, "fixed")
        # 置信度截断（不同阈值）
        tp_confs = {}
        for thresh in [0.3, 0.5, 0.7]:
            tp = simulate_speculative(accept, block, "confidence", thresh)
            tp_confs[thresh] = tp
        # 最优调度（穷举）
        tp_opt, opt_len = simulate_optimal_schedule(accept, block)
        # 无推测解码基线（纯 target）
        tp_baseline = 1.0 / T_VERIFY  # 每 T_verify 生成 1 token

        print(f"  纯 target 基线: {tp_baseline:.3f} tok/s")
        print(f"  固定验证 block={block}: {tp_fixed:.3f} tok/s (加速比 {tp_fixed/tp_baseline:.2f}x)")
        for thresh, tp in tp_confs.items():
            print(f"  置信度截断 thresh={thresh}: {tp:.3f} tok/s (加速比 {tp/tp_baseline:.2f}x)")
        print(f"  最优调度 verify_len={opt_len}: {tp_opt:.3f} tok/s (加速比 {tp_opt/tp_baseline:.2f}x)")

        results[name] = {
            "accept_rates": accept,
            "baseline": tp_baseline,
            "fixed": tp_fixed,
            "confidence": tp_confs,
            "optimal": (tp_opt, opt_len),
        }

    # 汇总
    print(f"\n{'='*70}")
    print("汇总：加速比（相对纯 target 基线）")
    print(f"{'='*70}")
    print(f"{'模型':<16} {'固定block':<12} {'置信度0.5':<12} {'最优调度':<12} {'最优verify_len'}")
    for name, r in results.items():
        fixed_speedup = r["fixed"] / r["baseline"]
        conf_speedup = r["confidence"][0.5] / r["baseline"]
        opt_speedup, opt_len = r["optimal"]
        opt_speedup /= r["baseline"]
        print(f"{name:<16} {fixed_speedup:.2f}x        {conf_speedup:.2f}x        {opt_speedup:.2f}x        {opt_len}")

    # 保存结果
    out = Path(__file__).parent / "data" / "stage3_schedule.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for name, r in results.items():
        serializable[name] = {
            "accept_rates": r["accept_rates"],
            "baseline": r["baseline"],
            "fixed": r["fixed"],
            "confidence": {str(k): v for k, v in r["confidence"].items()},
            "optimal": {"throughput": r["optimal"][0], "verify_len": r["optimal"][1]},
        }
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n结果保存到 {out}")


if __name__ == "__main__":
    main()
