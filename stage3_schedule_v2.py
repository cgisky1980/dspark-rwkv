"""阶段3v2：用 v3 高接受率数据重算调度收益。"""
import numpy as np
import json
from pathlib import Path

# v3 结果（5层hidden, 5000步）
RWKV_V3 = {
    4: [0.934, 0.998, 0.998, 0.988],
    6: [0.840, 1.000, 0.998, 0.994, 0.994, 0.965],
    8: [0.805, 0.996, 1.000, 1.000, 0.996, 0.998, 0.996, 0.996],
}
GRU_V2 = [0.445, 0.590, 0.582, 0.523]  # v2 GRU 对照

T_VERIFY = 1.0
T_DRAFT = 0.3


def simulate(accept, block, verify_len):
    """固定 verify_len 的吞吐。"""
    rng = np.random.default_rng(42)
    n_sim = 20000
    total_tok = 0
    total_time = 0
    for _ in range(n_sim):
        total_time += T_DRAFT + verify_len * T_VERIFY
        n_acc = 0
        for t in range(verify_len):
            if rng.random() < accept[t]:
                n_acc += 1
            else:
                break
        total_tok += n_acc + 1
    return total_tok / total_time


def optimal(accept, block):
    best_tp, best_len = 0, 1
    for vl in range(1, block + 1):
        tp = simulate(accept, block, vl)
        if tp > best_tp:
            best_tp, best_len = tp, vl
    return best_tp, best_len


def main():
    print("="*70)
    print("阶段3v2：用 v3 高接受率重算调度收益")
    print("="*70)
    print(f"假设: T_verify={T_VERIFY}, T_draft={T_DRAFT}")
    print(f"纯 target 基线: {1.0/T_VERIFY:.3f} tok/s\n")

    results = {}
    configs = [
        ("GRU v2 block=4", GRU_V2, 4),
        ("RWKV-7 v3 block=4", RWKV_V3[4], 4),
        ("RWKV-7 v3 block=6", RWKV_V3[6], 6),
        ("RWKV-7 v3 block=8", RWKV_V3[8], 8),
    ]
    for name, accept, block in configs:
        baseline = 1.0 / T_VERIFY
        # 固定全 block 验证
        tp_full = simulate(accept, block, block)
        # 最优调度
        tp_opt, opt_len = optimal(accept, block)
        # 各 verify_len 的吞吐
        tp_by_len = {vl: simulate(accept, block, vl) for vl in range(1, block + 1)}

        print(f"--- {name} ---")
        print(f"  accept: {accept}")
        print(f"  固定全block验证: {tp_full:.3f} tok/s ({tp_full/baseline:.2f}x)")
        print(f"  各 verify_len 吞吐: { {vl: f'{tp:.3f}({tp/baseline:.2f}x)' for vl, tp in tp_by_len.items()} }")
        print(f"  最优调度 verify_len={opt_len}: {tp_opt:.3f} tok/s ({tp_opt/baseline:.2f}x)")
        print()
        results[name] = {
            "accept": accept, "baseline": baseline,
            "full_block": tp_full, "optimal": (tp_opt, opt_len),
            "by_len": tp_by_len,
        }

    # 汇总
    print(f"{'='*70}")
    print(f"{'配置':<24} {'固定全block':<14} {'最优调度':<14} {'最优len'}")
    for name, r in results.items():
        b = r["baseline"]
        print(f"{name:<24} {r['full_block']/b:.2f}x          {r['optimal'][0]/b:.2f}x          {r['optimal'][1]}")

    out = Path(__file__).parent / "data" / "stage3_schedule_v2.json"
    serializable = {name: {
        "accept": r["accept"], "baseline": r["baseline"],
        "full_block": r["full_block"], "optimal": {"tp": r["optimal"][0], "len": r["optimal"][1]},
        "by_len": {str(k): v for k, v in r["by_len"].items()},
    } for name, r in results.items()}
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n保存到 {out}")


if __name__ == "__main__":
    main()
