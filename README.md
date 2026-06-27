# DSpark-RWKV

> Adapting DSpark semi-autoregressive speculative decoding to RWKV-7 target, achieving **3.75x speedup** in concurrent batched inference.

This repository reproduces and adapts the [DSpark](https://github.com/deepseek-ai/DeepSpec) framework (jointly proposed by DeepSeek and Peking University) to the [RWKV-7](https://github.com/BlinkDL/RWKV-LM) linear RNN target. The original DSpark paper experiments with Transformer autoregressive targets (Qwen3 series); this work verifies its feasibility on the RWKV-7 Delta Rule architecture.

**Paper**: [English](docs/paper_en.md) | [中文](docs/paper_zh.md)

## Key Results

| Metric | Value |
|---|---|
| Target | RWKV-7 0.1B (L=12, C=768, H=12, N=64, V=65536) |
| Acceptance Rate | **97.95%** (block=4, v3 training) |
| RWKV-7 vs GRU Sequential Head | **+31%** (acceptance lead) |
| Concurrent Speedup (batch=2, verify_len=4) | **3.75x** |
| Speedup Range (batch=1-32) | 1.56x – 3.75x |

## Main Contributions

1. **RWKV-7 DPLR as DSpark Sequential Head**: Ablation studies show token shift (+0.435 acceptance) and LayerNorm (+0.228 acceptance) are necessary conditions.
2. **Cross-attn Context Length is the Decisive Optimization**: Single-position → 8-position context boosts acceptance from 33% to 70%.
3. **Corrected Speculative Decoding Speedup Formula**: Target verification of vl tokens is 1 forward pass `[B, T=vl]`, not vl independent forwards. This correction overturns the erroneous conclusion that "speculative decoding cannot accelerate on single GPU".

## Repository Structure

```
dspark-rwkv/
├── pyproject.toml                # Dependency config (torch cu124)
├── stage1_experiment.py          # Stage 1: synthetic tasks (single/double key + ablation)
├── stage2_target.py              # RWKV-7 0.1B target (pure PyTorch)
├── stage2_gen_data.py            # 3-layer hidden data generation
├── stage2_gen_data_5layer.py     # 5-layer hidden data generation
├── stage2_train.py               # v1 training (cross-attn single position)
├── stage2_train_v2.py            # v2 optimization (8-position context + confidence BCE)
├── stage2_train_v3.py            # v3 comprehensive optimization (5 layers + 5000 steps, 97%+)
├── stage3_schedule.py            # Offline scheduling evaluation (v1 data)
├── stage3_schedule_v2.py         # Offline scheduling evaluation (v3 data)
├── stage4_concurrent.py          # Stage 4: concurrent measurement (core experiment)
├── archive/                      # Historical scripts
│   ├── experiment_v1.py
│   ├── experiment_xor.py
│   ├── stage2_benchmark.py
│   ├── stage2_verify_target.py
│   └── stage4_e2e_batch1_buggy.py
├── docs/                         # Documentation and papers
│   ├── paper_zh.md               # Chinese paper
│   ├── paper_en.md               # English paper
│   ├── 方案.md
│   ├── RWKV7公式参考.md
│   ├── 总报告.md
│   ├── 阶段1实验报告.md
│   ├── 阶段1实验汇总表.md
│   ├── 阶段2实验报告.md
│   ├── 阶段2v2优化报告.md
│   └── 阶段3-4报告.md
├── weights/                      # Weights directory (manual download required)
└── data/                         # Data directory (generated artifacts)
```

## Quick Start

### 1. Environment Setup

Requires Python 3.12+ and NVIDIA GPU (RTX 3090+ recommended; experiments used RTX 2080 Ti 22GB).

```bash
# Install uv (if not installed)
pip install uv

# Sync dependencies (includes CUDA 12.4 torch)
uv sync
```

### 2. Download RWKV-7 0.1B Weights

```bash
mkdir -p weights
# Download rwkv7-0.1b.pth from HuggingFace and place in weights/ directory
# https://huggingface.co/BlinkDL/rwkv-7-pile/tree/main
```

### 3. Run Experiments

```bash
# Stage 1: synthetic tasks (verify RWKV-7 DPLR sequential head + ablation)
uv run python stage1_experiment.py

# Stage 2: generate training data (5-layer hidden)
uv run python stage2_gen_data_5layer.py

# Stage 2 v3: training (5000 steps, ~30 minutes)
uv run python stage2_train_v3.py

# Stage 3: offline scheduling evaluation
uv run python stage3_schedule_v2.py

# Stage 4: concurrent measurement (core experiment)
uv run python stage4_concurrent.py
```

## Four-Stage Experimental Pipeline

| Stage | Content | Key Result |
|---|---|---|
| 1 | Sequential head unit test (synthetic) | token shift +0.435, LayerNorm +0.228 |
| 2 | Real RWKV target + DSpark dual-track | 97%+ acceptance, RWKV leads GRU by 31% |
| 3 | Offline scheduling evaluation | verify_len=1 offline prediction 1.49x |
| 4 | End-to-end concurrent measurement | batch=2, verify_len=4 achieves **3.75x** |

## Speedup Formula

Let $T_{draft}$ be the time to generate one block, $T_{target}(N, T)$ be the target forward time at batch=N and sequence length T, and $\tau$ be the expected accepted tokens per round.

$$
\text{speedup} = \frac{\tau \cdot T_{target}(N, 1)}{T_{draft}(N) + T_{target}(N, \text{vl})}
$$

**Key Observation**: Target verification of vl tokens requires only 1 forward pass `[B, T=vl]` (not vl independent forwards). GPU parallel processing of multiple tokens takes far less time than vl times the single-token time—this is the core source of DSpark acceleration.

## Comparison with Original DSpark Paper

| Dimension | DSpark Paper | This Work |
|---|---|---|
| Target | Qwen3-4B/8B/14B (Transformer) | RWKV-7 0.1B (linear RNN) |
| Sequential Head | GRU/RNN | RWKV-7 DPLR (31% lead over GRU) |
| Acceptance Rate | ~70-80% | 97%+ (v3) |
| Speedup | +60%~85% user speed (V4-Flash) | 1.56-3.75x throughput |
| Deployment | Multi-concurrent serving (DeepSeek-V4) | Single GPU batch=N |

## References

- DSpark paper: https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf
- RWKV-7 repository: https://github.com/BlinkDL/RWKV-LM
- Albatross inference library: https://github.com/BlinkDL/Albatross
- RWKV-7 numpy implementation: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py

## Citation

If you use this work, please cite:

```bibtex
@misc{dspark-rwkv2026,
  title={DSpark-RWKV: Adapting Semi-Autoregressive Speculative Decoding to RWKV-7},
  author={cgisky1980},
  year={2026},
  url={https://github.com/cgisky1980/dspark-rwkv}
}
```

And cite the original DSpark paper:

```bibtex
@misc{cheng2025dspark,
  title={DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation},
  author={Cheng, Xin and Yu, Xingkai and Shao, Chenze and others},
  year={2025},
  url={https://github.com/deepseek-ai/DeepSpec}
}
```

## License

MIT License - see [LICENSE](LICENSE)
