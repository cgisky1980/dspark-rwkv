# DSpark-RWKV

> Adapting DSpark semi-autoregressive speculative decoding to RWKV-7 target.

**Language**: [English](README.md) | [中文](README_zh.md)

**Paper**: [English](docs/paper_en.md) | [中文](docs/paper_zh.md)

This repository reproduces and adapts the [DSpark](https://github.com/deepseek-ai/DeepSpec) framework (jointly proposed by DeepSeek and Peking University) to the [RWKV-7](https://github.com/BlinkDL/RWKV-LM) linear RNN target. The original DSpark paper experiments with Transformer autoregressive targets (Qwen3 series); this work verifies its feasibility on the RWKV-7 Delta Rule architecture.

## ⚠️ Errata (2026-06-28)

**The previously reported 97%+ acceptance rate and 3.75x speedup are INVALID and have been retracted.** Root causes:

1. **Data leakage (most severe)**: `stage2_train_v3.py` used the same 512 sequences for both training and validation (`N_eval=512` covered the entire training set, no train/val/test split). The reported 97.95% acceptance was memorization, not generalization.
2. **3.75x speedup was derived from the invalid acceptance rate** and is therefore unreliable.
3. **K-state concurrent verification scheme** (stages 9-13): all variants achieved speedup < 1× due to serial draft forwards (406 draft iterations vs 41 target iterations).
4. **CUDA / CUDA Graph single-point optimizations** reduced draft forward from 6.5ms to 2.4ms but did not yield end-to-end speedup (architecture-level serialization bottleneck remained).
5. **DSpark architecture misunderstanding**: an earlier branch attempted to modify a 0.4B RWKV-7 into a drafter; the correct DSpark design is an independent small model trained from scratch.

**Valid contributions retained**:
- Token shift (+0.435) and LayerNorm (+0.228) ablation on synthetic tasks (no data leakage).
- Correction of the speculative decoding speedup formula: target verification of `vl` tokens is 1 forward `[B, T=vl]`, not `vl` independent forwards.
- Identification of cross-attention context length as the decisive optimization factor.

**Ongoing rework** (correct methodology):
- Independent train/val/test split (80/10/10) with 100K sequences from `mlabonne/open-perfectblend`.
- Larger 500M-parameter drafter (D=1280, L=10, RWKV-7 head rank=512).
- Target: RWKV-7 2.9B (g1a series) with 5-layer hidden [0,3,6,9,11].
- Prelinary result on small scale (10 prompts × 40 tokens, K=2, α=0.5): 85.83% acceptance with g1a-0.4B draft + g1a-2.9B target (after fixing WKV state dimension order bug).

See [`参考/dspark_rwkv/失败存档总结.md`](../参考/dspark_rwkv/失败存档总结.md) for the full failure archive.

## Main Contributions (Retained)

1. **RWKV-7 DPLR as DSpark Sequential Head**: Ablation studies on synthetic tasks show token shift (+0.435 acceptance) and LayerNorm (+0.228 acceptance) are necessary conditions.
2. **Cross-attn Context Length is the Decisive Optimization**: Single-position → 8-position context boosts acceptance from 33% to 70% (on synthetic tasks; real-target result invalidated by data leakage).
3. **Corrected Speculative Decoding Speedup Formula**: Target verification of `vl` tokens is 1 forward pass `[B, T=vl]`, not `vl` independent forwards. This correction overturns the erroneous conclusion that "speculative decoding cannot accelerate on single GPU".

## Repository Structure

```
dspark-rwkv/
├── pyproject.toml                # Dependency config (torch cu124)
├── stage1_experiment.py          # Stage 1: synthetic tasks (single/double key + ablation)
├── stage2_target.py              # RWKV-7 target (pure PyTorch)
├── stage2_gen_data.py            # 3-layer hidden data generation (legacy)
├── stage2_gen_data_5layer.py     # 5-layer hidden data generation (legacy, no split)
├── stage2_gen_data_fast.py       # Chunked data generation (fp16, 100K sequences, train/val/test split)
├── stage2_train.py               # v1 training (cross-attn single position)
├── stage2_train_v2.py            # v2 optimization (8-position context + confidence BCE)
├── stage2_train_v3.py            # v3 training (chunked loading, 500M drafter)
├── stage3_schedule.py            # Offline scheduling evaluation (v1 data)
├── stage3_schedule_v2.py         # Offline scheduling evaluation (v3 data)
├── stage4_concurrent.py          # Stage 4: concurrent measurement
├── rwkv_tokenizer.py             # RWKV BPE tokenizer
├── rwkv_vocab_v20230424.txt      # RWKV vocab
├── cuda/                         # CUDA kernels (WKV fp16/fp32)
├── albatross_ref/                # Albatross reference code
├── archive/                      # Historical scripts
│   ├── experiment_v1.py
│   ├── experiment_xor.py
│   ├── stage2_benchmark.py
│   ├── stage2_verify_target.py
│   ├── stage4_e2e_batch1_buggy.py
│   └── stage9_13_failed/          # K-state scheme (speedup < 1×, archived)
├── docs/                         # Documentation and papers
│   ├── paper_zh.md               # Chinese paper
│   ├── paper_en.md               # English paper
│   ├── RWKV7公式参考.md
│   └── archive_failed/           # Failed experiment reports (archived)
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

### 2. Download RWKV-7 Weights

```bash
mkdir -p weights
# Download RWKV-7 2.9B (g1a) weights from HuggingFace and place in weights/ directory
# https://huggingface.co/BlinkDL/rwkv-7-g1/tree/main
# Also download a draft-capable model, e.g., rwkv7-g1d-0.4b for speculative decoding baseline
```

### 3. Run Experiments

```bash
# Stage 1: synthetic tasks (verify RWKV-7 DPLR sequential head + ablation)
uv run python stage1_experiment.py

# Stage 2 v3 (rework): generate 100K sequences with proper train/val/test split
uv run python stage2_gen_data_fast.py

# Stage 2 v3 training (chunked loading, 500M drafter, 20000 steps)
uv run python -c "from stage2_train_v3 import run; run(8, 20000)"
```

## Speedup Formula

Let $T_{draft}$ be the time to generate one block, $T_{target}(N, T)$ be the target forward time at batch=N and sequence length T, and $\tau$ be the expected accepted tokens per round.

$$
\text{speedup} = \frac{\tau \cdot T_{target}(N, 1)}{T_{draft}(N) + T_{target}(N, \text{vl})}
$$

**Key Observation**: Target verification of `vl` tokens requires only 1 forward pass `[B, T=vl]` (not `vl` independent forwards). GPU parallel processing of multiple tokens takes far less time than `vl` times the single-token time—this is the core source of DSpark acceleration.

## References

- DSpark paper: https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf
- RWKV-7 repository: https://github.com/BlinkDL/RWKV-LM
- Albatross inference library: https://github.com/BlinkDL/Albatross
- RWKV-7 numpy implementation: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py
- Training data: https://huggingface.co/datasets/mlabonne/open-perfectblend

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
