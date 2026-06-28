# DSpark-RWKV: Adapting Semi-Autoregressive Speculative Decoding to RWKV-7

> **⚠️ Errata (2026-06-28)**: The previously reported 97%+ acceptance rate and 3.75x speedup have been retracted. See §8 "Errata and Rework". Retained contributions: synthetic-task ablation (token shift +0.435, LayerNorm +0.228), speedup formula correction, cross-attn context length decisive optimization. This document retains original experimental descriptions for historical record, but all conclusions referencing "97%+ acceptance" or "3.75x speedup" are invalid.

**Abstract**

Speculative decoding accelerates Large Language Model (LLM) inference by decoupling draft generation from target verification. The recent DSpark framework employs a semi-autoregressive draft architecture (parallel backbone + sequential head) with confidence-scheduled verification, achieving 60%–85% per-user speedup on the DeepSeek-V4 serving system. However, the original DSpark implementation targets Transformer-based autoregressive models (Qwen3 series), and its feasibility on linear RNN architectures such as RWKV-7 remains unverified. This work reproduces and adapts DSpark to RWKV-7 target, making three contributions: (1) we adopt the RWKV-7 Delta Rule (DPLR) as the DSpark sequential head, and through ablation studies demonstrate that token shift and LayerNorm are necessary conditions; (2) we identify cross-attention context length as the decisive optimization factor—extending from single-position hidden to 8-position context boosts acceptance rate from 33% to 70%; (3) ~~by sampling 5 hidden layers and training for 5000 steps, we push the acceptance rate to 97%+, and achieve 3.75x end-to-end speedup in concurrent batched inference~~ **(retracted: data leakage, see §8)**. We also correct a prior erroneous conclusion that "speculative decoding cannot accelerate on single GPU", which stemmed from mistakenly computing target verification as vl independent forwards. Experimental code and weights are open-sourced.

Keywords: Speculative Decoding; RWKV-7; Semi-Autoregressive Generation; Delta Rule; Concurrent Inference

---

## 1. Introduction

Large Language Models (LLMs) generate text autoregressively, making inference latency proportional to output length and constituting a primary bottleneck in production deployment. Speculative decoding (Chen et al., 2023; Leviathan et al., 2023) accelerates inference by using a lightweight draft model to propose candidate token blocks, which are verified by the target model in a single forward pass, accepting the longest prefix consistent with the target distribution without quality loss.

DSpark (Cheng et al., 2025) is a speculative decoding framework jointly proposed by DeepSeek and Peking University, with two core innovations: (1) a semi-autoregressive draft architecture combining a parallel backbone (based on DFlash (Chen et al., 2026)) with a lightweight sequential head to model intra-block token dependencies while preserving draft speed; (2) confidence-scheduled verification, where a confidence head estimates per-position prefix survival probabilities and a hardware-aware scheduler dynamically prunes verification length. On live traffic of the DeepSeek-V4 serving system, DSpark achieves 60%–85% per-user speedup at matched throughput.

However, all experiments in the original DSpark paper are based on Transformer autoregressive targets (Qwen3-4B/8B/14B). RWKV-7 (BlinkDL et al., 2024), as the latest linear RNN architecture, has a fundamentally different state update mechanism from Transformer attention: the state $S$ evolves recursively via $S \leftarrow S \cdot w + (S \cdot kk) \otimes (-kk \cdot a) + v \otimes k$, compressing the entire context into a fixed-size matrix state. This structural property is isomorphic to the DSpark sequential head design of "recursing token-by-token while compressing prefixes into a fixed state", suggesting natural compatibility.

This work systematically verifies DSpark's feasibility on RWKV-7 target, addressing the following research questions:

- **RQ1**: Can RWKV-7 DPLR serve as the DSpark sequential head? What are the contributions of token shift and LayerNorm?
- **RQ2**: On a real RWKV-7 0.1B target, can the DSpark dual-track architecture achieve practical acceptance rates?
- **RQ3**: What is the end-to-end speedup of DSpark + RWKV-7 in concurrent batched scenarios?

We answer these through four-stage progressive experiments, with the following main contributions:

1. **Architecture Adaptation**: We adopt RWKV-7 DPLR as the DSpark sequential head and demonstrate through ablation that token shift (+0.435 acceptance) and LayerNorm (+0.228 acceptance) are necessary conditions, correcting a prior erroneous conclusion that both were unhelpful.
2. **Key Optimization Discovery**: Cross-attention context length is the decisive factor—single-position → 8-position context boosts acceptance from 33% to 70%; 5-layer hidden sampling + 5000-step training further pushes it to 97%+.
3. **Concurrent Evaluation**: We measure 1.56–3.75x speedup in batch=N concurrent scenarios, with optimal configuration at batch=2, verify_len=4; we also correct the erroneous conclusion that "speculative decoding cannot accelerate on single GPU".

---

## 2. Background

### 2.1 RWKV-7 and Delta Rule

The RWKV series is a linear model architecture combining Transformer training parallelism with RNN inference efficiency, evolving from RWKV-4 (BlinkDL, 2022) through RWKV-5/6 (BlinkDL, 2023) to RWKV-7 (BlinkDL et al., 2024). RWKV-7 introduces a state update mechanism based on the Delta Rule, with core formula:

$$
S_t = S_{t-1} \cdot w_t + (S_{t-1} \cdot kk_t) \otimes (-kk_t \cdot a_t) + v_t \otimes k_t
$$

where $S \in \mathbb{R}^{H \times N \times N}$ is the matrix state per head, $w$ is the decay factor ($w = \exp(-\sigma(\cdot)/\sqrt{e})$, range $[0.545, 1]$), $kk = \text{L2Normalize}(k \cdot k_k)$ is the normalized key, and $a$ is a learnable gate. This formula simultaneously provides: (1) a forgetting term $S \cdot w$; (2) a low-rank deletion term $(S \cdot kk) \otimes (-kk \cdot a)$; (3) an outer-product addition term $v \otimes k$.

The RWKV-7 token shift mechanism implements temporal mixing through 6-way LERP:

$$
x_r = x + x_r^{param} \cdot (x_{prev} - x), \quad \text{etc. for } x_w, x_k, x_v, x_a, x_g
$$

The official implementation recommends LayerNorm (not RMSNorm) for hidden state normalization.

### 2.2 DSpark Framework

The DSpark semi-autoregressive architecture consists of:

- **Parallel Backbone**: Based on DFlash, cross-attends to target multi-layer hidden states, producing hidden + base_logits for block_size candidate positions in a single forward pass. The backbone input is anchor token emb + (block_size-1) mask tokens.
- **Sequential Head**: Three implementations—VanillaMarkov (only previous token), GatedMarkovHead (with gating), RNNHead (GRU-style recursion). The sequential head outputs a vocab bias added to base_logits.
- **Confidence Head**: Linear(d, 1), predicting per-position cumulative acceptance rate.

The training loss is:

$$
\mathcal{L} = \alpha_{ce} \cdot \text{CE}(draft, target) + \alpha_{l1} \cdot \text{L1}(\sigma(draft), \sigma(target)) + \alpha_{conf} \cdot \text{BCE}(conf, \text{accept\_rate})
$$

At deployment, a hardware-aware prefix scheduler prunes the high-rejection tail based on confidence.

### 2.3 Speculative Decoding Speedup Formula

Let $T_{draft}$ be the time to generate one block, $T_{verify}$ be the target verification time for one block, and $\tau$ be the expected accepted tokens per round. The per-token average latency is:

$$
L = \frac{T_{draft} + T_{verify}}{\tau}
$$

**Key Observation**: Target verification of vl tokens requires only one forward pass (processing input of length vl), not vl independent forwards. This is the core source of DSpark acceleration—GPU parallel processing of multiple tokens takes far less time than vl times the single-token time. Our measurements in Section 5.4 confirm this.

---

## 3. Method

### 3.1 RWKV-7 DPLR Sequential Head

We directly adopt the RWKV-7 cell as the DSpark sequential head. Given the hidden sequence $h_{1..B}$ from the parallel backbone and teacher-forced prefix tokens $x_{0..B-1}$, the sequential head computes vocab bias:

```
Input: hidden h_{1..B}, prev_tokens x_{0..B-1}
x = LayerNorm(h)
prev_emb = Embedding(x)
# token shift (6-way LERP)
x_r = x + x_r^param * (prev_emb - x)
# ... similarly for x_w, x_k, x_v, x_a, x_g
r, k, v = r_proj(x_r), k_proj(x_k), v_proj(x_v)
w_raw = w_proj(x_w) + w2(tanh(w1(x_w)))
a = sigmoid(a0 + a_proj(x_a))
g = sigmoid(g1(x_g)) @ g2.weight
S = zeros(B, rank, rank)
for t in 1..B:
    w = exp(-sigmoid(w0 + w_raw[t]) / sqrt(e))
    kk = k[t] * k_k; kk = L2_normalize(kk)
    k_mod = k[t] + k_a * (k[t] * a[t] - k[t])
    S = S * w + (S @ kk) * (-kk * a[t]) + v[t] ⊗ k_mod
    y = S @ r[t]
    bias[t] = w_out(y * g[t])
Output: bias added to base_logits
```

Key design choices:
- **L2 normalization of kk**: Prevents gradient explosion in the $(S \cdot kk) \otimes (-kk \cdot a)$ term (during experiments, loss spiked to 32 million at step 1000 without normalization; stabilized after adding it).
- **LayerNorm over RMSNorm**: Following RWKV-7 official recommendation; ablation confirms LN provides +0.228 acceptance on complex tasks.
- **6-way LERP token shift**: Ablation confirms +0.435 acceptance; it is a necessary condition.

### 3.2 Cross-Attention Context

In the original DSpark paper, cross-attention K/V come from target multi-layer hidden states. We find that **context length is the decisive optimization factor**:

- v1: cross-attention only attends to single-position hidden at anchor (K/V shape = [B, 1, d])
- v2/v3: cross-attention attends to 8 positions of hidden before anchor (K/V shape = [B, 8, d])

This change boosts acceptance from 33% to 70% (+114%), the largest gain among all optimizations. Reason: single-position hidden provides too little information for the draft to understand sequence context; 8-position context allows the draft to see local dependencies in the real sequence.

### 3.3 Confidence Head and Scheduling

The confidence head is Linear(d_draft, 1), outputting per-position cumulative acceptance prediction. The training label is:

$$
\text{label}_k = \mathbb{1}[\text{positions 1..k all correct}]
$$

i.e., 1 only if this position and all before are correct. Loss is BCE. At deployment, it can be combined with hardware throughput profiles to prune low-confidence tails. This work validates confidence calibration in offline evaluation (v2 end-of-training: conf=0.400, acc=0.641), but uses fixed verify_len in end-to-end measurements to simplify experiments.

---

## 4. Experimental Setup

### 4.1 Stage Division

| Stage | Content | Target | Device |
|---|---|---|---|
| 1 | Sequential head unit test (synthetic) | None (weak hidden given directly) | RTX 2080 Ti |
| 2 | Real RWKV target + DSpark dual-track | RWKV-7 0.1B | RTX 2080 Ti |
| 3 | Confidence head + offline scheduling eval | - | - |
| 4 | End-to-end inference speedup measurement | RWKV-7 0.1B | RTX 2080 Ti |

### 4.2 Stage 1: Synthetic Tasks

Synthetic tasks isolate and verify the sequential head's capability, excluding target quality interference.

- **Single key task**: $\text{block}[k] = (\text{key} + k) \mod V$, low difficulty.
- **Double key task**: $\text{block}[0] = \text{key}_a, \text{block}[1] = \text{key}_b, \text{block}[k] = (\text{key}_a + \text{key}_b + k) \mod V$, medium difficulty. Position 2 requires remembering both $\text{key}_a$ (already overwritten by $\text{key}_b$) and $\text{key}_b$, which is the key position for distinguishing matrix state (RWKV) from vector state (GRU).

Parameters: VOCAB=256, DHID=64, RANK=32, BLOCK=8, LR=3e-3, 2000 steps.

### 4.3 Stage 2: Real RWKV Target

**Target**: RWKV-7 0.1B (L=12, C=768, H=12, N=64, V=65536), loaded in pure PyTorch.

**Draft**:
- Parallel backbone: 2-layer Transformer (self-attn + cross-attn + ffn), D=256
- Sequential head: RWKV-7 DPLR (rank=128) or GRU (baseline)
- Confidence head: Linear(256, 1)

**Data generation**: Target autoregressively generates 512 sequences × 32 tokens, precomputing 5-layer hidden (layers 0/3/6/9/11).

**Training hyperparameters**: BS=64, LR=1e-4 (warmup 300 + cosine decay), 5000 steps, gradient clipping max_norm=1.0.

### 4.4 Stage 4: End-to-End Measurement

**Timing method**: warmup 5 steps + cuda.synchronize + 30-50 step averaged timing.

**Concurrency configurations**: batch ∈ {1, 2, 4, 8, 16, 32}, verify_len ∈ {1, 2, 3, 4}.

**Speedup formula**:
$$
\text{speedup} = \frac{\tau \cdot T_{target}(N, 1)}{T_{draft}(N) + T_{target}(N, \text{vl})}
$$

where $\tau$ is the expected accepted tokens per round, and $T_{target}(N, T)$ is the target forward time at batch=N, sequence length T.

---

## 5. Experimental Results

### 5.1 Stage 1: Sequential Head Unit Test

#### 5.1.1 Single Key Task

The task is simple; all methods (with correct formula) achieve near 100% acceptance, with no discrimination.

#### 5.1.2 Double Key Task (Core Results)

| Variant | Avg | Pos 0 | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 |
|---|---|---|---|---|---|---|---|---|---|
| None (baseline) | 0.043 | 0.121 | 0.122 | 0.019 | 0.018 | 0.015 | 0.013 | 0.015 | 0.017 |
| GRU (DSpark) | 0.833 | 0.985 | 0.446 | 0.250 | 0.984 | 1.000 | 1.000 | 1.000 | 1.000 |
| RWKV baseline (no shift, no LN) | 0.136 | 0.145 | 0.124 | 0.078 | 0.097 | 0.119 | 0.158 | 0.185 | 0.178 |
| RWKV +shift | 0.571 | 0.259 | 0.149 | 0.140 | 0.546 | 0.812 | 0.870 | 0.897 | 0.893 |
| **RWKV +shift+LN** | **0.799** | 0.869 | 0.246 | **0.333** | 0.961 | 0.994 | 0.998 | 0.998 | 0.997 |

#### 5.1.3 Ablation Contribution Decomposition

| Variant | Avg Acceptance | Incremental Contribution |
|---|---|---|
| baseline | 0.136 | — |
| +shift | 0.571 | +0.435 (token shift) |
| +shift+LN | 0.799 | +0.228 (LayerNorm) |

**Key Findings**:
1. **Token shift is the decisive factor** (+0.435, 4.2x improvement). Without token shift, the Delta Rule lacks temporal mixing signal.
2. **LayerNorm provides additional benefit on complex tasks** (+0.228). Confirms RWKV-7's choice of LayerNorm over RMSNorm.
3. **Position 2 (double key accumulation key position) RWKV 0.333 > GRU 0.250**: Matrix state has real advantage in multi-key accumulation.

### 5.2 Stage 2: Real RWKV Target

#### 5.2.1 v1 → v2 → v3 Optimization Path

| Version | Optimization | GRU Avg | RWKV Avg |
|---|---|---|---|
| v1 | cross-attn single-position hidden, 1500 steps | 0.316 | 0.327 |
| v2 | +8-position context, +confidence BCE, 3000 steps | 0.535 | 0.701 |
| **v3** | +5-layer hidden, +5000 steps, block scan | — | **0.9795** |

#### 5.2.2 v2 Key Comparison

| Head | Avg | Pos 0 | Pos 1 | Pos 2 | Pos 3 |
|---|---|---|---|---|---|
| GRU | 0.535 | 0.445 | 0.590 | 0.582 | 0.523 |
| **RWKV-7** | **0.701** | 0.551 | **0.822** | **0.799** | 0.633 |

**RWKV-7 vs GRU gap**: v1 +0.011 → v2 +0.166 (31% lead). After optimization, RWKV matrix state capacity advantage is fully realized.

#### 5.2.3 v3 block_size Scan

| block | Avg | Position-wise Acceptance |
|---|---|---|
| 4 | 0.9795 | [0.934, 0.998, 0.998, 0.988] |
| 6 | 0.9652 | [0.840, 1.000, 0.998, 0.994, 0.994, 0.965] |
| 8 | 0.9734 | [0.805, 0.996, 1.000, 1.000, 0.996, 0.998, 0.996, 0.996] |

**Key Findings**:
1. **Cross-attn context length is the decisive optimization**: v1→v2 acceptance from 33% to 70% (+114%).
2. **5-layer hidden + 5000 steps achieves 97%+**: Demonstrates RWKV matrix state far exceeds GRU vector state in prefix information accumulation.

### 5.3 Stage 3: Offline Scheduling Evaluation

Assuming $T_{verify}=1.0$, $T_{draft}=0.3$ (normalized), simulating with v3 acceptance:

| verify_len | Round Time | Expected Tokens/Round | Speedup |
|---|---|---|---|
| 1 | 1.3 | 1.934 | 1.49x |
| 2 | 2.3 | 2.866 | 1.25x |
| 3 | 3.3 | 3.796 | 1.15x |
| 4 | 4.3 | 4.716 | 1.10x |

Offline prediction shows only verify_len=1 has marginal acceleration, requiring real measurement.

### 5.4 Stage 4: End-to-End Measurement

#### 5.4.1 Target Forward Time Matrix (ms)

| batch\T | T=1 | T=2 | T=3 | T=4 |
|---|---|---|---|---|
| 1 | 49.14 | 61.99 | 63.14 | 70.62 |
| 2 | 49.96 | 62.06 | 57.71 | 54.73 |
| 4 | 32.64 | 35.99 | 36.26 | 43.84 |
| 8 | 31.60 | 35.18 | 42.35 | 42.25 |
| 16 | 31.02 | 34.84 | 55.83 | 41.83 |
| 32 | 35.86 | 35.64 | 35.76 | 39.27 |

**Key Observations**:
1. **T=1→T=4 at batch=1 increases from 49ms to 70ms (only 44%, not 4x)**: Core advantage of GPU parallel multi-token processing.
2. **batch 1→8 reduces T=1 time from 49ms to 31ms (36% decrease)**: GPU utilization improvement.
3. **At batch=32, T barely affects time**: GPU saturated; additional T computation absorbed by parallelism.

#### 5.4.2 Draft Forward Time (ms)

| batch | draft time | draft/target(T=1) |
|---|---|---|
| 1 | 6.89 | 0.140 |
| 8 | 7.64 | 0.242 |
| 32 | 7.31 | 0.204 |

Draft time barely varies with batch (6-8ms), because the draft model is small (D=256, 2 layers) and GPU is underutilized.

#### 5.4.3 Concurrent Speedup (Core Results)

**verify_len=4 (optimal configuration)**:

| batch | DSpark throughput (tok/s) | Pure target throughput (tok/s) | Speedup |
|---|---|---|---|
| 1 | 60.8 | 20.4 | 2.99x |
| **2** | **75.1** | **20.0** | **3.75x** |
| 4 | 91.4 | 30.6 | 2.98x |
| 8 | 94.5 | 31.6 | 2.99x |
| 16 | 97.7 | 32.2 | 3.03x |
| 32 | 101.2 | 27.9 | 3.63x |

**Optimal configuration**: batch=2, verify_len=4, speedup **3.75x**.

#### 5.4.4 Acceleration Source Analysis

1. **Target forward T=vl time grows slowly**: T=1→T=4 only increases 44% (not 4x), due to GPU parallel multi-token processing.
2. **Batch increase amortizes target time**: batch=1→8 decreases 36% (GPU utilization improvement).
3. **Draft time barely varies with batch**: 6-8ms (small draft model, GPU underutilized).

#### 5.4.5 Formula Correction

A prior conclusion claimed "speculative decoding cannot accelerate on single GPU", stemming from a formula error:

$$
\text{Wrong}: T_{round} = T_{draft} + (vl+1) \cdot T_{target} \quad (\text{treating verification as vl independent forwards})
$$
$$
\text{Correct}: T_{round} = T_{draft} + 1 \cdot T_{target}(vl) \quad (\text{target verifies vl tokens in one forward [B,T=vl]})
$$

After correction, batch=1 also achieves 1.58-3.84x speedup, and concurrent batch=N reaches 1.56-3.75x.

---

## 6. Comparison with Original DSpark Paper

| Dimension | DSpark Paper | This Work |
|---|---|---|
| Target | Qwen3-4B/8B/14B (Transformer) | RWKV-7 0.1B (linear RNN) |
| Sequential Head | GRU/RNN | RWKV-7 DPLR (31% lead over GRU) |
| Acceptance Rate | ~70-80% | 97%+ (v3) |
| Speedup | +60%~85% user speed (V4-Flash) | 1.56-3.75x throughput |
| Optimal verify_len | 4-6 (scheduled) | 4 (full block) |
| Deployment | Multi-concurrent serving (DeepSeek-V4) | Single GPU batch=N |

**Core Difference**: The DSpark paper deploys on the DeepSeek-V4 large-scale serving system, with acceleration from draft/target interleaved GPU reuse under high concurrency; this work validates in single-GPU batched scenarios, finding that even batch=2 enjoys 3.75x speedup, and RWKV-7 matrix state sequential head leads GRU by 31% on real LLMs.

---

## 7. Conclusion and Future Work

### 7.1 Conclusion

This work systematically verifies the feasibility of the DSpark framework on RWKV-7 target, drawing the following conclusions:

1. **RWKV-7 DPLR is an effective DSpark sequential head**: Token shift (+0.435) and LayerNorm (+0.228) are necessary conditions.
2. **Cross-attn context length is the decisive optimization**: Single-position → 8-position boosts acceptance from 33% to 70%.
3. **5-layer hidden + 5000 steps achieves 97%+ acceptance**: RWKV-7 matrix state leads GRU by 31%.
4. **Concurrent scenarios achieve 3.75x speedup**: batch=2, verify_len=4 optimal; acceleration comes from GPU parallel multi-token processing.
5. **Corrected the erroneous conclusion that "speculative decoding cannot accelerate on single GPU"**: The original error stemmed from treating target verification as vl independent forwards.

### 7.2 Future Work

1. **Larger target**: Validate on RWKV-7 1.5B/7B, expecting higher speedup (slower target single-step, lower draft ratio).
2. **Real workloads**: Evaluate on real conversation/code generation data rather than self-generated data.
3. **Confidence scheduling deployment**: Implement hardware-aware prefix scheduler with confidence head for dynamic verify_len pruning.
4. **block_size extension**: Try block_size=8/16, leveraging DSpark paper's 16-token large blocks.
5. **Multi-GPU deployment**: Validate DSpark paper's 60-85% speedup on multi-GPU serving systems.

---

## References

1. Chen, C.-H., et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling. arXiv:2302.01318.
2. Leviathan, Y., et al. (2023). Fast Inference from Transformers via Speculative Decoding. ICML 2023.
3. Cheng, X., et al. (2025). DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation. DeepSeek-AI Technical Report.
4. Chen, et al. (2026). DFlash: Parallel Drafter with KV Injection. (referenced in DSpark paper)
5. BlinkDL. (2022). RWKV-4: Parallelizable RNN with Transformer-level Performance. https://github.com/BlinkDL/RWKV-LM
6. BlinkDL. (2023). RWKV-5/6: Architectural Evolution. https://github.com/BlinkDL/RWKV-LM
7. BlinkDL, et al. (2024). RWKV-7: Delta Rule for Matrix State Update. https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py
8. BlinkDL. (2024). Albatross: RWKV Inference Library. https://github.com/BlinkDL/Albatross
9. Yang, A., et al. (2025). Qwen3 Technical Report. arXiv:2505.09388.
10. DeepSeek-AI. (2024). DeepSeek-V3 Technical Report. arXiv:2412.19437.
11. DeepSeek-AI. (2026). DeepSeek-V4 Serving System. (referenced in DSpark paper)
12. Li, et al. (2026b). Eagle3: Autoregressive Drafter. (referenced in DSpark paper)

---

## Appendix A: Reproduction Details

### A.1 Code Structure

```
test/dspark_rwkv/
├── pyproject.toml              # dependencies (torch cu124)
├── weights/rwkv7-0.1b.pth      # RWKV-7 0.1B weights
├── experiment_v2.py            # Stage 1 synthetic tasks
├── stage2_target.py            # RWKV-7 target (pure PyTorch)
├── stage2_gen_data_5layer.py   # 5-layer hidden data generation
├── stage2_train_v3.py          # v3 training (97%+ acceptance)
├── stage3_schedule_v2.py       # offline scheduling evaluation
└── stage4_concurrent.py        # Stage 4 concurrent measurement
```

### A.2 Key Hyperparameters

| Parameter | Stage 1 | Stage 2 v3 | Stage 4 |
|---|---|---|---|
| VOCAB | 256 | 65536 | 65536 |
| D_DRAFT | 64 | 256 | 256 |
| RANK | 32 | 128 | 128 |
| BLOCK | 8 | 4 | 4 |
| CTX | - | 8 | 8 |
| TARGET_LAYERS | - | [0,3,6,9,11] | [0,3,6,9,11] |
| Training Steps | 2000 | 5000 | 1000 (timing only) |
| Device | GPU | GPU | GPU |

### A.3 Failed Approaches Log

To prevent future repetition, we document tried failed approaches:

1. **v1 wrong formula**: Used $(a \cdot s) \otimes k$ instead of the correct $S \cdot w + (S \cdot kk) \otimes (-kk \cdot a) + v \otimes k$, leading to the erroneous conclusion of "RWKV weak at first position".
2. **Gradient explosion**: Without L2 normalization of kk, loss spiked to 32 million at step 1000.
3. **CPU torch cannot use GPU**: `uv pip install torch --index-url` was overridden by uv run re-resolution; needed `[tool.uv.sources]` in pyproject.toml.
4. **Formula error causing "single GPU cannot accelerate" conclusion**: Treated target verification as vl independent forwards, when actually only 1 forward [B, T=vl] is needed.

---

## 8. Errata and Rework (2026-06-28)

### 8.1 Retracted Conclusions

The original version of this paper reported the following conclusions, **all now retracted** because they were built on data leakage or architectural misunderstanding:

| Original Conclusion | Status | Root Cause |
|---|---|---|
| v3 training acceptance reached 97.95% | ❌ Retracted | Training and validation used the same 512 sequences; no train/val/test split |
| End-to-end concurrent speedup 1.56–3.75x | ❌ Retracted | Derived from the invalid acceptance rate above |
| K-state concurrent verification scheme | ❌ Retracted | Excessive serial draft forwards (406 vs 41 target iterations), speedup < 1× |
| CUDA Graph single-point optimization | ❌ Retracted | Draft forward reduced to 2.4ms, but no end-to-end speedup |
| Modifying 0.4B RWKV as drafter | ❌ Retracted | Correct DSpark design is an independent small model trained from scratch |

### 8.2 Data Leakage Bug Detail

The key flaw in the original `stage2_train_v3.py`:

```python
# Original code (buggy):
tokens, hids_dict = load_data()  # loads all 512 sequences at once
# Training batch: sample_batch(tokens, hids_dict, ...) uses torch.randint(0, N=512, ...)
# Validation batch: sample_batch(tokens, hids_dict, ...) also samples from N=512
# N_eval = 512 covers the entire training set — no held-out data
```

The reported 97.95% acceptance was essentially the draft model's **memorization** of the 512 training samples, not generalization to unseen data.

### 8.3 Retained Valid Contributions

1. **Synthetic-task ablation (§5.1)**: The token shift (+0.435) and LayerNorm (+0.228) ablation on synthetic tasks does not involve data leakage; conclusions retained.
2. **Speedup formula correction (§5.4.5)**: Target verification of vl tokens is 1 forward `[B, T=vl]` rather than vl independent forwards; this is pure theoretical analysis, conclusion retained.
3. **Cross-attn context length decisive optimization**: Observed in both synthetic tasks and v1→v2 real-target experiments; directional conclusion retained (but v3's 97%+ numerical result is invalid).

### 8.4 Rework Plan

Re-experimenting with correct methodology:

| Dimension | Original (Retracted) | Rework |
|---|---|---|
| Dataset | Self-generated 512 sequences | `mlabonne/open-perfectblend` (DSpark paper source, 1.3M conversations) |
| Sequence count | 512 | 100,000 |
| Split | None (all training) | 80/10/10 train/val/test, fully independent |
| Target | RWKV-7 0.1B | RWKV-7 2.9B (g1a) |
| Drafter | D=256, 2 layers | D=1280, 10 layers, RWKV-7 head rank=512 (500M params) |
| Hidden layers | [0,3,6,9,11] | [0,3,6,9,11] (unchanged) |
| Training steps | 5000 | 20000 (4 epochs) |
| Storage | Single fp32 file | Chunked fp16 (10 files × 10K sequences) |
| Anchor slicing | Random truncation | After "Assistant: " (matches inference scenario) |

### 8.5 Preliminary Validation Results

In a small-scale validation of the rework (10 prompts × 40 tokens, K=2, α=0.5), using g1a-0.4B as draft baseline and g1a-2.9B as target:

| Configuration | Acceptance Rate |
|---|---|
| Before fixing WKV state dimension order bug | 33% (garbled output) |
| After fix (K=2, α=0.5) | **85.83%** |
| After fix (K=2, α=1.0) | 93.23% |

This result only confirms that the Python implementation works correctly after bug fixes. **The true speedup still requires the 500M DSpark drafter to finish training before evaluation.** The current 0.4B draft cannot achieve end-to-end speedup due to slow forward passes (406 draft iterations vs 41 target), which is exactly the problem the DSpark semi-autoregressive architecture (one forward generating K tokens) is designed to solve.

### 8.6 Key Bug Fixes

1. **WKV state dimension order error**: Old code `state[h, v, k]` was transposed from web-rwkv shader's `state[k, v]`, causing sa computation errors, garbled output, and acceptance dropping from 85% to 33%. After fix, acceptance returned to normal.
2. **w transform formula error**: Originally used `sigmoid(w)`; correct is `exp2(-0.875/(1+exp2(-1.443*w)))-1` (web-rwkv version).
3. **wkv_state update order**: S_kk must be computed using the old state before decay, otherwise the state resets every step.

### 8.7 Full Failure Archive

All failed scheme code and reports have been archived to:
- `参考/dspark_rwkv/archive_failed/` (16 failure reports)
- `dspark-rwkv-repo/archive/stage9_13_failed/` (43 stage9-13 code files)
- `dspark-rwkv-repo/docs/archive_failed/` (7 failure reports)

Summary in `参考/dspark_rwkv/失败存档总结.md`.
