# DSpark-RWKV：将半自回归推测解码适配于 RWKV-7

> **⚠️ 勘误声明（2026-06-28）**：本文原报告的 97%+ 接受率与 3.75x 加速比已撤回。详见 §8「勘误与重做」。保留的有效贡献为：合成任务消融实验（token shift +0.435、LayerNorm +0.228）、加速比公式修正、cross-attn context 长度决定性优化发现。本文档保留原始实验描述用于历史记录，但所有「97%+ 接受率」「3.75x 加速比」相关结论均不可信。

**摘要**

推测解码通过解耦草稿生成与目标验证来加速大语言模型推理。近期提出的 DSpark 框架采用半自回归草稿架构（并行主干 + 顺序头）与置信度调度验证，在 DeepSeek-V4 服务系统上实现 60%–85% 的用户速度提升。然而，DSpark 原始实现面向 Transformer 类自回归 target（Qwen3 系列），其在以 RWKV-7 为代表的线性 RNN 架构上的可行性尚未被验证。本工作复现并适配 DSpark 至 RWKV-7 target，提出三项贡献：（1）将 RWKV-7 Delta Rule（DPLR）作为 DSpark 顺序头，通过 token shift 与 LayerNorm 的消融实验证明二者是必要条件；（2）发现 cross-attention 的 context 长度是决定性优化因素，单位置 hidden 到 8 位置 context 使接受率从 33% 提升至 70%；（3）~~通过 5 层 hidden 采样与 5000 步训练将接受率推至 97%+，并在并发批处理场景下实测达 3.75x 端到端加速比~~ **（已撤回：数据泄露，见 §8）**。我们同时修正了此前"单 GPU 推测解码无法加速"的错误结论，该错误源于将 target 验证误算为 vl 次独立 forward。实验代码与权重已开源。

关键词：推测解码；RWKV-7；半自回归生成；Delta Rule；并发推理

---

## 1. 引言

大语言模型（LLM）的自回归生成特性使推理延迟与输出长度成正比，构成生产部署的主要瓶颈。推测解码（Chen et al., 2023; Leviathan et al., 2023）通过轻量级 draft 模型提出候选 token 块，由 target 模型单次前向验证，接受与自身分布一致的最长前缀，从而在不损失质量的前提下加速推理。

DSpark（Cheng et al., 2025）是 DeepSeek 与北京大学联合提出的推测解码框架，其核心包含两点创新：（1）半自回归草稿架构，将并行主干（基于 DFlash（Chen et al., 2026））与轻量顺序头结合，在保持 draft 速度的同时建模 block 内 token 依赖；（2）置信度调度验证，通过置信度头估计每位置前缀存活概率，并由硬件感知调度器动态裁剪验证长度。在 DeepSeek-V4 服务系统的真实流量上，DSpark 在等吞吐下使用户速度提升 60%–85%。

然而，DSpark 原始论文的实验全部基于 Transformer 自回归 target（Qwen3-4B/8B/14B）。RWKV-7（BlinkDL et al., 2024）作为最新的线性 RNN 架构，其 Delta Rule 状态更新机制与 Transformer 注意力有本质区别：状态 $S$ 通过 $S \leftarrow S \cdot w + (S \cdot kk) \otimes (-kk \cdot a) + v \otimes k$ 递归演化，将整个上下文压缩进固定大小的矩阵状态。这种结构特性与 DSpark 顺序头"逐 token 递归、将前缀压缩进固定状态"的设计同构，理论上具备天然适配性。

本工作系统验证 DSpark 在 RWKV-7 target 上的可行性，回答以下研究问题：

- **RQ1**：RWKV-7 DPLR 能否作为 DSpark 顺序头？token shift 与 LayerNorm 的贡献如何？
- **RQ2**：在真实 RWKV-7 0.1B target 上，DSpark 双轨架构的接受率能否达到可用水平？
- **RQ3**：在并发批处理场景下，DSpark + RWKV-7 的端到端加速比如何？

我们通过四阶段渐进实验回答这些问题，主要贡献如下：

1. **架构适配**：将 RWKV-7 DPLR 作为 DSpark 顺序头，通过消融实验证明 token shift（+0.435 接受率）与 LayerNorm（+0.228 接受率）是必要条件，纠正了此前"两者无益"的错误结论。
2. **关键优化发现**：cross-attention 的 context 长度是决定性因素，单位置 → 8 位置 context 使接受率从 33% 升至 70%；5 层 hidden 采样 + 5000 步训练进一步推至 97%+。
3. **并发实测**：在 batch=N 并发场景下实测 1.56–3.75x 加速比，最优配置为 batch=2、verify_len=4；同时修正了"单 GPU 推测解码无法加速"的错误结论。

---

## 2. 背景

### 2.1 RWKV-7 与 Delta Rule

RWKV 系列是一个结合 Transformer 训练并行性与 RNN 推理效率的线性模型架构，从 RWKV-4（BlinkDL, 2022）演进至 RWKV-5/6（BlinkDL, 2023）和 RWKV-7（BlinkDL et al., 2024）。RWKV-7 引入了基于 Delta Rule 的状态更新机制，其核心公式为：

$$
S_t = S_{t-1} \cdot w_t + (S_{t-1} \cdot kk_t) \otimes (-kk_t \cdot a_t) + v_t \otimes k_t
$$

其中 $S \in \mathbb{R}^{H \times N \times N}$ 是每个 head 的矩阵状态，$w$ 是衰减因子（$w = \exp(-\sigma(\cdot)/\sqrt{e})$，范围 $[0.545, 1]$），$kk = \text{L2Normalize}(k \cdot k_k)$ 是归一化后的 key，$a$ 是可学习门控。该公式同时具备：（1）遗忘项 $S \cdot w$；（2）低秩删除项 $(S \cdot kk) \otimes (-kk \cdot a)$；（3）外积添加项 $v \otimes k$。

RWKV-7 的 token shift 机制通过 6 路 LERP 实现时间混合：

$$
x_r = x + x_r^{param} \cdot (x_{prev} - x), \quad \text{etc. for } x_w, x_k, x_v, x_a, x_g
$$

官方实现推荐 LayerNorm（非 RMSNorm）用于 hidden 状态归一化。

### 2.2 DSpark 框架

DSpark 的半自回归架构包含：

- **并行主干**：基于 DFlash，cross-attention 到 target 多层 hidden states，一次前向产出 block_size 个候选位置的 hidden + base_logits。主干输入为 anchor token emb + (block_size-1) 个 mask token。
- **顺序头**：三种实现可选——VanillaMarkov（仅前一个 token）、GatedMarkovHead（带门控）、RNNHead（GRU 式递归）。顺序头输出 vocab bias 加到 base_logits 上。
- **置信度头**：Linear(d, 1)，预测每位置累积接受率。

训练损失为：

$$
\mathcal{L} = \alpha_{ce} \cdot \text{CE}(draft, target) + \alpha_{l1} \cdot \text{L1}(\sigma(draft), \sigma(target)) + \alpha_{conf} \cdot \text{BCE}(conf, \text{accept\_rate})
$$

部署期采用硬件感知前缀调度器，根据置信度截断高拒绝率尾部。

### 2.3 推测解码加速比公式

设 $T_{draft}$ 为 draft 生成一个 block 的时间，$T_{verify}$ 为 target 验证一个 block 的时间，$\tau$ 为每轮期望接受 token 数。每 token 平均延迟为：

$$
L = \frac{T_{draft} + T_{verify}}{\tau}
$$

**关键观察**：target 验证 vl 个 token 只需一次 forward（处理序列长度为 vl 的输入），而非 vl 次独立 forward。这是 DSpark 加速的核心来源——GPU 并行处理多 token 的时间远小于 vl 倍单 token 时间。本文第 5.4 节的实测将证实这一点。

---

## 3. 方法

### 3.1 RWKV-7 DPLR 顺序头

我们将 RWKV-7 cell 直接作为 DSpark 顺序头。给定并行主干输出的 hidden 序列 $h_{1..B}$ 和 teacher forcing 的前缀 token $x_{0..B-1}$，顺序头计算 vocab bias：

```
输入：hidden h_{1..B}, prev_tokens x_{0..B-1}
x = LayerNorm(h)
prev_emb = Embedding(x)
# token shift (6 路 LERP)
x_r = x + x_r^param * (prev_emb - x)
# ... 同理 x_w, x_k, x_v, x_a, x_g
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
输出：bias 加到 base_logits
```

关键设计选择：
- **L2 归一化 kk**：防止 $(S \cdot kk) \otimes (-kk \cdot a)$ 项梯度爆炸（实验中 step 1000 时 loss 曾飙至 3200 万，加入归一化后稳定）。
- **LayerNorm 而非 RMSNorm**：遵循 RWKV-7 官方推荐，消融实验证实 LN 在复杂任务上 +0.228 接受率。
- **token shift 6 路 LERP**：消融实验证实 +0.435 接受率，是必要条件。

### 3.2 Cross-Attention Context

DSpark 原论文中 cross-attn 的 K/V 来自 target 多层 hidden states。我们发现 **context 长度是决定性优化因素**：

- v1：cross-attn 只看 anchor 单位置 hidden（K/V shape = [B, 1, d]）
- v2/v3：cross-attn 看 anchor 之前 8 位置 hidden（K/V shape = [B, 8, d]）

这一改动使接受率从 33% 跃升至 70%（+114%），是所有优化中收益最大的一项。原因：单位置 hidden 信息量太少，draft 无法理解序列上下文；8 位置 context 让 draft 能看到真实序列的局部依赖。

### 3.3 置信度头与调度

置信度头为 Linear(d_draft, 1)，输出每位置的累积接受率预测。训练标签为：

$$
\text{label}_k = \mathbb{1}[\text{位置 1..k 全部正确}]
$$

即该位置及之前全对才为 1。损失为 BCE。部署期可结合硬件吞吐 profile 截断低置信尾部。本工作在离线评估中验证置信度校准（v2 训练末尾 conf=0.400, acc=0.641），但端到端实测采用固定 verify_len 以简化实验。

---

## 4. 实验设置

### 4.1 阶段划分

| 阶段 | 内容 | Target | 设备 |
|---|---|---|---|
| 1 | 顺序头单测（合成任务） | 无（weak hidden 直接给） | RTX 2080 Ti |
| 2 | 真实 RWKV target + DSpark 双轨 | RWKV-7 0.1B | RTX 2080 Ti |
| 3 | 置信度头 + 离线调度评估 | - | - |
| 4 | 端到端推理加速实测 | RWKV-7 0.1B | RTX 2080 Ti |

### 4.2 阶段1：合成任务

合成任务用于隔离验证顺序头本身的能力，排除 target 质量干扰。

- **单 key 任务**：$\text{block}[k] = (\text{key} + k) \mod V$，难度低。
- **双 key 任务**：$\text{block}[0] = \text{key}_a, \text{block}[1] = \text{key}_b, \text{block}[k] = (\text{key}_a + \text{key}_b + k) \mod V$，难度中。位置 2 需同时记住 $\text{key}_a$（已被 $\text{key}_b$ 覆盖）和 $\text{key}_b$，是区分矩阵状态（RWKV）与向量状态（GRU）的关键位置。

参数：VOCAB=256, DHID=64, RANK=32, BLOCK=8, LR=3e-3, 2000 步。

### 4.3 阶段2：真实 RWKV target

**Target**：RWKV-7 0.1B（L=12, C=768, H=12, N=64, V=65536），纯 PyTorch 加载权重。

**Draft**：
- 并行主干：2 层 Transformer（self-attn + cross-attn + ffn），D=256
- 顺序头：RWKV-7 DPLR（rank=128）或 GRU（对照）
- 置信度头：Linear(256, 1)

**数据生成**：target 自回归生成 512 序列 × 32 token，预计算 5 层 hidden（layer 0/3/6/9/11）。

**训练超参**：BS=64, LR=1e-4（warmup 300 + cosine decay），5000 步，梯度裁剪 max_norm=1.0。

### 4.4 阶段4：端到端实测

**测时方法**：warmup 5 步 + cuda.synchronize + 30-50 步计时平均。

**并发配置**：batch ∈ {1, 2, 4, 8, 16, 32}，verify_len ∈ {1, 2, 3, 4}。

**加速比公式**：
$$
\text{speedup} = \frac{\tau \cdot T_{target}(N, 1)}{T_{draft}(N) + T_{target}(N, \text{vl})}
$$

其中 $\tau$ 为每轮期望接受 token 数，$T_{target}(N, T)$ 为 target 在 batch=N、序列长 T 时的 forward 时间。

---

## 5. 实验结果

### 5.1 阶段1：顺序头单测

#### 5.1.1 单 key 任务

任务简单，所有方法（正确公式后）均接近 100% 接受率，无区分度。

#### 5.1.2 双 key 任务（核心结果）

| 变体 | 平均 | 位置 0 | 位置 1 | 位置 2 | 位置 3 | 位置 4 | 位置 5 | 位置 6 | 位置 7 |
|---|---|---|---|---|---|---|---|---|---|
| None (baseline) | 0.043 | 0.121 | 0.122 | 0.019 | 0.018 | 0.015 | 0.013 | 0.015 | 0.017 |
| GRU (DSpark) | 0.833 | 0.985 | 0.446 | 0.250 | 0.984 | 1.000 | 1.000 | 1.000 | 1.000 |
| RWKV baseline（无 shift 无 LN） | 0.136 | 0.145 | 0.124 | 0.078 | 0.097 | 0.119 | 0.158 | 0.185 | 0.178 |
| RWKV +shift | 0.571 | 0.259 | 0.149 | 0.140 | 0.546 | 0.812 | 0.870 | 0.897 | 0.893 |
| **RWKV +shift+LN** | **0.799** | 0.869 | 0.246 | **0.333** | 0.961 | 0.994 | 0.998 | 0.998 | 0.997 |

#### 5.1.3 消融贡献分解

| 变体 | 平均接受率 | 增量贡献 |
|---|---|---|
| baseline | 0.136 | — |
| +shift | 0.571 | +0.435（token shift） |
| +shift+LN | 0.799 | +0.228（LayerNorm） |

**关键发现**：
1. **token shift 是决定性因素**（+0.435，4.2 倍提升）。无 token shift 时 Delta Rule 缺少时间混合信号。
2. **LayerNorm 在复杂任务上有额外收益**（+0.228）。证实 RWKV-7 用 LayerNorm 而非 RMSNorm 是对的。
3. **位置 2（双 key 累积关键位）RWKV 0.333 > GRU 0.250**：矩阵状态在多 key 累积上有真实优势。

### 5.2 阶段2：真实 RWKV target

#### 5.2.1 v1 → v2 → v3 优化路径

| 版本 | 优化点 | GRU 平均 | RWKV 平均 |
|---|---|---|---|
| v1 | cross-attn 单位置 hidden, 1500 步 | 0.316 | 0.327 |
| v2 | +8 位置 context, +置信度 BCE, 3000 步 | 0.535 | 0.701 |
| **v3** | +5 层 hidden, +5000 步, block 扫描 | — | **0.9795** |

#### 5.2.2 v2 关键对比

| 顺序头 | 平均 | 位置 0 | 位置 1 | 位置 2 | 位置 3 |
|---|---|---|---|---|---|
| GRU | 0.535 | 0.445 | 0.590 | 0.582 | 0.523 |
| **RWKV-7** | **0.701** | 0.551 | **0.822** | **0.799** | 0.633 |

**RWKV-7 vs GRU 差距**：v1 +0.011 → v2 +0.166（领先 31%）。优化后 RWKV 矩阵状态容量优势充分体现。

#### 5.2.3 v3 block_size 扫描

| block | 平均 | 各位置接受率 |
|---|---|---|
| 4 | 0.9795 | [0.934, 0.998, 0.998, 0.988] |
| 6 | 0.9652 | [0.840, 1.000, 0.998, 0.994, 0.994, 0.965] |
| 8 | 0.9734 | [0.805, 0.996, 1.000, 1.000, 0.996, 0.998, 0.996, 0.996] |

**关键发现**：
1. **cross-attn context 长度是决定性优化**：v1→v2 接受率从 33% 升至 70%（+114%）。
2. **5 层 hidden 采样 + 5000 步达 97%+**：证明 RWKV 矩阵状态在累积前缀信息上远优于 GRU 向量状态。

### 5.3 阶段3：离线调度评估

假设 $T_{verify}=1.0$, $T_{draft}=0.3$（归一化），基于 v3 接受率模拟：

| verify_len | 每轮时间 | 每轮期望 token | 加速比 |
|---|---|---|---|
| 1 | 1.3 | 1.934 | 1.49x |
| 2 | 2.3 | 2.866 | 1.25x |
| 3 | 3.3 | 3.796 | 1.15x |
| 4 | 4.3 | 4.716 | 1.10x |

离线预测仅 verify_len=1 有微弱加速，需实测验证。

### 5.4 阶段4：端到端实测

#### 5.4.1 target forward 时间矩阵 (ms)

| batch\T | T=1 | T=2 | T=3 | T=4 |
|---|---|---|---|---|
| 1 | 49.14 | 61.99 | 63.14 | 70.62 |
| 2 | 49.96 | 62.06 | 57.71 | 54.73 |
| 4 | 32.64 | 35.99 | 36.26 | 43.84 |
| 8 | 31.60 | 35.18 | 42.35 | 42.25 |
| 16 | 31.02 | 34.84 | 55.83 | 41.83 |
| 32 | 35.86 | 35.64 | 35.76 | 39.27 |

**关键观察**：
1. **T=1→T=4 时 batch=1 时间从 49ms 升到 70ms（只增 44%，非 4 倍）**：GPU 并行处理多 token 的核心优势。
2. **batch 1→8 时 T=1 时间从 49ms 降到 31ms（下降 36%）**：GPU 利用率提升。
3. **batch=32 时 T 几乎不影响时间**：GPU 已饱和，T 增加的计算被并行吸收。

#### 5.4.2 draft forward 时间 (ms)

| batch | draft 时间 | draft/target(T=1) |
|---|---|---|
| 1 | 6.89 | 0.140 |
| 8 | 7.64 | 0.242 |
| 32 | 7.31 | 0.204 |

draft 时间几乎不随 batch 变化（6-8ms），因为 draft 模型小（D=256, 2 层），GPU 未饱和。

#### 5.4.3 并发加速比（核心结果）

**verify_len=4（最优配置）**：

| batch | DSpark 吞吐 (tok/s) | 纯 target 吞吐 (tok/s) | 加速比 |
|---|---|---|---|
| 1 | 60.8 | 20.4 | 2.99x |
| **2** | **75.1** | **20.0** | **3.75x** |
| 4 | 91.4 | 30.6 | 2.98x |
| 8 | 94.5 | 31.6 | 2.99x |
| 16 | 97.7 | 32.2 | 3.03x |
| 32 | 101.2 | 27.9 | 3.63x |

**最优配置**：batch=2, verify_len=4, 加速比 **3.75x**。

#### 5.4.4 加速来源分析

1. **target forward T=vl 时间增长缓慢**：T=1→T=4 只增 44%（非 4 倍），因为 GPU 并行处理多 token。
2. **batch 增加摊薄 target 时间**：batch=1→8 下降 36%（GPU 利用率提升）。
3. **draft 时间几乎不随 batch 变化**：6-8ms（draft 模型小，GPU 未饱和）。

#### 5.4.5 公式修正

此前有结论称"单 GPU 推测解码无法加速"，源于公式错误：

$$
\text{错误}: T_{round} = T_{draft} + (vl+1) \cdot T_{target} \quad (\text{把验证当 vl 次独立 forward})
$$
$$
\text{正确}: T_{round} = T_{draft} + 1 \cdot T_{target}(vl) \quad (\text{target 一次 forward [B,T=vl] 验证 vl 个 token})
$$

修正后 batch=1 也有 1.58-3.84x 加速，并发 batch=N 达 1.56-3.75x。

---

## 6. 与 DSpark 原论文对比

| 维度 | DSpark 论文 | 本工作 |
|---|---|---|
| Target | Qwen3-4B/8B/14B（Transformer） | RWKV-7 0.1B（线性 RNN） |
| 顺序头 | GRU/RNN | RWKV-7 DPLR（领先 GRU 31%） |
| 接受率 | ~70-80% | 97%+（v3） |
| 加速比 | +60%~85% 用户速度（V4-Flash） | 1.56-3.75x 吞吐 |
| 最优 verify_len | 4-6（调度） | 4（完整 block） |
| 部署环境 | 多并发服务（DeepSeek-V4） | 单 GPU batch=N |

**核心差异**：DSpark 原论文在 DeepSeek-V4 大规模服务系统上部署，加速来自高并发下 draft/target 交错复用 GPU；本工作在单 GPU 批处理场景下验证，发现即使 batch=2 也能享受 3.75x 加速，且 RWKV-7 矩阵状态顺序头在真实 LLM 上领先 GRU 31%。

---

## 7. 结论与未来工作

### 7.1 结论

本工作系统验证了 DSpark 框架在 RWKV-7 target 上的可行性，得出以下结论：

1. **RWKV-7 DPLR 是 DSpark 顺序头的有效实现**：token shift（+0.435）与 LayerNorm（+0.228）是必要条件。
2. **cross-attn context 长度是决定性优化**：单位置→8 位置使接受率从 33% 升至 70%。
3. **5 层 hidden + 5000 步训练达 97%+ 接受率**：RWKV-7 矩阵状态领先 GRU 31%。
4. **并发场景下达 3.75x 加速**：batch=2, verify_len=4 最优；加速来自 GPU 并行处理多 token 的特性。
5. **修正了"单 GPU 推测解码无法加速"的错误结论**：原错误源于把 target 验证误算为 vl 次独立 forward。

### 7.2 未来工作

1. **更大 target**：在 RWKV-7 1.5B/7B 上验证，预期加速比更高（target 单步更慢，draft 占比更低）。
2. **真实 workload**：在真实对话/代码生成数据上评估，而非自生成数据。
3. **置信度调度部署**：实现硬件感知前缀调度器，结合置信度头动态裁剪 verify_len。
4. **block_size 扩展**：试 block_size=8/16，结合 DSpark 论文的 16 token 大 block。
5. **多 GPU 部署**：在多 GPU 服务系统上验证 DSpark 原论文的 60-85% 加速。

---

## 参考文献

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

## 附录 A：复现细节

### A.1 代码结构

```
test/dspark_rwkv/
├── pyproject.toml              # 依赖（torch cu124）
├── weights/rwkv7-0.1b.pth      # RWKV-7 0.1B 权重
├── experiment_v2.py            # 阶段1 合成任务
├── stage2_target.py            # RWKV-7 target（纯 PyTorch）
├── stage2_gen_data_5layer.py   # 5 层 hidden 数据生成
├── stage2_train_v3.py          # v3 训练（97%+ 接受率）
├── stage3_schedule_v2.py       # 离线调度评估
└── stage4_concurrent.py        # 阶段4 并发实测
```

### A.2 关键超参

| 参数 | 阶段1 | 阶段2 v3 | 阶段4 |
|---|---|---|---|
| VOCAB | 256 | 65536 | 65536 |
| D_DRAFT | 64 | 256 | 256 |
| RANK | 32 | 128 | 128 |
| BLOCK | 8 | 4 | 4 |
| CTX | - | 8 | 8 |
| TARGET_LAYERS | - | [0,3,6,9,11] | [0,3,6,9,11] |
| 训练步数 | 2000 | 5000 | 1000（仅测时） |
| 设备 | GPU | GPU | GPU |

### A.3 失败方案记录

为避免后续重复，记录已尝试的失败方案：

1. **v1 错误公式**：用 $(a \cdot s) \otimes k$ 而非正确的 $S \cdot w + (S \cdot kk) \otimes (-kk \cdot a) + v \otimes k$，导致"RWKV 首位弱"的错误结论。
2. **梯度爆炸**：未对 kk 做 L2 归一化，step 1000 时 loss 飙至 3200 万。
3. **CPU torch 无法用 GPU**：`uv pip install torch --index-url` 被 uv run 重新解析覆盖，需在 pyproject.toml 配置 `[tool.uv.sources]`。
4. **公式错误导致"单 GPU 无法加速"结论**：把 target 验证误算为 vl 次独立 forward，实际只需 1 次 forward [B, T=vl]。

---

## 8. 勘误与重做（2026-06-28）

### 8.1 撤回的结论

本文原版本报告了下述结论，**现已全部撤回**，因为它们建立在数据泄露或架构理解错误之上：

| 原结论 | 状态 | 根因 |
|---|---|---|
| v3 训练接受率达 97.95% | ❌ 撤回 | 训练集与验证集使用同一份 512 条序列，无 train/val/test split |
| 端到端并发加速比 1.56–3.75x | ❌ 撤回 | 基于上述无效接受率推导 |
| K-state 并发验证方案 | ❌ 撤回 | 串行 draft forward 过多（406 vs 41 次 target），加速比 < 1× |
| CUDA Graph 单点优化 | ❌ 撤回 | draft forward 降到 2.4ms，但端到端无加速 |
| 改造 0.4B RWKV 作 drafter | ❌ 撤回 | DSpark 正确做法是从零训练独立小模型 |

### 8.2 数据泄露 bug 详解

`stage2_train_v3.py` 原版的关键缺陷：

```python
# 原代码（错误）：
tokens, hids_dict = load_data()  # 一次性加载全部 512 条
# 训练 batch: sample_batch(tokens, hids_dict, ...) 用 torch.randint(0, N=512, ...)
# 验证 batch: sample_batch(tokens, hids_dict, ...) 同样在 N=512 中采样
# N_eval = 512 覆盖整个训练集 —— 没有任何 held-out 数据
```

报告的 97.95% 接受率本质是 draft 模型对 512 条训练样本的**记忆**，而非对未知数据的泛化能力。

### 8.3 保留的有效贡献

1. **合成任务消融（§5.1）**：合成任务的 token shift（+0.435）与 LayerNorm（+0.228）消融实验不涉及数据泄露，结论保留。
2. **加速比公式修正（§5.4.5）**：target 验证 vl 个 token 是 1 次 forward `[B, T=vl]` 而非 vl 次独立 forward，这是纯理论分析，结论保留。
3. **cross-attn context 长度决定性优化**：在合成任务与 v1→v2 真实 target 实验中均观察到此规律，方向性结论保留（但 v3 的 97%+ 数值无效）。

### 8.4 重做方案

基于正确方法学重新实验：

| 维度 | 原方案（已撤回） | 重做方案 |
|---|---|---|
| 数据集 | 自生成 512 条 | `mlabonne/open-perfectblend`（DSpark 论文同源，130 万条对话） |
| 序列数 | 512 | 100,000（10 万条） |
| 划分 | 无（全训练） | 80/10/10 train/val/test，完全独立 |
| Target | RWKV-7 0.1B | RWKV-7 2.9B (g1a) |
| Drafter | D=256, 2 层 | D=1280, 10 层, RWKV-7 head rank=512（500M 参数） |
| Hidden 层 | [0,3,6,9,11] | [0,3,6,9,11]（保持） |
| 训练步数 | 5000 | 20000（4 epoch） |
| 存储 | 单文件 fp32 | 分 chunk fp16（10 文件 × 1W 条） |
| Anchor 切片 | 随机截断 | 从 "Assistant: " 之后切片（符合推理场景） |

### 8.5 初步验证结果

在重做方案的小规模验证中（10 prompts × 40 tokens，K=2，α=0.5），使用 g1a-0.4B 作为 draft 基线、g1a-2.9B 作为 target：

| 配置 | 接受率 |
|---|---|
| 修复 WKV state 维度顺序 bug 前 | 33%（输出乱码） |
| 修复后（K=2, α=0.5） | **85.83%** |
| 修复后（K=2, α=1.0） | 93.23% |

此结果仅证明 Python 实现修复后能正常工作，**真正的加速比仍需等待 500M DSpark drafter 训练完成后评估**。当前 0.4B draft 因前向过慢（406 次 draft 迭代 vs 41 次 target）无法实现端到端加速，这正是 DSpark 半自回归架构（一次 forward 生成 K token）要解决的核心问题。

### 8.6 关键 bug 修复记录

1. **WKV state 维度顺序错误**：旧代码 `state[h, v, k]` 与 web-rwkv shader `state[k, v]` 转置，导致 sa 计算错误，输出乱码、接受率从 85% 跌到 33%。修复后接受率恢复正常。
2. **w 变换公式错误**：原用 `sigmoid(w)`，正确为 `exp2(-0.875/(1+exp2(-1.443*w)))-1`（web-rwkv 版本）。
3. **wkv_state 更新顺序**：S_kk 必须用衰减前的旧 state 计算，否则每步都重置。

### 8.7 完整失败存档

所有失败方案代码与报告已归档至：
- `参考/dspark_rwkv/archive_failed/`（16 个失败报告）
- `dspark-rwkv-repo/archive/stage9_13_failed/`（43 个 stage9-13 代码）
- `dspark-rwkv-repo/docs/archive_failed/`（7 个失败报告）

总结见 `参考/dspark_rwkv/失败存档总结.md`。
