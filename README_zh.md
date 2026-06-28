# DSpark-RWKV

> 将 DSpark 半自回归推测解码适配于 RWKV-7 target。

**语言**: [English](README.md) | [中文](README_zh.md)

**论文**: [English](docs/paper_en.md) | [中文](docs/paper_zh.md)

本仓库复现并适配 DeepSeek 与北京大学联合提出的 [DSpark](https://github.com/deepseek-ai/DeepSpec) 框架至 [RWKV-7](https://github.com/BlinkDL/RWKV-LM) 线性 RNN target。原 DSpark 论文实验基于 Transformer 自回归 target（Qwen3 系列），本工作验证其在 RWKV-7 Delta Rule 架构上的可行性。

## ⚠️ 勘误 (2026-06-28)

**之前报告的 97%+ 接受率与 3.75x 加速比无效，已撤回。** 根因如下：

1. **数据泄露（最严重）**：`stage2_train_v3.py` 训练集与验证集使用同一份 512 条序列（`N_eval=512` 覆盖全部训练集，无 train/val/test 划分）。报告的 97.95% 接受率是过拟合记忆，并非泛化能力。
2. **3.75x 加速比基于无效接受率**，因此不可信。
3. **K-state 并发验证方案**（stages 9-13）：所有变体加速比 < 1×，原因是串行 draft forward 过多（406 次 draft 迭代 vs 41 次 target 迭代）。
4. **CUDA / CUDA Graph 单点优化**：将 draft forward 从 6.5ms 降到 2.4ms，但端到端加速比仍 < 1×（架构级串行瓶颈未消除）。
5. **DSpark 架构理解错误**：曾尝试改造 0.4B RWKV-7 作为 drafter；正确做法是从零训练独立小模型。

**保留的有效贡献**：
- 合成任务上的 token shift（+0.435）与 LayerNorm（+0.228）消融实验（无数据泄露）。
- 推测解码加速比公式修正：target 验证 `vl` 个 token 是 1 次 forward `[B, T=vl]`，而非 `vl` 次独立 forward。
- 发现 cross-attention context 长度是决定性优化因素。

**正在进行的重做**（正确方法学）：
- 独立 train/val/test 划分（80/10/10），使用 `mlabonne/open-perfectblend` 数据集 10 万条序列。
- 更大的 500M 参数 drafter（D=1280, L=10, RWKV-7 head rank=512）。
- Target：RWKV-7 2.9B（g1a 系列），5 层 hidden [0,3,6,9,11]。
- 小规模初步验证（10 prompts × 40 tokens, K=2, α=0.5）：g1a-0.4B draft + g1a-2.9B target 接受率 85.83%（修复 WKV state 维度顺序 bug 后）。

完整失败存档见 [`参考/dspark_rwkv/失败存档总结.md`](../参考/dspark_rwkv/失败存档总结.md)。

## 主要贡献（保留）

1. **RWKV-7 DPLR 作为 DSpark 顺序头**：合成任务消融实验证明 token shift（+0.435 接受率）与 LayerNorm（+0.228 接受率）是必要条件。
2. **Cross-attn context 长度是决定性优化**：单位置 → 8 位置 context 使接受率从 33% 升至 70%（合成任务；真实 target 上的结果因数据泄露而无效）。
3. **修正推测解码加速比公式**：target 验证 `vl` 个 token 是 1 次 forward `[B, T=vl]`，而非 `vl` 次独立 forward。此修正推翻了"单 GPU 推测解码无法加速"的错误结论。

## 仓库结构

```
dspark-rwkv/
├── pyproject.toml                # 依赖配置（torch cu124）
├── stage1_experiment.py          # 阶段1：合成任务（单/双 key + 消融）
├── stage2_target.py              # RWKV-7 target（纯 PyTorch）
├── stage2_gen_data.py            # 3 层 hidden 数据生成（旧）
├── stage2_gen_data_5layer.py     # 5 层 hidden 数据生成（旧，无 split）
├── stage2_gen_data_fast.py       # 分 chunk 数据生成（fp16，10W 条，train/val/test split）
├── stage2_train.py               # v1 训练（cross-attn 单位置）
├── stage2_train_v2.py            # v2 优化（8 位置 context + 置信度 BCE）
├── stage2_train_v3.py            # v3 训练（chunked loading，500M drafter）
├── stage3_schedule.py            # 离线调度评估（v1 数据）
├── stage3_schedule_v2.py         # 离线调度评估（v3 数据）
├── stage4_concurrent.py          # 阶段4：并发实测
├── rwkv_tokenizer.py             # RWKV BPE 分词器
├── rwkv_vocab_v20230424.txt      # RWKV 词表
├── cuda/                         # CUDA kernels（WKV fp16/fp32）
├── albatross_ref/                # Albatross 参考代码
├── archive/                      # 历史脚本
│   ├── experiment_v1.py
│   ├── experiment_xor.py
│   ├── stage2_benchmark.py
│   ├── stage2_verify_target.py
│   ├── stage4_e2e_batch1_buggy.py
│   └── stage9_13_failed/         # K-state 方案（加速比 < 1×，已归档）
├── docs/                         # 文档与论文
│   ├── paper_zh.md               # 中文论文
│   ├── paper_en.md               # 英文论文
│   ├── RWKV7公式参考.md
│   └── archive_failed/           # 失败实验报告（已归档）
├── weights/                      # 权重目录（需手动下载）
└── data/                         # 数据目录（生成产物）
```

## 快速开始

### 1. 环境准备

需要 Python 3.12+ 和 NVIDIA GPU（推荐 RTX 3090+，本实验用 RTX 2080 Ti 22GB）。

```bash
# 安装 uv（如未安装）
pip install uv

# 同步依赖（含 CUDA 12.4 torch）
uv sync
```

### 2. 下载 RWKV-7 权重

```bash
mkdir -p weights
# 从 HuggingFace 下载 RWKV-7 2.9B (g1a) 权重放到 weights/ 目录
# https://huggingface.co/BlinkDL/rwkv-7-g1/tree/main
# 同时下载可用于 draft 基线对比的模型，如 rwkv7-g1d-0.4b
```

### 3. 运行实验

```bash
# 阶段1：合成任务（验证 RWKV-7 DPLR 顺序头 + 消融）
uv run python stage1_experiment.py

# 阶段2 v3（重做）：生成 10W 条序列，含正确 train/val/test split
uv run python stage2_gen_data_fast.py

# 阶段2 v3 训练（chunked loading，500M drafter，20000 步）
uv run python -c "from stage2_train_v3 import run; run(8, 20000)"
```

## 加速比公式

设 $T_{draft}$ 为 draft 生成一个 block 的时间，$T_{target}(N, T)$ 为 target 在 batch=N、序列长 T 时的 forward 时间，$\tau$ 为每轮期望接受 token 数。

$$
\text{speedup} = \frac{\tau \cdot T_{target}(N, 1)}{T_{draft}(N) + T_{target}(N, \text{vl})}
$$

**关键观察**：target 验证 `vl` 个 token 只需 1 次 forward `[B, T=vl]`（而非 `vl` 次独立 forward）。GPU 并行处理多 token 的时间远小于 `vl` 倍单 token 时间——这是 DSpark 加速的核心来源。

## 参考

- DSpark 论文：https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf
- RWKV-7 仓库：https://github.com/BlinkDL/RWKV-LM
- Albatross 推理库：https://github.com/BlinkDL/Albatross
- RWKV-7 numpy 实现：https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py
- 训练数据：https://huggingface.co/datasets/mlabonne/open-perfectblend

## 引用

如使用本工作，请引用：

```bibtex
@misc{dspark-rwkv2026,
  title={DSpark-RWKV: Adapting Semi-Autoregressive Speculative Decoding to RWKV-7},
  author={cgisky1980},
  year={2026},
  url={https://github.com/cgisky1980/dspark-rwkv}
}
```

并引用原 DSpark 论文：

```bibtex
@misc{cheng2025dspark,
  title={DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation},
  author={Cheng, Xin and Yu, Xingkai and Shao, Chenze and others},
  year={2025},
  url={https://github.com/deepseek-ai/DeepSpec}
}
```

## License

MIT License - 见 [LICENSE](LICENSE)
