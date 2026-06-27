# DSpark-RWKV

> 将 DSpark 半自回归推测解码适配于 RWKV-7 target，并发场景下实测达 **3.75x 加速**。

本仓库复现并适配 DeepSeek + 北大联合提出的 [DSpark](https://github.com/deepseek-ai/DeepSpec) 框架至 [RWKV-7](https://github.com/BlinkDL/RWKV-LM) 线性 RNN target。原 DSpark 论文实验基于 Transformer 自回归 target（Qwen3 系列），本工作验证其在 RWKV-7 Delta Rule 架构上的可行性。

## 核心结果

| 指标 | 数值 |
|---|---|
| Target | RWKV-7 0.1B (L=12, C=768, H=12, N=64, V=65536) |
| 接受率 | **97.95%** (block=4, v3 训练) |
| RWKV-7 vs GRU 顺序头 | **+31%** (接受率领先) |
| 并发加速比 (batch=2, verify_len=4) | **3.75x** |
| 加速比范围 (batch=1-32) | 1.56x – 3.75x |

## 关键贡献

1. **RWKV-7 DPLR 作为 DSpark 顺序头**：通过消融实验证明 token shift（+0.435 接受率）与 LayerNorm（+0.228 接受率）是必要条件。
2. **Cross-attn context 长度是决定性优化**：单位置 → 8 位置 context 使接受率从 33% 升至 70%。
3. **修正推测解码加速比公式**：target 验证 vl 个 token 是 1 次 forward `[B, T=vl]`，而非 vl 次独立 forward。此修正推翻了"单 GPU 推测解码无法加速"的错误结论。

## 仓库结构

```
dspark-rwkv/
├── pyproject.toml                # 依赖配置（torch cu124）
├── stage1_experiment.py          # 阶段1：合成任务（单/双 key + 消融）
├── stage2_target.py              # RWKV-7 0.1B target（纯 PyTorch）
├── stage2_gen_data.py            # 3 层 hidden 数据生成
├── stage2_gen_data_5layer.py     # 5 层 hidden 数据生成
├── stage2_train.py               # v1 训练（cross-attn 单位置）
├── stage2_train_v2.py            # v2 优化（8 位置 context + 置信度 BCE）
├── stage2_train_v3.py            # v3 综合优化（5 层 + 5000 步，97%+）
├── stage3_schedule.py            # 离线调度评估（v1 数据）
├── stage3_schedule_v2.py         # 离线调度评估（v3 数据）
├── stage4_concurrent.py          # 阶段4：并发实测（核心实验）
├── archive/                      # 历史脚本
│   ├── experiment_v1.py
│   ├── experiment_xor.py
│   ├── stage2_benchmark.py
│   ├── stage2_verify_target.py
│   └── stage4_e2e_batch1_buggy.py
├── docs/                         # 文档与论文
│   ├── paper_zh.md               # 中文论文
│   ├── paper_en.md               # 英文论文
│   ├── 方案.md
│   ├── RWKV7公式参考.md
│   ├── 总报告.md
│   ├── 阶段1实验报告.md
│   ├── 阶段1实验汇总表.md
│   ├── 阶段2实验报告.md
│   ├── 阶段2v2优化报告.md
│   └── 阶段3-4报告.md
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

### 2. 下载 RWKV-7 0.1B 权重

```bash
mkdir -p weights
# 从 HuggingFace 下载 rwkv7-0.1b.pth 放到 weights/ 目录
# https://huggingface.co/BlinkDL/rwkv-7-pile/tree/main
```

### 3. 运行实验

```bash
# 阶段1：合成任务（验证 RWKV-7 DPLR 顺序头 + 消融）
uv run python stage1_experiment.py

# 阶段2：生成训练数据（5 层 hidden）
uv run python stage2_gen_data_5layer.py

# 阶段2 v3：训练（5000 步，约 30 分钟）
uv run python stage2_train_v3.py

# 阶段3：离线调度评估
uv run python stage3_schedule_v2.py

# 阶段4：并发实测（核心实验）
uv run python stage4_concurrent.py
```

## 四阶段实验路线

| 阶段 | 内容 | 关键结果 |
|---|---|---|
| 1 | 顺序头单测（合成任务） | token shift +0.435, LayerNorm +0.228 |
| 2 | 真实 RWKV target + DSpark 双轨 | 接受率 97%+, RWKV 领先 GRU 31% |
| 3 | 离线调度评估 | verify_len=1 离线预测 1.49x |
| 4 | 端到端并发实测 | batch=2, verify_len=4 达 **3.75x** |

## 加速比公式

设 $T_{draft}$ 为 draft 生成一个 block 的时间，$T_{target}(N, T)$ 为 target 在 batch=N、序列长 T 时的 forward 时间，$\tau$ 为每轮期望接受 token 数。

$$
\text{speedup} = \frac{\tau \cdot T_{target}(N, 1)}{T_{draft}(N) + T_{target}(N, \text{vl})}
$$

**关键观察**：target 验证 vl 个 token 只需 1 次 forward `[B, T=vl]`（而非 vl 次独立 forward）。GPU 并行处理多 token 的时间远小于 vl 倍单 token 时间——这是 DSpark 加速的核心来源。

## 与 DSpark 原论文对比

| 维度 | DSpark 论文 | 本工作 |
|---|---|---|
| Target | Qwen3-4B/8B/14B (Transformer) | RWKV-7 0.1B (线性 RNN) |
| 顺序头 | GRU/RNN | RWKV-7 DPLR（领先 GRU 31%） |
| 接受率 | ~70-80% | 97%+ (v3) |
| 加速比 | +60%~85% 用户速度 (V4-Flash) | 1.56-3.75x 吞吐 |
| 部署环境 | 多并发服务 (DeepSeek-V4) | 单 GPU batch=N |

## 参考

- DSpark 论文：https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf
- RWKV-7 仓库：https://github.com/BlinkDL/RWKV-LM
- Albatross 推理库：https://github.com/BlinkDL/Albatross
- RWKV-7 numpy 实现：https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py

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
