# Legacy Scripts Archive

> 本目录存放已被取代的早期版本脚本，保留供历史参考，不再维护。
> 当前活跃脚本位于仓库根目录：`stage2_target.py`, `stage2_gen_data_fast.py`, `stage2_train_v3.py`, `rwkv_tokenizer.py`。

## 文件清单与失效原因

| 文件 | 用途 | 失效原因 |
|---|---|---|
| `stage1_experiment.py` | 阶段1 合成任务实验（单/双 key + 消融） | 公式验证已完成，结论已写入论文 §5.1，无需再运行 |
| `stage2_gen_data.py` | v1 数据生成（0.1B target + 3 层 + 512 序列，无 split） | 数据规模、target 维度、split 均不符合当前要求 |
| `stage2_gen_data_5layer.py` | 5 层 hidden 数据生成（2.9B target + 2048 序列） | 数据规模偏小，已被 `stage2_gen_data_fast.py`（100K chunked）取代 |
| `stage2_gen_data_sharegpt.py` | ShareGPT 数据源尝试版（4096 序列） | 数据源尝试性版本，已改用 `mlabonne/open-perfectblend` |
| `stage2_gen_data_text.py` | LAMBADA 文本数据生成（512 序列） | 对照实验已结束 |
| `stage2_train.py` | v1 训练（D=256, 2 层, 3 层 hidden, 1500 步） | 基线版本，已被 v3 取代 |
| `stage2_train_v2.py` | v2 优化训练（cross-attn 8 位置 + 置信度 BCE, 3000 步） | v2 优化点已合并到 v3 |
| `stage2_train_v3_text.py` | LAMBADA 文本对比训练（5000 步） | 对照实验脚本，对照已完成 |
| `stage3_schedule.py` | v1 离线调度评估 | 基于 v1 接受率，已被 v2 取代 |
| `stage3_schedule_v2.py` | v2 离线调度评估 | ⚠️ 硬编码了早期数据泄露版本的 97%+ 无效接受率，**不可复用** |
| `stage4_concurrent.py` | 并发实测脚本 | ⚠️ 1) 硬编码 97%+ 无效接受率；2) import `load_data` 但 v3 已改为 `load_chunk`，API drift，运行会报错 |

## ⚠️ 重要警告

`stage3_schedule_v2.py` 与 `stage4_concurrent.py` 硬编码的接受率来自数据泄露版本的无效数据。
**新训练完成后必须重写这两个脚本**，不可直接复用。

## 活跃脚本（仓库根目录）

| 脚本 | 作用 |
|---|---|
| `stage2_target.py` | RWKV-7 2.9B target 模型（纯 PyTorch，fp16 GPU） |
| `stage2_gen_data_fast.py` | 分 chunk 数据生成（10W 条，open-perfectblend，fp16） |
| `stage2_train_v3.py` | v3 训练（500M drafter, chunked loading, 20000 步） |
| `rwkv_tokenizer.py` | RWKV BPE 分词器（基础组件） |
