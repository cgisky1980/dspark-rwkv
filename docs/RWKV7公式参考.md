# RWKV-7 公式参考（基于官方实现核对）

来源：
- `参考/Albatross/_ref_slower_/reference/rwkv7.py`（CUDA 参考实现）
- `https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py`（numpy 参考实现）
- `https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py`（numpy 简化版）
- `https://zhiyuan1i.github.io/posts/dplr-mathematics`（DPLR 数学理论）

## 状态更新公式（DPLR 右乘形式，RWKV 原生）

```python
# run_rwkv7_qwen35.py L54-56
def DPLR_RWKV(S, R, W, K, V, A, B):
    # S: [H, N, N] (H heads, N=64 head_size)
    # R,W,K,V,A,B: [H, N] 每头的 r/w/k/v/a/b 向量
    S = S*W + (S@A)*B + V⊗K           # 注意是 + (S@A)*B，不是 (A·S)⊗K
    return S@R, S
```

即每 head 的状态更新：
```
S_h = S_h * w_h + (S_h @ a_h) * b_h + v_h ⊗ k_h
```

**与我阶段 1 错误实现的对比**：
- ❌ 我写的：`s = s*w + k⊗v + (a·s)⊗k`  —— 把 b 和 k 混淆了
- ✅ 正确：`s = s*w + (s@a)*b + v⊗k`    —— b 是独立向量，不是 k

## TMix 完整流程（run_rwkv7_qwen35.py L182-201）

```python
x = LAYER_NORM(x0, ln1_w, ln1_b)
prev, state["x"] = state["x"], x
# 6 路 time-mix（lerp 当前与上一 token）
xr = LERP(x, prev, x_r)  # x + x_r*(prev - x)
xw, xk, xv, xa, xg = ... (同理)

r = xr @ rW
k = xk @ kW
v = xv @ vW
# v 残差（layer 0 用 v_first = v，后续层做 lerp）
if v_first is None: v_first = v
else: v = LERP(v, v_first, sigmoid(v0 + xv @ v1 @ v2))

w = exp(-sigmoid(w0 + tanh(xw @ w1) @ w2) / sqrt(e))   # 注意 /sqrt(e) 和 exp(-...)
a = sigmoid(a0 + xa @ a1 @ a2)

kk = k * k_k
k = LERP(k, k*a, k_a)                  # k = k + k_a*(k*a - k)
r,w,k,v,kk,a = [z.reshape(H,N) for z in ...]
kk = L2_RWKV(kk)                        # kk = kk / max(||kk||, 1e-12)，对 kk 归一化，不是对 S

y, state["rnn"] = DPLR_RWKV(state["rnn"], r, w, k, v, kk, -kk*a)
# 即 S = S*w + (S@kk)*(-kk*a) + v⊗k
#      S = S*w - (S@kk)*(kk*a) + v⊗k

y = GROUP_NORM(y, ln_x_w, ln_x_b, eps=64e-5)   # group_norm over H groups
y += (sum(r * k * r_k, axis=1, keepdims=True) * v).reshape(-1)   # 残差项
g = sigmoid(xg @ g1) @ g2
output = x0 + (y * g) @ oW
```

## 关键工程细节（之前遗漏）

1. **kk 的 L2 归一化**：`kk = kk / max(||kk||, 1e-12)`，对 kk 向量归一化，不是对整个状态 S 归一化
2. **w 的计算**：`w = exp(-sigmoid(w0 + tanh(xw@w1)@w2) / sqrt(e))`，范围约 [exp(-1/sqrt(e)), 1] ≈ [0.545, 1]
   - 我之前用 `sigmoid*0.9+0.05` 完全错误
3. **k 的修正**：`k = LERP(k, k*a, k_a)`，让 k 朝 `k*a` 方向偏移
4. **v 的残差**：`v = LERP(v, v_first, sigmoid(v0 + xv@v1@v2))`，跨层 v 残差
5. **输出残差**：`y += (r*k*r_k).sum * v`，除了 `S@r` 还要加这个
6. **group_norm**：对 y 做 H 组 group_norm，eps=64e-5
7. **DPLR 的 b 向量 = -kk*a**：在 RWKV-7 里，`b = -kk*a`（负号！），所以 `S = S*w - (S@kk)*(kk*a) + v⊗k`
   - 这正是 DPLR 数学文说的"低秩更新"，b=-kk*a 提供跨维度耦合

## DPLR 理论视角（zhiyuan1i 文）

RWKV-7 状态更新是 DPLR（Diagonal Plus Low Rank）结构：
```
S_t = D_t * S_{t-1} + (S_{t-1} @ a_t) * b_t + v_t ⊗ k_t
    = (D_t + b_t a_t^T) S_{t-1} + k_t v_t^T    （左乘视角）
```
其中 D_t = diag(w_t) 对角衰减，b_t a_t^T 低秩更新。

DPLR 支持 chunk-wise Affine 并行：`S' = M*S + B`，M/B 可由 chunk 内 K/V/A/B/G 计算。
这是阶段 2 并行训练 RWKV-7 target 的理论基础。

## Albatross CUDA 实现的权重命名（faster3_2605/rwkv7_fast_v3.py）

权重 key 格式：`blocks.{layer}.att.{param}` 和 `blocks.{layer}.ffn.{param}`
- att: x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, receptance.weight, key.weight, value.weight, output.weight, ln_x.weight, ln_x.bias
- ffn: x_k, key.weight, value.weight
- 顶层: emb.weight, ln0.weight/bias, ln_out.weight/bias, head.weight
- att.r_k: [H, N] 形状（用于检测 H 和 N）

## 对阶段 1 实验的修正

我阶段 1 的 Rwkv7Head 实现错误：
1. 公式写错：用 `(a·s)⊗k` 应为 `(s@a)*b`
2. w 范围错：应为 `exp(-sigmoid(...)/sqrt(e))`
3. 缺 k 修正、v 残差、kk L2 归一化、输出残差项
4. 对 S 整体 L2 归一化是错的，应对 kk 归一化

阶段 1 的"首位弱后位强"结论可能部分来自实现错误。需要用正确公式重跑。
但阶段 1b 的"短 block 下 RWKV 不如 GRU"结论可能仍成立，因为 Delta Rule 的主动删除特性还在（b=-kk*a 是负号）。
