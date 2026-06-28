"""DSpark → RWKV 复现 · 阶段2：真实 RWKV-7 target + DSpark 双轨架构

基于 Albatross/_ref_slower_/reference/rwkv7.py 的公式，用纯 PyTorch CPU 加载
RWKV-7 0.1B 权重作为 target，搭 DSpark 双轨架构：
- 并行主干：小 Transformer，cross-attn 到 target 多层 hidden
- 顺序头：GRU / RWKV-7 DPLR
- 置信度头：预测接受率

阶段2a（本文件）：先验证 target 能正确加载和推理，并生成训练数据
（token 序列 + target 多层 hidden + target logits）。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

torch.manual_seed(42)

WEIGHTS = Path(__file__).parent / "weights" / "rwkv7-2.9b.pth"
HEAD_SIZE = 64
SQRT_E = math.sqrt(math.e)
DEVICE = "cuda"
DTYPE = torch.float16  # GPU 用 fp16 省显存


def lerp(x, y, w):
    return x + w * (y - x)


def layer_norm(x, w, b, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


def group_norm(x, w, b, eps=64e-5):
    """x: [..., H*N] -> group_norm over H groups."""
    *lead, C = x.shape
    H = C // HEAD_SIZE
    x = x.reshape(*lead, H, HEAD_SIZE)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x = (x - mean) / (var + eps).sqrt()
    x = x.reshape(*lead, C)
    return x * w + b


class RWKV7Target:
    """RWKV-7 0.1B target（纯 PyTorch CPU fp32，基于 Albatross 参考实现）。

    只实现 forward_seq_batch，返回 (last_logits, all_layer_hidden, all_logits)。
    all_layer_hidden: list of [B, T, C]，每层的 token 输出（残差流）
    all_logits: [B, T, V]，每位置的 logits
    """
    def __init__(self, path):
        print(f"加载权重: {path}")
        z = torch.load(path, map_location="cpu", weights_only=True)
        # 先读 r_k 的 shape（flatten 前是 [H, N]）
        r_k_raw = z["blocks.0.att.r_k"].squeeze()
        self.H, self.N = r_k_raw.shape  # [H, N]
        # 转换权重格式（参考 Albatross rwkv7.py）
        keys = list(z.keys())
        max_layer = -1
        for k in keys:
            v = z[k].squeeze().to(DTYPE)
            if "key.weight" in k or "value.weight" in k or "receptance.weight" in k or "output.weight" in k or "head.weight" in k:
                v = v.t()
            if k.endswith("att.r_k"):
                v = v.flatten()
            z[k] = v.contiguous()
            parts = k.split(".")
            if parts[0] == "blocks":
                max_layer = max(max_layer, int(parts[1]))
        self.n_layer = max_layer + 1
        self.C = self.H * self.N
        self.V = z["emb.weight"].shape[0]
        # emb 预处理：layer_norm with ln0（在 CPU 上做完再搬设备）
        z["emb.weight"] = layer_norm(z["emb.weight"], z["blocks.0.ln0.weight"], z["blocks.0.ln0.bias"])
        # 搬到目标设备
        self.z = {k: v.to(DEVICE) for k, v in z.items()}
        print(f"模型: L={self.n_layer} C={self.C} H={self.H} N={self.N} V={self.V}")

    def zero_state(self, B, device=DEVICE):
        return [
            torch.zeros(self.n_layer, 2, B, self.C, dtype=DTYPE, device=device),  # shift state
            torch.zeros(self.n_layer, B, self.H, self.N, self.N, dtype=DTYPE, device=device),  # wkv state
        ]

    @torch.no_grad()
    def forward(self, tokens, state, return_hidden_layers=None):
        """tokens: [B, T] -> (logits_all [B,T,V], hidden_layers list of [B,T,C])"""
        z = self.z
        B, T = tokens.shape
        if return_hidden_layers is None:
            return_hidden_layers = list(range(self.n_layer))
        x = z["emb.weight"][tokens]  # [B,T,C]
        v_first = torch.zeros_like(x)
        hidden_layers = []
        for i in range(self.n_layer):
            p = f"blocks.{i}."
            att = p + "att."
            ffn = p + "ffn."
            xx = layer_norm(x, z[p+"ln1.weight"], z[p+"ln1.bias"])
            xx, v_first = self.tmix(i, xx, state[0][i], v_first, state[1][i], att)
            x = x + xx
            if i in return_hidden_layers:
                hidden_layers.append(x.clone())
            xx = layer_norm(x, z[p+"ln2.weight"], z[p+"ln2.bias"])
            x = x + self.cmix(xx, state[0][i], ffn)
        x = layer_norm(x, z["ln_out.weight"], z["ln_out.bias"])
        logits = x @ z["head.weight"]  # [B,T,V]
        return logits, hidden_layers

    def tmix(self, layer, x, shift_state, v_first, wkv_state, p):
        z = self.z
        B, T, C = x.shape
        H, N = self.H, self.N
        # time shift: shift_state 是 [2, B, C]，shift_state[0] 是 att 的上一 token
        prev = torch.cat([shift_state[0].unsqueeze(1), x[:, :-1, :]], dim=1)  # [B,T,C]
        shift_state[0] = x[:, -1, :]
        xr = lerp(x, prev, z[p+"x_r"])
        xw = lerp(x, prev, z[p+"x_w"])
        xk = lerp(x, prev, z[p+"x_k"])
        xv = lerp(x, prev, z[p+"x_v"])
        xa = lerp(x, prev, z[p+"x_a"])
        xg = lerp(x, prev, z[p+"x_g"])
        r = xr @ z[p+"receptance.weight"]
        k = xk @ z[p+"key.weight"]
        v = xv @ z[p+"value.weight"]
        w = torch.tanh(xw @ z[p+"w1"]) @ z[p+"w2"]
        a = torch.sigmoid(z[p+"a0"] + (xa @ z[p+"a1"]) @ z[p+"a2"])
        g = torch.sigmoid(xg @ z[p+"g1"]) @ z[p+"g2"]
        kk = k * z[p+"k_k"]
        k = k * (1 + (a - 1) * z[p+"k_a"])
        if layer == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(z[p+"v0"] + (xv @ z[p+"v1"]) @ z[p+"v2"])
        w = torch.sigmoid(z[p+"w0"] + w)
        # DPLR 状态更新：S = S*w - (S@kk)*(kk*a) + v⊗k
        # 逐 token 递归（CPU 简化实现）
        kk = kk.view(B, T, H, N)
        k = k.view(B, T, H, N)
        v = v.view(B, T, H, N)
        r = r.view(B, T, H, N)
        w = w.view(B, T, H, N)
        a = a.view(B, T, H, N)
        # kk L2 归一化
        kk_norm = kk / (kk.norm(dim=-1, keepdim=True) + 1e-12)
        y = torch.zeros(B, T, H * N, dtype=DTYPE, device=x.device)
        for t in range(T):
            kt = k[:, t]  # [B,H,N]
            vt = v[:, t]
            rt = r[:, t]
            wt = w[:, t]
            at = a[:, t]
            kkt = kk_norm[:, t]
            # S: [B,H,N,N]
            # S = S * w（按 N 维衰减，w 是 [B,H,N]）
            wkv_state = wkv_state * wt.unsqueeze(-1)
            # S@kk: [B,H,N,N] @ [B,H,N] -> [B,H,N]（按最后一维求和）
            S_kk = (wkv_state * kkt.unsqueeze(-2)).sum(dim=-1)  # [B,H,N]
            # 低秩更新 b = -kk*a
            wkv_state = wkv_state + S_kk.unsqueeze(-1) * (-kkt * at).unsqueeze(-2)
            # v⊗k
            wkv_state = wkv_state + vt.unsqueeze(-2) * kt.unsqueeze(-1)
            # 输出 y = S@r
            yt = (wkv_state * rt.unsqueeze(-2)).sum(dim=-1)  # [B,H,N]
            y[:, t] = yt.reshape(B, H * N)
        # 注：y 已在正确设备（与 wkv_state 同设备）
        # group_norm
        y = group_norm(y, z[p+"ln_x.weight"], z[p+"ln_x.bias"])
        # 残差项
        rkr = (r.reshape(B, T, H, N) * k * z[p+"r_k"].view(1, 1, H, N)).sum(dim=-1, keepdim=True)
        y = y + (rkr * v.reshape(B, T, H, N)).reshape(B, T, H * N)
        return (y * g) @ z[p+"output.weight"], v_first

    def cmix(self, x, shift_state, p):
        z = self.z
        B, T, C = x.shape
        prev = torch.cat([shift_state[1].unsqueeze(1), x[:, :-1, :]], dim=1)  # [B,T,C]
        shift_state[1] = x[:, -1, :]
        k = lerp(x, prev, z[p+"x_k"])
        k = torch.relu(k @ z[p+"key.weight"]) ** 2
        return k @ z[p+"value.weight"]


def smoke_test():
    """验证 target 能加载和推理。"""
    target = RWKV7Target(WEIGHTS)
    # 简单 token 序列
    tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=DEVICE)
    state = target.zero_state(1)
    logits, hids = target.forward(tokens, state, return_hidden_layers=[0, target.n_layer // 2, target.n_layer - 1])
    print(f"tokens shape: {tokens.shape}")
    print(f"logits shape: {logits.shape}")
    print(f"hidden layers: {len(hids)} x {hids[0].shape}")
    print(f"logits[0,-1,:5]: {logits[0, -1, :5].tolist()}")
    # top-5 预测
    top5 = logits[0, -1].topk(5)
    print(f"top-5 next token: {top5.indices.tolist()} logits: {[f'{x:.2f}' for x in top5.values.tolist()]}")
    print("smoke test OK")


if __name__ == "__main__":
    smoke_test()
