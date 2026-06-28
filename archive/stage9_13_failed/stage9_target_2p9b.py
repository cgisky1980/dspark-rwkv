"""DSpark → RWKV · 阶段9：2.9B target (GPU FP16)

基于 albatross_ref/rwkv7.py 参考实现，使用 F.linear 和正确转置规则：
- 权重转置：att.g1/g2/a1/a2/w1/w2/v1/v2, ffn.value.weight 需要 .t()
- att.receptance/key/value/output.weight, ffn.key.weight, head.weight 不转置
- forward 全部用 F.linear(x, W) = x @ W.t()，要求 W 是 [out, in]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

WEIGHTS = Path(__file__).parent / "weights" / "rwkv7-2.9b.pth"
HEAD_SIZE = 64
SQRT_E = math.sqrt(math.e)
DEVICE = "cuda"
DTYPE = torch.float16  # RTX 2080 Ti FP16 性能好


def lerp(x, y, w):
    return x + w * (y - x)


def layer_norm(x, w, b, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


def group_norm(x, w, b, eps=64e-5):
    *lead, C = x.shape
    H = C // HEAD_SIZE
    x = x.reshape(*lead, H, HEAD_SIZE)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x = (x - mean) / (var + eps).sqrt()
    x = x.reshape(*lead, C)
    return x * w + b


class RWKV7Target2p9B:
    """RWKV-7 2.9B target (GPU FP16, 纯 PyTorch)。

    forward(tokens, state, return_hidden_layers) -> (logits, hidden_layers)
      - tokens: [B, T]
      - logits: [B, T, V]
      - hidden_layers: list of [B, T, C]，每层残差流
    """
    def __init__(self, path=WEIGHTS):
        print(f"加载权重: {path}")
        z = torch.load(path, map_location="cpu", weights_only=True)
        r_k_raw = z["blocks.0.att.r_k"].squeeze()
        self.H, self.N = r_k_raw.shape
        C = self.H * self.N
        keys = list(z.keys())
        max_layer = -1
        # 参考 albatross_ref/rwkv7.py line 119-120 的转置规则
        # att.g1/g2/a1/a2/w1/w2/v1/v2 需要转置
        # 但不同版本权重维度顺序可能不同：
        # - g1h-2.9B: w1 [C, mid] = [in, out] → 需转置
        # - g1a-0.4B: w1 [mid, C] = [out, in] → 不需转置
        # 判断方法：g1/w1/a1/v1 的对偶是 g2/w2/a2/v2
        # 如果 w1.shape[0] == C，说明 w1 是 [in, out]，需要转置，w2 也需要转置
        # 如果 w1.shape[0] != C，说明 w1 已是 [out, in]，不需要转置，w2 也不需要
        # 用 layer 0 的 w1 判断整个模型
        w1_key = "blocks.0.att.w1"
        if w1_key in z and z[w1_key].dim() == 2:
            need_transpose = (z[w1_key].shape[0] == C)
        else:
            need_transpose = True  # 默认转置
        print(f"  权重转置判断: w1.shape={z[w1_key].shape}, C={C}, need_transpose={need_transpose}")
        for k in keys:
            v = z[k].squeeze().to(DTYPE)
            need_t_keys = ('att.g1', 'att.g2', 'att.a1', 'att.a2',
                          'att.w1', 'att.w2', 'att.v1', 'att.v2')
            if need_transpose and any(kw in k for kw in need_t_keys):
                v = v.t()
            if k.endswith("att.r_k"):
                v = v.flatten()
            z[k] = v.contiguous()
            parts = k.split(".")
            if parts[0] == "blocks":
                max_layer = max(max_layer, int(parts[1]))
        self.z = z
        self.n_layer = max_layer + 1
        self.C = C
        self.V = z["emb.weight"].shape[0]
        # emb 预处理：layer_norm with ln0
        z["emb.weight"] = layer_norm(z["emb.weight"], z["blocks.0.ln0.weight"], z["blocks.0.ln0.bias"])
        # 权重搬到 GPU
        self.z = {k: v.to(DEVICE) for k, v in z.items()}
        print(f"模型: L={self.n_layer} C={self.C} H={self.H} N={self.N} V={self.V}")
        print(f"权重已搬到 {DEVICE} ({DTYPE})")

    def zero_state(self, B, device=DEVICE):
        return [
            torch.zeros(self.n_layer, 2, B, self.C, dtype=DTYPE, device=device),
            torch.zeros(self.n_layer, B, self.H, self.N, self.N, dtype=DTYPE, device=device),
        ]

    @torch.no_grad()
    def forward(self, tokens, state, return_hidden_layers=None):
        """tokens: [B, T] -> (logits [B,T,V], hidden_layers list of [B,T,C])"""
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
        logits = F.linear(x, z["head.weight"])
        return logits, hidden_layers

    @torch.no_grad()
    def forward_seq_with_snapshots(self, tokens, state):
        """逐 token forward T=1，每步保存完整 state snapshot

        返回 (logits_list, state_snapshots)
          logits_list[t] = 看到前 t 个 token 后预测下一个 token 的 logits [1, V]
            logits_list[0] = 初始 logits（forward 前，None）
            logits_list[t] = forward(tokens[0:t]) 后的 logits
          state_snapshots[t] = 看到前 t 个 token 后的 state
            state_snapshots[0] = 初始 state clone
            state_snapshots[T] = 最终 state

        用于 speculative decoding: 拒绝时直接取 state_snapshots[n_accept]
        """
        T = tokens.shape[1]
        logits_list = [None]  # logits_list[0] = None（初始无 logits）
        state_snapshots = [[s.clone() for s in state]]  # state_snapshots[0] = 初始 state

        cur_logits = None
        for t in range(T):
            tok = tokens[:, t:t+1]  # [1, 1]
            cur_logits, _ = self.forward(tok, state, return_hidden_layers=[])
            logits_list.append(cur_logits)  # logits_list[t+1] = forward tokens[0:t+1] 后的 logits
            state_snapshots.append([s.clone() for s in state])

        return logits_list, state_snapshots

    def tmix(self, layer, x, shift_state, v_first, wkv_state, p):
        z = self.z
        B, T, C = x.shape
        H, N = self.H, self.N
        prev = torch.cat([shift_state[0].unsqueeze(1), x[:, :-1, :]], dim=1)
        shift_state[0] = x[:, -1, :]
        xr = lerp(x, prev, z[p+"x_r"])
        xw = lerp(x, prev, z[p+"x_w"])
        xk = lerp(x, prev, z[p+"x_k"])
        xv = lerp(x, prev, z[p+"x_v"])
        xa = lerp(x, prev, z[p+"x_a"])
        xg = lerp(x, prev, z[p+"x_g"])
        # 全部用 F.linear(x, W) = x @ W.t()，W 是 [out, in]
        r = F.linear(xr, z[p+"receptance.weight"])
        k = F.linear(xk, z[p+"key.weight"])
        v = F.linear(xv, z[p+"value.weight"])
        w = F.linear(torch.tanh(F.linear(xw, z[p+"w1"])), z[p+"w2"], bias=z[p+"w0"])
        a = torch.sigmoid(F.linear(F.linear(xa, z[p+"a1"]), z[p+"a2"], bias=z[p+"a0"]))
        g = F.linear(torch.sigmoid(F.linear(xg, z[p+"g1"])), z[p+"g2"])
        kk = k * z[p+"k_k"]
        k = k * (1 + (a - 1) * z[p+"k_a"])
        if layer == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(F.linear(F.linear(xv, z[p+"v1"]), z[p+"v2"], bias=z[p+"v0"]))
        # w 变换：web-rwkv 官方推理库用 exp(-0.606531 * sigmoid(w))
        # 参考 web-rwkv/src/shaders/time_mix_v7.wgsl:69
        # 0.606531 = exp(-0.5)，w 范围 [exp(-0.606531)≈0.545, 1.0]，直接作衰减因子
        w = torch.exp(-0.606531 * torch.sigmoid(w))
        kk = kk.view(B, T, H, N)
        k = k.view(B, T, H, N)
        v = v.view(B, T, H, N)
        r = r.view(B, T, H, N)
        w = w.view(B, T, H, N)
        a = a.view(B, T, H, N)
        kk_norm = kk / (kk.norm(dim=-1, keepdim=True) + 1e-12)
        y = torch.zeros(B, T, H * N, dtype=DTYPE, device=x.device)
        for t in range(T):
            kt = k[:, t]
            vt = v[:, t]
            rt = r[:, t]
            wt = w[:, t]
            at = a[:, t]
            kkt = kk_norm[:, t]
            # web-rwkv shader line 178-206:
            # state 维度: [B, H, k_index, v_index]
            # sa[v] = sum_k state[k, v] * a[k]  (用更新前 state)
            # state[k, v] = state[k, v] * w[k] + k[k] * v[v] + sa * b[k]
            # y[v] = sum_k r[k] * state[k, v]
            # 注意：w, k, r, kk, b 都是按 k_index 索引，v 按 v_index 索引
            S_kk = (wkv_state * (-kkt).unsqueeze(-1)).sum(dim=-2)  # sa[v] = sum_k state[k,v] * (-kk[k])
            wkv_state.mul_(wt.unsqueeze(-1))  # state[k,v] *= w[k]
            wkv_state.add_(S_kk.unsqueeze(-2) * (kkt * at).unsqueeze(-1))  # + sa[v] * b[k] = sa * (kk*a)
            wkv_state.add_(kt.unsqueeze(-1) * vt.unsqueeze(-2))  # + k[k] * v[v] (outer)
            yt = (wkv_state * rt.unsqueeze(-1)).sum(dim=-2)  # y[v] = sum_k r[k] * state[k,v]
            y[:, t] = yt.reshape(B, H * N)
        y = group_norm(y, z[p+"ln_x.weight"], z[p+"ln_x.bias"])
        rkr = (r.reshape(B, T, H, N) * k * z[p+"r_k"].view(1, 1, H, N)).sum(dim=-1, keepdim=True)
        y = y + (rkr * v.reshape(B, T, H, N)).reshape(B, T, H * N)
        return F.linear((y * g), z[p+"output.weight"]), v_first

    def cmix(self, x, shift_state, p):
        z = self.z
        prev = torch.cat([shift_state[1].unsqueeze(1), x[:, :-1, :]], dim=1)
        shift_state[1] = x[:, -1, :]
        k = lerp(x, prev, z[p+"x_k"])
        k = torch.relu(F.linear(k, z[p+"key.weight"])) ** 2
        return F.linear(k, z[p+"value.weight"])
