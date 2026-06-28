"""DSpark → RWKV · 阶段9：RWKV-7 0.4B draft + 每层 cross-attn

架构思路：
- Draft backbone: RWKV-7 g1d 0.4B (L=24, C=1024, H=16, N=64)
- 每层 RWKV 后接 cross-attn，接收 target 的 5 层 hidden
- 全量微调（backbone + cross-attn 一起训练）
- 输出 lm_head（draft 自己的 vocab 投影，从 0.4B 原始 head 初始化）

与之前 Transformer draft 对比：
- Transformer draft 需要从头学习 RWKV 分布，跨架构模仿难
- RWKV draft 架构一致，分布接近，只需学习如何利用 target hidden 修正

数据复用 stage9_train.pt（5 层 target hidden + tokens）
"""
import math
import time
import logging
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================================================================
# 参数解析
# ==================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["baseline", "ss"], default="baseline")
parser.add_argument("--steps", type=int, default=10000)
args = parser.parse_args()

MODE = args.mode
SS_MAX = 0.0 if MODE == "baseline" else 0.8
TAG = "baseline" if MODE == "baseline" else "ss"

# ==================================================================
# 日志
# ==================================================================
log_path = Path(__file__).parent / f"stage9_rwkv_draft_{TAG}.log"
logger = logging.getLogger(f"stage9_rwkv_{TAG}")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(ch)

def log(msg):
    logger.info(msg)

# ==================================================================
# 配置
# ==================================================================
torch.manual_seed(42)

DEVICE = "cuda"
DRAFT_WEIGHTS = Path(__file__).parent / "weights" / "rwkv7-g1d-0.4b-20260210-ctx8192.pth"
TRAIN_PATH = Path(__file__).parent / "data" / "stage9_train.pt"
OUT_WEIGHTS = Path(__file__).parent / "weights" / f"stage9_rwkv_{TAG}.pth"

VOCAB = 65536
D_TARGET = 2560          # 2.9B target 的 C
N_TARGET_LAYERS = 5
TARGET_LAYERS = [0, 8, 16, 24, 31]  # target 的 5 层
DRAFT_C = 1024           # 0.4B 的 C
DRAFT_L = 24             # 0.4B 层数
DRAFT_H = 16             # 0.4B head 数
DRAFT_N = 64             # head size
BLOCK = 4                # draft block size
CTX = 8                  # cross-attn context length
LR = 3e-5                # 微调用较小学习率
N_STEPS = args.steps
WARMUP = 500
BS = 32                  # 0.4B + target 一起，batch 适当减小
EVAL_EVERY = 1000
EVAL_SAMPLES = 64
VAL_SPLIT = 1024
SQRT_E = math.sqrt(math.e)

# ==================================================================
# 数据加载
# ==================================================================
def load_data(path):
    d = torch.load(path, map_location="cpu", weights_only=True)
    tokens = d["tokens"]
    hids = {l: d[f"hidden_{l}"] for l in TARGET_LAYERS}
    return tokens, hids


def sample_batch_cpu(tokens, hids_dict, bs, block, ctx):
    """从 CPU 数据采样，返回 GPU tensor。"""
    N, T1 = tokens.shape
    T = T1 - 1
    max_anchor = T - block
    idx = torch.randint(0, N, (bs,))
    anc = torch.randint(ctx, max_anchor + 1, (bs,))

    ctx_list = []
    for l in TARGET_LAYERS:
        ctx_l = torch.stack([hids_dict[l][idx[b], anc[b]-ctx:anc[b]] for b in range(bs)], dim=0)
        ctx_list.append(ctx_l)
    ctx_hidden = torch.cat(ctx_list, dim=-1).to(DEVICE).float()  # [bs, ctx, D_TARGET*5]

    anchor_token = tokens[idx, anc].to(DEVICE)
    # draft 输入：anchor + BLOCK-1 个 mask token（或用真实 token 做 teacher forcing）
    # 这里用真实 token 做 teacher forcing
    block_tokens = torch.stack([tokens[idx, anc + 1 + k] for k in range(block)], dim=1).to(DEVICE)
    # draft 输入序列：[anchor, tok1, tok2, ..., tok_{BLOCK-1}]
    draft_input = torch.cat([anchor_token.unsqueeze(1), block_tokens[:, :-1]], dim=1)  # [bs, BLOCK]
    # 目标：[tok1, tok2, ..., tok_BLOCK]
    target_tokens = block_tokens  # [bs, BLOCK]
    return draft_input, ctx_hidden, target_tokens


# ==================================================================
# RWKV-7 计算（可训练版本）
# ==================================================================
def lerp(x, y, w):
    return x + w * (y - x)


def group_norm(x, w, b, eps=64e-5):
    *lead, C = x.shape
    H = C // DRAFT_N
    x = x.reshape(*lead, H, DRAFT_N)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x = (x - mean) / (var + eps).sqrt()
    x = x.reshape(*lead, C)
    return x * w + b


class CrossAttn(nn.Module):
    """Cross-attention 到 target hidden。"""
    def __init__(self, d_draft, d_target_kv, n_heads=8, ctx_len=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_draft // n_heads
        self.ctx_len = ctx_len
        self.ln_q = nn.LayerNorm(d_draft)
        self.ln_kv = nn.LayerNorm(d_target_kv)
        self.q_proj = nn.Linear(d_draft, d_draft, bias=False)
        self.k_proj = nn.Linear(d_target_kv, d_draft, bias=False)
        self.v_proj = nn.Linear(d_target_kv, d_draft, bias=False)
        self.out_proj = nn.Linear(d_draft, d_draft, bias=False)
        # 可学习的门控
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, ctx_kv):
        """x: [B, T, d_draft], ctx_kv: [B, ctx_len, d_target_kv]"""
        B, T, D = x.shape
        q = self.q_proj(self.ln_q(x))  # [B, T, D]
        k = self.k_proj(self.ln_kv(ctx_kv))  # [B, ctx_len, D]
        v = self.v_proj(self.ln_kv(ctx_kv))

        # reshape to heads
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, T, dh]
        k = k.view(B, self.ctx_len, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, ctx, dh]
        v = v.view(B, self.ctx_len, self.n_heads, self.d_head).transpose(1, 2)

        # scaled dot-product attention
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)  # [B, h, T, ctx]
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # [B, h, T, dh]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        # 门控：初始为 0（不改变 backbone 行为），训练中学习
        return x + torch.tanh(self.gate) * out


class Rwkv7DraftLayer(nn.Module):
    """RWKV-7 单层（可训练）+ cross-attn。"""
    def __init__(self, layer_idx, C, H, N, d_target_kv, n_attn_heads=8, ctx_len=8):
        super().__init__()
        self.layer_idx = layer_idx
        self.C = C
        self.H = H
        self.N = N
        # RWKV-7 att 参数（作为可训练 Parameter）
        p = f"blocks.{layer_idx}.att."
        self.x_r = nn.Parameter(torch.zeros(C))
        self.x_w = nn.Parameter(torch.zeros(C))
        self.x_k = nn.Parameter(torch.zeros(C))
        self.x_v = nn.Parameter(torch.zeros(C))
        self.x_a = nn.Parameter(torch.zeros(C))
        self.x_g = nn.Parameter(torch.zeros(C))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.w1 = nn.Linear(C, C // 16, bias=False)  # 0.4B: w1 是 [C, C/16]
        self.w2 = nn.Linear(C // 16, C, bias=False)
        self.a0 = nn.Parameter(torch.zeros(C))
        self.a1 = nn.Linear(C, C // 16, bias=False)  # 0.4B: a1 是 [C, C/16]
        self.a2 = nn.Linear(C // 16, C, bias=False)
        self.v0 = nn.Parameter(torch.zeros(C))
        self.v1 = nn.Linear(C, C // 32, bias=False)  # 0.4B: v1 是 [C, C/32]
        self.v2 = nn.Linear(C // 32, C, bias=False)
        self.w0 = nn.Parameter(torch.zeros(C))
        self.g1 = nn.Linear(C, C // 8, bias=False)   # 0.4B: g1 是 [C, C/8]
        self.g2 = nn.Linear(C // 8, C, bias=False)
        self.k_k = nn.Parameter(torch.ones(C))
        self.k_a = nn.Parameter(torch.zeros(C))
        self.r_k = nn.Parameter(torch.ones(H, N))
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x_w = nn.Parameter(torch.ones(C))
        self.ln_x_b = nn.Parameter(torch.zeros(C))
        # ffn
        ffn_p = f"blocks.{layer_idx}.ffn."
        self.ffn_x_k = nn.Parameter(torch.zeros(C))
        self.ffn_key = nn.Linear(C, C * 4, bias=False)
        self.ffn_value = nn.Linear(C * 4, C, bias=False)
        # ln
        self.ln1_w = nn.Parameter(torch.ones(C))
        self.ln1_b = nn.Parameter(torch.zeros(C))
        self.ln2_w = nn.Parameter(torch.ones(C))
        self.ln2_b = nn.Parameter(torch.zeros(C))
        # cross-attn
        self.cross_attn = CrossAttn(C, d_target_kv, n_attn_heads, ctx_len)

    def load_pretrained(self, z):
        """从预训练权重加载。"""
        p = f"blocks.{self.layer_idx}."
        att = p + "att."
        ffn = p + "ffn."
        # att 参数
        self.x_r.data.copy_(z[att+"x_r"].squeeze())
        self.x_w.data.copy_(z[att+"x_w"].squeeze())
        self.x_k.data.copy_(z[att+"x_k"].squeeze())
        self.x_v.data.copy_(z[att+"x_v"].squeeze())
        self.x_a.data.copy_(z[att+"x_a"].squeeze())
        self.x_g.data.copy_(z[att+"x_g"].squeeze())
        # Linear 权重：att 的 receptance/key/value/output/w1/w2/a1/a2/v1/v2/g1/g2 原始是 [in, out]
        # （因为原始代码用 x @ W），nn.Linear 是 [out, in]，需 .t()
        # ffn_key/ffn_value/head 原始已是 [out, in]，不转置
        self.receptance.weight.data.copy_(z[att+"receptance.weight"].squeeze().t())
        self.key.weight.data.copy_(z[att+"key.weight"].squeeze().t())
        self.value.weight.data.copy_(z[att+"value.weight"].squeeze().t())
        self.w1.weight.data.copy_(z[att+"w1"].squeeze().t())
        self.w2.weight.data.copy_(z[att+"w2"].squeeze().t())
        self.a0.data.copy_(z[att+"a0"].squeeze())
        self.a1.weight.data.copy_(z[att+"a1"].squeeze().t())
        self.a2.weight.data.copy_(z[att+"a2"].squeeze().t())
        self.v0.data.copy_(z[att+"v0"].squeeze())
        self.v1.weight.data.copy_(z[att+"v1"].squeeze().t())
        self.v2.weight.data.copy_(z[att+"v2"].squeeze().t())
        self.w0.data.copy_(z[att+"w0"].squeeze())
        self.g1.weight.data.copy_(z[att+"g1"].squeeze().t())
        self.g2.weight.data.copy_(z[att+"g2"].squeeze().t())
        self.k_k.data.copy_(z[att+"k_k"].squeeze())
        self.k_a.data.copy_(z[att+"k_a"].squeeze())
        self.r_k.data.copy_(z[att+"r_k"].squeeze())
        self.output.weight.data.copy_(z[att+"output.weight"].squeeze().t())
        self.ln_x_w.data.copy_(z[att+"ln_x.weight"].squeeze())
        self.ln_x_b.data.copy_(z[att+"ln_x.bias"].squeeze())
        # ffn: ffn_key 原始 [4C, C]=[out, in]，ffn_value 原始 [C, 4C]=[out, in]，均不转置
        self.ffn_x_k.data.copy_(z[ffn+"x_k"].squeeze())
        self.ffn_key.weight.data.copy_(z[ffn+"key.weight"].squeeze())
        self.ffn_value.weight.data.copy_(z[ffn+"value.weight"].squeeze())
        # ln
        self.ln1_w.data.copy_(z[p+"ln1.weight"].squeeze())
        self.ln1_b.data.copy_(z[p+"ln1.bias"].squeeze())
        self.ln2_w.data.copy_(z[p+"ln2.weight"].squeeze())
        self.ln2_b.data.copy_(z[p+"ln2.bias"].squeeze())

    def tmix(self, x, shift_state, v_first, wkv_state, layer_idx):
        """RWKV-7 attention（可训练，带梯度）。"""
        B, T, C = x.shape
        H, N = self.H, self.N
        prev = torch.cat([shift_state[0].unsqueeze(1), x[:, :-1, :]], dim=1)
        shift_state[0] = x[:, -1, :]
        xr = lerp(x, prev, self.x_r)
        xw = lerp(x, prev, self.x_w)
        xk = lerp(x, prev, self.x_k)
        xv = lerp(x, prev, self.x_v)
        xa = lerp(x, prev, self.x_a)
        xg = lerp(x, prev, self.x_g)
        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        w = self.w2(torch.tanh(self.w1(xw)))
        a = torch.sigmoid(self.a0 + self.a2(self.a1(xa)))
        g = self.g2(torch.sigmoid(self.g1(xg)))
        kk = k * self.k_k
        k = k * (1 + (a - 1) * self.k_a)
        if layer_idx == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + self.v2(self.v1(xv)))
        w = torch.sigmoid(self.w0 + w)
        kk = kk.view(B, T, H, N)
        k = k.view(B, T, H, N)
        v = v.view(B, T, H, N)
        r = r.view(B, T, H, N)
        w = w.view(B, T, H, N)
        a = a.view(B, T, H, N)
        kk_norm = kk / (kk.norm(dim=-1, keepdim=True) + 1e-12)
        y = torch.zeros(B, T, H * N, dtype=x.dtype, device=x.device)
        for t in range(T):
            kt, vt, rt, wt, at = k[:, t], v[:, t], r[:, t], w[:, t], a[:, t]
            kkt = kk_norm[:, t]
            wkv_state = wkv_state * wt.unsqueeze(-1)
            S_kk = (wkv_state * kkt.unsqueeze(-2)).sum(dim=-1)
            wkv_state = wkv_state + S_kk.unsqueeze(-1) * (-kkt * at).unsqueeze(-2)
            wkv_state = wkv_state + vt.unsqueeze(-2) * kt.unsqueeze(-1)
            yt = (wkv_state * rt.unsqueeze(-2)).sum(dim=-1)
            y[:, t] = yt.reshape(B, H * N)
        y = group_norm(y, self.ln_x_w, self.ln_x_b)
        rkr = (r.reshape(B, T, H, N) * k * self.r_k.view(1, 1, H, N)).sum(dim=-1, keepdim=True)
        y = y + (rkr * v.reshape(B, T, H, N)).reshape(B, T, H * N)
        out = (y * g) @ self.output.weight.t()
        return out, v_first

    def cmix(self, x, shift_state):
        prev = torch.cat([shift_state[1].unsqueeze(1), x[:, :-1, :]], dim=1)
        shift_state[1] = x[:, -1, :]
        k = lerp(x, prev, self.ffn_x_k)
        k = torch.relu(self.ffn_key(k)) ** 2
        return self.ffn_value(k)

    def forward(self, x, shift_state, v_first, wkv_state, ctx_hidden):
        """x: [B, T, C], ctx_hidden: [B, ctx, d_target_kv]"""
        xx = F.layer_norm(x, (self.C,), self.ln1_w, self.ln1_b)
        xx, v_first = self.tmix(xx, shift_state, v_first, wkv_state, self.layer_idx)
        x = x + xx
        # cross-attn（在 RWKV 层后）
        x = self.cross_attn(x, ctx_hidden)
        xx = F.layer_norm(x, (self.C,), self.ln2_w, self.ln2_b)
        x = x + self.cmix(xx, shift_state)
        return x, v_first


class Rwkv7Draft(nn.Module):
    """RWKV-7 0.4B draft + 每层 cross-attn。"""
    def __init__(self, C, L, H, N, d_target_kv, n_attn_heads=8, ctx_len=8, block=4):
        super().__init__()
        self.C = C
        self.L = L
        self.H = H
        self.N = N
        self.block = block
        self.token_emb = nn.Embedding(VOCAB, C)
        self.ln0_w = nn.Parameter(torch.ones(C))
        self.ln0_b = nn.Parameter(torch.zeros(C))
        self.ln_out_w = nn.Parameter(torch.ones(C))
        self.ln_out_b = nn.Parameter(torch.zeros(C))
        self.head = nn.Linear(C, VOCAB, bias=False)
        self.layers = nn.ModuleList([
            Rwkv7DraftLayer(i, C, H, N, d_target_kv, n_attn_heads, ctx_len)
            for i in range(L)
        ])

    def load_pretrained(self, z):
        """加载 0.4B 预训练权重。"""
        self.token_emb.weight.data.copy_(z["emb.weight"].squeeze())
        self.ln0_w.data.copy_(z["blocks.0.ln0.weight"].squeeze())
        self.ln0_b.data.copy_(z["blocks.0.ln0.bias"].squeeze())
        self.ln_out_w.data.copy_(z["ln_out.weight"].squeeze())
        self.ln_out_b.data.copy_(z["ln_out.bias"].squeeze())
        self.head.weight.data.copy_(z["head.weight"].squeeze())  # [V, C] 一致
        for layer in self.layers:
            layer.load_pretrained(z)
        # emb 预处理：layer_norm with ln0
        with torch.no_grad():
            self.token_emb.weight.data = F.layer_norm(
                self.token_emb.weight.data, (self.C,), self.ln0_w, self.ln0_b
            )

    def zero_state(self, B, device=DEVICE):
        return [
            torch.zeros(self.L, 2, B, self.C, device=device),  # shift state
            torch.zeros(self.L, B, self.H, self.N, self.N, device=device),  # wkv state
        ]

    def forward(self, tokens, ctx_hidden, state=None):
        """tokens: [B, T], ctx_hidden: [B, ctx, d_target_kv]
        返回: logits [B, T, V]
        """
        B, T = tokens.shape
        if state is None:
            state = self.zero_state(B, device=tokens.device)
        x = self.token_emb(tokens)  # [B, T, C]
        v_first = torch.zeros_like(x)
        for i, layer in enumerate(self.layers):
            x, v_first = layer(x, state[0][i], v_first, state[1][i], ctx_hidden)
        x = F.layer_norm(x, (self.C,), self.ln_out_w, self.ln_out_b)
        logits = self.head(x)
        return logits


# ==================================================================
# 评估
# ==================================================================
@torch.no_grad()
def proper_eval(model, target, tokens, hids, n_samples=EVAL_SAMPLES):
    """真实评估：draft 自回归 + 2.9B target 验证"""
    model.eval()
    N, T1 = tokens.shape
    T = T1 - 1
    max_anchor = T - BLOCK
    idx_list = torch.randperm(N)[:n_samples].tolist()

    greedy_accept = [0] * BLOCK
    n_total = 0
    BATCH_EVAL = 8

    for batch_start in range(0, n_samples, BATCH_EVAL):
        batch_idx = idx_list[batch_start:batch_start + BATCH_EVAL]
        B = len(batch_idx)
        anchors = [torch.randint(CTX, max_anchor + 1, (1,)).item() for _ in batch_idx]

        ctx_list = []
        for b in range(B):
            i_seq = batch_idx[b]
            a = anchors[b]
            ctx_l = torch.cat([hids[l][i_seq, a-CTX:a] for l in TARGET_LAYERS], dim=-1)
            ctx_list.append(ctx_l)
        ctx_hidden = torch.stack(ctx_list, dim=0).to(DEVICE).float()

        anchor_tokens = torch.tensor([tokens[batch_idx[b], anchors[b]] for b in range(B)],
                                      device=DEVICE, dtype=torch.long)

        # draft 自回归生成
        draft_tokens = torch.zeros(B, BLOCK, device=DEVICE, dtype=torch.long)
        draft_input = anchor_tokens.clone()  # [B]

        for t in range(BLOCK):
            inp = draft_input.unsqueeze(1) if t == 0 else draft_input_seq
            if t == 0:
                inp = anchor_tokens.unsqueeze(1)  # [B, 1]
            else:
                inp = torch.cat([anchor_tokens.unsqueeze(1), draft_tokens[:, :t]], dim=1)
            draft_logits = model(inp, ctx_hidden)
            next_tok = draft_logits[:, -1, :].argmax(dim=-1)
            draft_tokens[:, t] = next_tok

        # target 验证
        verify_input = torch.cat([anchor_tokens.unsqueeze(1), draft_tokens], dim=1)
        state = target.zero_state(B)
        target_logits, _ = target.forward(verify_input, state, return_hidden_layers=[])
        target_logits_for_draft = target_logits[:, 1:BLOCK+1, :]
        target_argmax = target_logits_for_draft.argmax(dim=-1)

        greedy_match = (draft_tokens == target_argmax)
        for t in range(BLOCK):
            greedy_accept[t] += greedy_match[:, t].sum().item()
        n_total += B

        del state, target_logits
        torch.cuda.empty_cache()

    greedy_rate = [greedy_accept[t] / n_total for t in range(BLOCK)]
    avg = sum(greedy_rate) / BLOCK
    expected_len = 0.0
    cum_p = 1.0
    for t in range(BLOCK):
        cum_p *= greedy_rate[t]
        expected_len += cum_p
    return greedy_rate, avg, expected_len


def lr_schedule(step, warmup, total):
    if step < warmup:
        return step / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def ss_schedule(step, total, ss_max):
    return min(ss_max, step / total * ss_max)


# ==================================================================
# 主训练
# ==================================================================
def main():
    log(f"=" * 60)
    log(f"DSpark → RWKV · 阶段9 [RWKV DRAFT {TAG.upper()}]")
    log(f"  2.9B target + RWKV-7 0.4B draft + 每层 cross-attn")
    log(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"DEVICE={DEVICE}  BLOCK={BLOCK}  N_STEPS={N_STEPS}")
    log(f"MODE={MODE}  SS_MAX={SS_MAX}  EVAL_EVERY={EVAL_EVERY}")
    log(f"DRAFT: L={DRAFT_L} C={DRAFT_C} H={DRAFT_H} N={DRAFT_N}")
    log(f"D_TARGET={D_TARGET}  TARGET_LAYERS={TARGET_LAYERS}")
    log(f"=" * 60)

    # 1. 加载数据
    log(f"\n[加载数据]")
    all_tokens, all_hids = load_data(TRAIN_PATH)
    log(f"  数据: {TRAIN_PATH.name}  tokens={all_tokens.shape}")
    N = all_tokens.shape[0]
    val_tokens = all_tokens[N-VAL_SPLIT:]
    val_hids = {k: v[N-VAL_SPLIT:] for k, v in all_hids.items()}
    train_tokens = all_tokens[:N-VAL_SPLIT]
    train_hids = {k: v[:N-VAL_SPLIT] for k, v in all_hids.items()}
    log(f"  训练: {train_tokens.shape}  验证: {val_tokens.shape}")

    # 2. 加载 target
    log(f"\n[加载 2.9B target]")
    from stage9_target_2p9b import RWKV7Target2p9B
    target = RWKV7Target2p9B()
    log(f"  target GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # 3. 加载 draft
    log(f"\n[加载 RWKV-7 0.4B draft]")
    z = torch.load(DRAFT_WEIGHTS, map_location="cpu", weights_only=True)
    model = Rwkv7Draft(DRAFT_C, DRAFT_L, DRAFT_H, DRAFT_N, D_TARGET * N_TARGET_LAYERS,
                       n_attn_heads=8, ctx_len=CTX, block=BLOCK).to(DEVICE)
    model.load_pretrained(z)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  参数量: {n_params/1e6:.1f}M (可训练: {n_trainable/1e6:.1f}M)")
    log(f"  GPU（target+draft）: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    del z

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # 4. 训练前评估
    log(f"\n[训练前评估]")
    g_rate, g_avg, e_len = proper_eval(model, target, val_tokens, val_hids)
    log(f"  预训练 0.4B（无微调）: greedy_avg={g_avg:.4f}  E[len]={e_len:.4f}/{BLOCK}")
    log(f"  各位置: {[f'{r:.4f}' for r in g_rate]}")

    # 5. 训练循环
    log(f"\n[开始训练]")
    model.train()
    best_eval = g_avg
    best_step = 0

    for step in range(N_STEPS):
        lr = LR * lr_schedule(step, WARMUP, N_STEPS)
        for g in opt.param_groups:
            g['lr'] = lr
        ss_prob = ss_schedule(step, N_STEPS, SS_MAX)

        draft_input, ctx_hidden, target_tokens = sample_batch_cpu(
            train_tokens, train_hids, BS, BLOCK, CTX)

        # teacher forcing: 输入 [anchor, tok1, ..., tok_{BLOCK-1}]，目标 [tok1, ..., tok_BLOCK]
        draft_logits = model(draft_input, ctx_hidden)
        # 对齐：draft_logits[:, t] 预测 target_tokens[:, t]
        loss = F.cross_entropy(draft_logits.reshape(-1, VOCAB), target_tokens.reshape(-1))

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 250 == 0:
            with torch.no_grad():
                acc = (draft_logits.argmax(-1) == target_tokens).float()
                avg_acc = acc.mean().item()
            log(f"  step {step+1:5d}  lr={lr:.2e}  loss={loss.item():.4f}  train_acc={avg_acc:.3f}")

        if (step + 1) % EVAL_EVERY == 0:
            model.eval()
            g_rate, g_avg, e_len = proper_eval(model, target, val_tokens, val_hids)
            log(f"\n  [评估 step {step+1}] greedy_avg={g_avg:.4f}  E[len]={e_len:.4f}/{BLOCK}")
            log(f"  各位置: {[f'{r:.4f}' for r in g_rate]}")
            if g_avg > best_eval:
                best_eval = g_avg
                best_step = step + 1
                torch.save({
                    "model_state": model.state_dict(),
                    "step": step + 1,
                    "greedy_avg": g_avg,
                    "expected_len": e_len,
                    "config": {"block": BLOCK, "ss_max": SS_MAX, "mode": MODE, "n_steps": N_STEPS,
                               "draft_C": DRAFT_C, "draft_L": DRAFT_L, "draft_H": DRAFT_H, "draft_N": DRAFT_N},
                }, OUT_WEIGHTS)
                log(f"  新最优 {g_avg:.4f}，保存到 {OUT_WEIGHTS.name}")
            else:
                log(f"  (未超越最优 {best_eval:.4f} @ step {best_step})")
            model.train()

    # 6. 最终评估
    log(f"\n{'='*60}")
    log(f"[训练结束] [RWKV DRAFT {TAG.upper()}]")
    log(f"  最优 greedy_avg: {best_eval:.4f} @ step {best_step}")

    if OUT_WEIGHTS.exists():
        ckpt = torch.load(OUT_WEIGHTS, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        log(f"  已加载最优权重 (step {ckpt['step']})")
    model.eval()
    g_rate, g_avg, e_len = proper_eval(model, target, val_tokens, val_hids, n_samples=128)
    log(f"\n  最终评估 (128 样本):")
    log(f"  greedy_avg: {g_avg:.4f}")
    log(f"  E[len]:     {e_len:.4f}/{BLOCK}")
    log(f"  各位置:     {[f'{r:.4f}' for r in g_rate]}")

    log(f"\n日志: {log_path}")


if __name__ == "__main__":
    main()
