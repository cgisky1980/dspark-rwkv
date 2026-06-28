"""DSpark → RWKV · 阶段9：2.9B target + 400M draft 训练（baseline）

基于 stage8_compare.py 改造：
- target: 2.9B RWKV-7 (L=32, C=2560)
- draft: 400M (d_draft=1024, 8 层, rank=512)
- 数据: stage9_train.pt (32768 条，LAMBADA + WikiText)
- 5 层 hidden: [0, 8, 16, 24, 31]
- 数据保留在 CPU（27GB），sample 时搬 batch 到 GPU
- 评估用 2.9B target forward 验证

用法：
  uv run python -u stage9_train_baseline.py
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
log_path = Path(__file__).parent / f"stage9_{TAG}_400m.log"
logger = logging.getLogger(f"stage9_{TAG}")
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
TRAIN_PATH = Path(__file__).parent / "data" / "stage9_train.pt"
OUT_WEIGHTS = Path(__file__).parent / "weights" / f"stage9_{TAG}_400m.pth"

VOCAB = 65536
D_TARGET = 2560          # 2.9B 的 C
N_TARGET_LAYERS = 5
TARGET_LAYERS = [0, 8, 16, 24, 31]
D_DRAFT = 1024           # draft hidden size（400M 版）
N_DRAFT_LAYERS = 8       # draft 层数（400M 版）
RANK = 512               # RWKV head rank（400M 版）
N_HEADS = 16             # 1024/16=64 per head
CTX = 8                  # cross-attn context length
BLOCK = 4                # draft block size
LR = 1e-4
N_STEPS = args.steps
WARMUP = 500
BS = 64
SQRT_E = math.sqrt(math.e)

EVAL_EVERY = 1000
EVAL_SAMPLES = 64
VAL_SPLIT = 1024         # 从训练集尾部取 1024 条做验证

# ==================================================================
# 数据加载（保留 CPU，sample 时搬 GPU）
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
    ctx_hidden = torch.cat(ctx_list, dim=-1).to(DEVICE).float()  # [bs, ctx, D_TARGET*5] FP16->FP32

    anchor_token = tokens[idx, anc].to(DEVICE)
    block_tokens = torch.stack([tokens[idx, anc + 1 + k] for k in range(block)], dim=1).to(DEVICE)
    prev = torch.zeros_like(block_tokens)
    prev[:, 0] = anchor_token
    prev[:, 1:] = block_tokens[:, :-1]
    return anchor_token, ctx_hidden, prev, block_tokens


# ==================================================================
# 模型定义
# ==================================================================
class CrossAttnLayer(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln3 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))

    def forward(self, x, ctx_kv):
        h = self.ln1(x)
        a, _ = self.self_attn(h, h, h)
        x = x + a
        h = self.ln2(x)
        a, _ = self.cross_attn(h, ctx_kv, ctx_kv)
        x = x + a
        h = self.ln3(x)
        x = x + self.ffn(h)
        return x


class ParallelTrunk(nn.Module):
    def __init__(self, vocab, d_draft, n_layers, n_heads, block_size, d_target, n_target_layers, ctx):
        super().__init__()
        self.block_size = block_size
        self.ctx = ctx
        self.token_emb = nn.Embedding(vocab, d_draft)
        self.mask_emb = nn.Parameter(torch.randn(1, 1, d_draft) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, block_size, d_draft) * 0.02)
        self.ctx_pos_emb = nn.Parameter(torch.randn(1, ctx, d_draft) * 0.02)
        self.target_fc = nn.Linear(d_target * n_target_layers, d_draft)
        self.target_ln = nn.LayerNorm(d_draft)
        self.layers = nn.ModuleList([
            CrossAttnLayer(d_draft, n_heads) for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_draft)
        self.lm_head = nn.Linear(d_draft, vocab, bias=False)

    def forward(self, anchor_token, ctx_hidden):
        B = anchor_token.shape[0]
        anchor_emb = self.token_emb(anchor_token).unsqueeze(1)
        mask = self.mask_emb.expand(B, self.block_size - 1, -1)
        x = torch.cat([anchor_emb, mask], dim=1)
        x = x + self.pos_emb
        ctx = self.target_fc(ctx_hidden)
        ctx = self.target_ln(ctx)
        ctx = ctx + self.ctx_pos_emb
        for layer in self.layers:
            x = layer(x, ctx)
        x = self.final_ln(x)
        return self.lm_head(x), x


class Rwkv7Head(nn.Module):
    def __init__(self, vocab, d_draft, rank=256):
        super().__init__()
        self.rank = rank
        self.token_emb = nn.Embedding(vocab, d_draft)
        self.x_r = nn.Parameter(torch.zeros(d_draft))
        self.x_w = nn.Parameter(torch.zeros(d_draft))
        self.x_k = nn.Parameter(torch.zeros(d_draft))
        self.x_v = nn.Parameter(torch.zeros(d_draft))
        self.x_a = nn.Parameter(torch.zeros(d_draft))
        self.x_g = nn.Parameter(torch.zeros(d_draft))
        self.ln = nn.LayerNorm(d_draft)
        self.r_proj = nn.Linear(d_draft, rank, bias=False)
        self.k_proj = nn.Linear(d_draft, rank, bias=False)
        self.v_proj = nn.Linear(d_draft, rank, bias=False)
        self.w_proj = nn.Linear(d_draft, rank, bias=False)
        self.a_proj = nn.Linear(d_draft, rank, bias=False)
        self.w1 = nn.Linear(d_draft, rank, bias=False)
        self.w2 = nn.Linear(rank, rank, bias=False)
        self.a0 = nn.Parameter(torch.zeros(rank))
        self.w0 = nn.Parameter(torch.zeros(rank))
        self.k_k = nn.Parameter(torch.ones(rank) * 0.1)
        self.k_a = nn.Parameter(torch.zeros(rank))
        self.r_k = nn.Parameter(torch.ones(rank) * 0.1)
        self.g1 = nn.Linear(d_draft, rank, bias=False)
        self.g2 = nn.Linear(rank, rank, bias=False)
        self.w_out = nn.Linear(rank, vocab, bias=False)

    def forward_block(self, base_logits, prev_tokens, hidden):
        """teacher forcing forward"""
        B, T, V = base_logits.shape
        prev_emb = self.token_emb(prev_tokens)
        x = self.ln(hidden)
        xr = x + self.x_r * (prev_emb - x)
        xw = x + self.x_w * (prev_emb - x)
        xk = x + self.x_k * (prev_emb - x)
        xv = x + self.x_v * (prev_emb - x)
        xa = x + self.x_a * (prev_emb - x)
        xg = x + self.x_g * (prev_emb - x)
        r = self.r_proj(xr); k = self.k_proj(xk); v = self.v_proj(xv)
        w_raw = self.w_proj(xw) + self.w2(torch.tanh(self.w1(xw)))
        a = torch.sigmoid(self.a0 + self.a_proj(xa))
        g = torch.sigmoid(self.g1(xg)) @ self.g2.weight
        S = torch.zeros(B, self.rank, self.rank, device=base_logits.device)
        biases = []
        eps = 1e-12
        for t in range(T):
            rt, kt, vt = r[:, t], k[:, t], v[:, t]
            at = a[:, t]
            w = torch.exp(-torch.sigmoid(self.w0 + w_raw[:, t]) / SQRT_E)
            kk = kt * self.k_k
            kk = kk / (kk.norm(dim=-1, keepdim=True) + eps)
            k_mod = kt + self.k_a * (kt * at - kt)
            S_kk = (S * kk.unsqueeze(1)).sum(dim=2)
            S = S * w.unsqueeze(1)
            S = S + S_kk.unsqueeze(2) * (-kk * at).unsqueeze(1)
            S = S + vt.unsqueeze(2) * k_mod.unsqueeze(1)
            y = (S * rt.unsqueeze(1)).sum(dim=2)
            rkr_sum = (rt * k_mod * self.r_k).sum(dim=-1, keepdim=True)
            y = y + rkr_sum * vt
            biases.append(self.w_out(y * g[:, t]))
        return torch.stack(biases, dim=1)

    def forward_block_ss(self, base_logits, gt_prev_tokens, hidden, ss_prob=0.0):
        """scheduled sampling forward"""
        B, T, V = base_logits.shape
        x = self.ln(hidden)
        S = torch.zeros(B, self.rank, self.rank, device=base_logits.device)
        biases = []
        eps = 1e-12
        prev_token = gt_prev_tokens[:, 0]
        for t in range(T):
            prev_emb = self.token_emb(prev_token)
            xt = x[:, t]
            xr = xt + self.x_r * (prev_emb - xt)
            xw = xt + self.x_w * (prev_emb - xt)
            xk = xt + self.x_k * (prev_emb - xt)
            xv = xt + self.x_v * (prev_emb - xt)
            xa = xt + self.x_a * (prev_emb - xt)
            xg = xt + self.x_g * (prev_emb - xt)
            r = self.r_proj(xr); k = self.k_proj(xk); v = self.v_proj(xv)
            w_raw = self.w_proj(xw) + self.w2(torch.tanh(self.w1(xw)))
            a = torch.sigmoid(self.a0 + self.a_proj(xa))
            g = torch.sigmoid(self.g1(xg)) @ self.g2.weight
            w = torch.exp(-torch.sigmoid(self.w0 + w_raw) / SQRT_E)
            kk = k * self.k_k
            kk = kk / (kk.norm(dim=-1, keepdim=True) + eps)
            k_mod = k + self.k_a * (k * a - k)
            S_kk = (S * kk.unsqueeze(1)).sum(dim=2)
            S = S * w.unsqueeze(1)
            S = S + S_kk.unsqueeze(2) * (-kk * a).unsqueeze(1)
            S = S + v.unsqueeze(2) * k_mod.unsqueeze(1)
            y = (S * r.unsqueeze(1)).sum(dim=2)
            rkr_sum = (r * k_mod * self.r_k).sum(dim=-1, keepdim=True)
            y = y + rkr_sum * v
            bias_t = self.w_out(y * g)
            biases.append(bias_t)
            if t + 1 < T:
                draft_tok = (base_logits[:, t] + bias_t).argmax(dim=-1)
                if ss_prob > 0:
                    mask = torch.rand(B, device=base_logits.device) < ss_prob
                    prev_token = torch.where(mask, draft_tok, gt_prev_tokens[:, t + 1])
                else:
                    prev_token = gt_prev_tokens[:, t + 1]
        return torch.stack(biases, dim=1)


class DSparkDraft(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.trunk = ParallelTrunk(VOCAB, D_DRAFT, N_DRAFT_LAYERS, N_HEADS, block_size, D_TARGET, N_TARGET_LAYERS, CTX)
        self.head = Rwkv7Head(VOCAB, D_DRAFT, rank=RANK)
        self.conf_head = nn.Linear(D_DRAFT, 1)

    def forward(self, anchor_token, ctx_hidden, prev_tokens):
        base_logits, hidden = self.trunk(anchor_token, ctx_hidden)
        bias = self.head.forward_block(base_logits, prev_tokens, hidden)
        return base_logits + bias, torch.sigmoid(self.conf_head(hidden)).squeeze(-1)

    def forward_ss(self, anchor_token, ctx_hidden, gt_prev_tokens, ss_prob=0.0):
        base_logits, hidden = self.trunk(anchor_token, ctx_hidden)
        bias = self.head.forward_block_ss(base_logits, gt_prev_tokens, hidden, ss_prob)
        return base_logits + bias, torch.sigmoid(self.conf_head(hidden)).squeeze(-1)


# ==================================================================
# 评估
# ==================================================================
@torch.no_grad()
def proper_eval(model, target, tokens, hids, n_samples=EVAL_SAMPLES):
    """真实评估：draft 自回归 + 2.9B target 验证"""
    N, T1 = tokens.shape
    T = T1 - 1
    max_anchor = T - BLOCK
    idx_list = torch.randperm(N)[:n_samples].tolist()

    greedy_accept = [0] * BLOCK
    n_total = 0
    BATCH_EVAL = 8  # 2.9B target forward 显存大，小 batch

    for batch_start in range(0, n_samples, BATCH_EVAL):
        batch_idx = idx_list[batch_start:batch_start + BATCH_EVAL]
        B = len(batch_idx)
        anchors = [torch.randint(CTX, max_anchor + 1, (1,)).item() for _ in batch_idx]

        # 从 CPU 数据取 ctx_hidden
        ctx_list = []
        for b in range(B):
            i_seq = batch_idx[b]
            a = anchors[b]
            ctx_l = torch.cat([hids[l][i_seq, a-CTX:a] for l in TARGET_LAYERS], dim=-1)
            ctx_list.append(ctx_l)
        ctx_hidden = torch.stack(ctx_list, dim=0).to(DEVICE).float()  # FP16->FP32

        anchor_tokens = torch.tensor([tokens[batch_idx[b], anchors[b]] for b in range(B)],
                                      device=DEVICE, dtype=torch.long)

        # draft 自回归生成
        draft_tokens = torch.zeros(B, BLOCK, device=DEVICE, dtype=torch.long)
        prev_tokens = torch.zeros(B, BLOCK, device=DEVICE, dtype=torch.long)
        prev_tokens[:, 0] = anchor_tokens

        for t in range(BLOCK):
            draft_logits, _ = model.forward_ss(anchor_tokens, ctx_hidden, prev_tokens, ss_prob=1.0)
            next_tok = draft_logits[:, t, :].argmax(dim=-1)
            draft_tokens[:, t] = next_tok
            if t + 1 < BLOCK:
                prev_tokens[:, t + 1] = next_tok

        # target 验证（2.9B forward）
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
    log(f"DSpark → RWKV · 阶段9 [{TAG.upper()}] 2.9B target + 400M draft")
    log(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"DEVICE={DEVICE}  BLOCK={BLOCK}  N_STEPS={N_STEPS}")
    log(f"MODE={MODE}  SS_MAX={SS_MAX}  EVAL_EVERY={EVAL_EVERY}")
    log(f"D_DRAFT={D_DRAFT}  N_DRAFT_LAYERS={N_DRAFT_LAYERS}  RANK={RANK}")
    log(f"D_TARGET={D_TARGET}  TARGET_LAYERS={TARGET_LAYERS}")
    log(f"=" * 60)

    # 1. 加载数据（CPU）
    log(f"\n[加载数据]")
    all_tokens, all_hids = load_data(TRAIN_PATH)
    log(f"  数据: {TRAIN_PATH.name}  tokens={all_tokens.shape}")

    # 分割训练/验证
    N = all_tokens.shape[0]
    val_tokens = all_tokens[N-VAL_SPLIT:]
    val_hids = {k: v[N-VAL_SPLIT:] for k, v in all_hids.items()}
    train_tokens = all_tokens[:N-VAL_SPLIT]
    train_hids = all_hids
    # 只保留训练部分
    train_hids = {k: v[:N-VAL_SPLIT] for k, v in all_hids.items()}
    log(f"  训练: {train_tokens.shape}  验证: {val_tokens.shape}")

    # 2. 加载 2.9B target
    log(f"\n[加载 2.9B target]")
    from stage9_target_2p9b import RWKV7Target2p9B
    target = RWKV7Target2p9B()
    log(f"  target GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # 3. 初始化 draft 模型
    log(f"\n[初始化 draft 模型]")
    model = DSparkDraft(BLOCK).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  参数量: {n_params/1e6:.1f}M")
    log(f"  GPU（target+draft）: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # 4. 训练前评估
    log(f"\n[训练前评估]")
    model.eval()
    g_rate, g_avg, e_len = proper_eval(model, target, val_tokens, val_hids)
    log(f"  随机初始化: greedy_avg={g_avg:.4f}  E[len]={e_len:.4f}/{BLOCK}")

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

        anc, ch, prev, tgt = sample_batch_cpu(train_tokens, train_hids, BS, BLOCK, CTX)

        if MODE == "baseline":
            draft_logits, conf = model(anc, ch, prev)
        else:
            draft_logits, conf = model.forward_ss(anc, ch, prev, ss_prob=ss_prob)

        loss_ce = F.cross_entropy(draft_logits.reshape(-1, VOCAB), tgt.reshape(-1))
        with torch.no_grad():
            correct = (draft_logits.argmax(-1) == tgt).float()
        cum_correct = (correct.cumsum(dim=1) == torch.arange(1, BLOCK+1, device=correct.device).unsqueeze(0)).float()
        loss_conf = F.binary_cross_entropy(conf, cum_correct)
        loss = loss_ce + 0.5 * loss_conf

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 250 == 0:
            with torch.no_grad():
                acc = (draft_logits.argmax(-1) == tgt).float()
                avg_acc = acc.mean().item()
            log(f"  step {step+1:5d}  lr={lr:.2e}  ss={ss_prob:.2f}  loss={loss.item():.4f} (ce={loss_ce.item():.3f})  train_acc={avg_acc:.3f}")

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
                               "d_draft": D_DRAFT, "n_layers": N_DRAFT_LAYERS, "rank": RANK},
                }, OUT_WEIGHTS)
                log(f"  新最优 {g_avg:.4f}，保存到 {OUT_WEIGHTS.name}")
            else:
                log(f"  (未超越最优 {best_eval:.4f} @ step {best_step})")
            model.train()

    # 6. 最终评估
    log(f"\n{'='*60}")
    log(f"[训练结束] [{TAG.upper()}]")
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
