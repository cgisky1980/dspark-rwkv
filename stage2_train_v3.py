"""DSpark → RWKV 复现 · 阶段2v3 综合优化

优化项：
A. block_size 扫描（4/6/8）
B. target hidden 5 层（layer 0/3/6/9/11）
C. 训练 5000 步
保留 v2 的：cross-attn 8 位置 context + 置信度 BCE + warmup+cosine

只跑 RWKV-7 顺序头（v2 已证明优于 GRU），最后与 v2 最优结果对比。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path(__file__).parent / "data"

VOCAB = 65536
D_TARGET = 2560  # 2.9B 模型 C=2560
N_TARGET_LAYERS = 5  # 5 层
TARGET_LAYERS = [0, 3, 6, 9, 11]
D_DRAFT = 1280      # 500M draft
N_DRAFT_LAYERS = 10  # 10 层 Transformer
N_HEADS = 16         # 配合 D=1280
CTX = 8
LR = 1e-4
N_STEPS = 20000     # 2 万步
WARMUP = 1000       # warmup 比例保持
BS = 16             # 500M 模型显存较大，BS 降到 16
SQRT_E = math.sqrt(math.e)


def load_chunk(chunk_idx):
    """加载单个 chunk 文件（fp16，1W 条）。

    chunk 划分:
      0-7: train (8 个 chunk = 8W 条)
      8:   val   (1W 条)
      9:   test  (1W 条)
    """
    path = DATA_DIR / f"chunk_{chunk_idx:02d}.pt"
    d = torch.load(path, map_location="cpu", weights_only=True)
    tokens = d["tokens"]
    hids = {l: d[f"hidden_{l}"].half() for l in TARGET_LAYERS}
    return tokens, hids


TRAIN_CHUNKS = list(range(8))  # 0-7
VAL_CHUNK = 8
TEST_CHUNK = 9


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
    def __init__(self, vocab, d_draft, n_layers, n_heads, block_size, d_target, ctx):
        super().__init__()
        self.block_size = block_size
        self.ctx = ctx
        self.token_emb = nn.Embedding(vocab, d_draft)
        self.mask_emb = nn.Parameter(torch.randn(1, 1, d_draft) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, block_size, d_draft) * 0.02)
        self.ctx_pos_emb = nn.Parameter(torch.randn(1, ctx, d_draft) * 0.02)
        self.target_fc = nn.Linear(d_target * N_TARGET_LAYERS, d_draft)
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
    def __init__(self, vocab, d_draft, rank=128):
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
        S = torch.zeros(B, self.rank, self.rank, device=base_logits.device, dtype=base_logits.dtype)
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


class DSparkDraft(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.trunk = ParallelTrunk(VOCAB, D_DRAFT, N_DRAFT_LAYERS, N_HEADS, block_size, D_TARGET, CTX)
        self.head = Rwkv7Head(VOCAB, D_DRAFT, rank=512)  # 500M 配置
        self.conf_head = nn.Linear(D_DRAFT, 1)

    def forward(self, anchor_token, ctx_hidden, prev_tokens):
        base_logits, hidden = self.trunk(anchor_token, ctx_hidden)
        bias = self.head.forward_block(base_logits, prev_tokens, hidden)
        # 返回 conf logits（不用 sigmoid，用 BCEWithLogits 算 loss）
        return base_logits + bias, self.conf_head(hidden).squeeze(-1)


def sample_batch(tokens, hids_dict, bs, block, ctx):
    N, T1 = tokens.shape
    T = T1 - 1
    max_anchor = T - block
    idx = torch.randint(0, N, (bs,))
    anc = torch.randint(ctx, max_anchor + 1, (bs,))
    ctx_list = []
    for l in TARGET_LAYERS:
        ctx_l = torch.stack([hids_dict[l][idx[b], anc[b]-ctx:anc[b]] for b in range(bs)], dim=0)
        ctx_list.append(ctx_l)
    ctx_hidden = torch.cat(ctx_list, dim=-1)
    anchor_token = tokens[idx, anc]
    block_tokens = torch.stack([tokens[idx, anc + 1 + k] for k in range(block)], dim=1)
    prev = torch.zeros_like(block_tokens)
    prev[:, 0] = anchor_token
    prev[:, 1:] = block_tokens[:, :-1]
    return anchor_token, ctx_hidden, prev, block_tokens


def lr_schedule(step, warmup, total):
    if step < warmup:
        return step / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def run(block, n_steps=N_STEPS):
    print(f"\n{'='*60}\n=== RWKV-7 (block={block}, 5layer, steps={n_steps}) ===\n{'='*60}")
    # 验证用独立的 val chunk（修复数据泄露）
    val_tokens, val_hids = load_chunk(VAL_CHUNK)
    val_tokens = val_tokens.to(DEVICE)
    val_hids = {k: v.to(DEVICE) for k, v in val_hids.items()}

    model = DSparkDraft(block).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params/1e6:.1f}M  train_chunks={TRAIN_CHUNKS} val_chunk={VAL_CHUNK} (val={val_tokens.shape[0]})")
    # 用 autocast 混合精度（forward fp16，loss/grad fp32）
    scaler = torch.amp.GradScaler()

    # chunk 轮换训练：每个 chunk 跑 chunk_steps 步
    chunk_steps = n_steps // len(TRAIN_CHUNKS)
    print(f"每 chunk {chunk_steps} 步，共 {len(TRAIN_CHUNKS)} chunk")
    tokens = None
    hids_dict = None
    cur_chunk = -1

    for step in range(n_steps):
        # 切换 chunk
        target_chunk = step // chunk_steps
        if target_chunk >= len(TRAIN_CHUNKS):
            target_chunk = len(TRAIN_CHUNKS) - 1
        if target_chunk != cur_chunk:
            cur_chunk = target_chunk
            ci = TRAIN_CHUNKS[cur_chunk]
            print(f"  [step {step}] 加载 train chunk {ci}...")
            tokens, hids_dict = load_chunk(ci)
            tokens = tokens.to(DEVICE)
            hids_dict = {k: v.to(DEVICE) for k, v in hids_dict.items()}
            print(f"    chunk {ci}: {tokens.shape[0]} 条")

        lr = LR * lr_schedule(step, WARMUP, n_steps)
        for g in opt.param_groups:
            g['lr'] = lr
        anc, ch, prev, tgt = sample_batch(tokens, hids_dict, BS, block, CTX)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            draft_logits, conf = model(anc, ch, prev)
            loss_ce = F.cross_entropy(draft_logits.reshape(-1, VOCAB), tgt.reshape(-1))
            with torch.no_grad():
                correct = (draft_logits.argmax(-1) == tgt).float()
            cum_correct = (correct.cumsum(dim=1) == torch.arange(1, block+1, device=correct.device).unsqueeze(0)).float()
            loss_conf = F.binary_cross_entropy_with_logits(conf.float(), cum_correct)
            loss = loss_ce + 0.5 * loss_conf
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if (step + 1) % 50 == 0:
            with torch.no_grad():
                acc = (draft_logits.argmax(-1) == tgt).float()
                pos_acc = [acc[:, t].mean().item() for t in range(block)]
                avg_acc = acc.mean().item()
            print(f"  step {step+1:4d}  loss={loss.item():.4f} (ce={loss_ce.item():.3f})  acc={avg_acc:.3f}  pos={[f'{a:.2f}' for a in pos_acc]}  chunk={cur_chunk}")

    # 用独立的 val 集评估真实接受率
    model.eval()
    pos_acc = [0.0] * block
    pos_cnt = [0] * block
    with torch.no_grad():
        N_eval = val_tokens.shape[0]
        for i in range(0, N_eval, BS):
            n = min(BS, N_eval - i)
            anc, ch, prev, tgt = sample_batch(val_tokens, val_hids, n, block, CTX)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                dl, _ = model(anc, ch, prev)
            acc = (dl.argmax(-1) == tgt).float()
            for t in range(block):
                pos_acc[t] += acc[:, t].sum().item()
                pos_cnt[t] += acc[:, t].numel()
    pos_rate = [pos_acc[t] / max(pos_cnt[t], 1) for t in range(block)]
    avg = sum(pos_rate) / block
    print(f"\n  [VAL] 最终位置接受率: {[f'{r:.3f}' for r in pos_rate]}")
    print(f"  [VAL] 平均接受率: {avg:.4f}")
    return pos_rate, avg


def main():
    print(f"DEVICE={DEVICE} 5层hidden block扫描 5000步")
    results = {}
    for block in [4, 6, 8]:
        torch.manual_seed(42)
        pos_rate, avg = run(block)
        results[block] = (pos_rate, avg)
    print(f"\n{'='*60}\nblock_size 扫描汇总\n{'='*60}")
    print(f"{'block':<8} {'平均':<8} 各位置")
    for block, (pos_rate, avg) in results.items():
        print(f"{block:<8} {avg:.4f}   {[f'{r:.3f}' for r in pos_rate]}")


if __name__ == "__main__":
    main()
