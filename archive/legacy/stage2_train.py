"""DSpark → RWKV 复现 · 阶段2：真实 RWKV-7 target + DSpark 双轨架构

架构：
- 并行主干：2 层 Transformer，cross-attn 到 target 多层 hidden（layer 0/6/11）
  - 输入：anchor token emb + block_size-1 个 mask token
  - 输出：base_logits + hidden_states（给顺序头）
- 顺序头：GRU / RWKV-7 DPLR（带 token shift + LayerNorm）
  - 输出：bias 加到 base_logits
- 置信度头：预测接受率

训练数据：预计算的 stage2_train.pt（512 序列 × 32 位置 × 3 层 hidden）
block_size=4，anchor 位置随机采样
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = Path(__file__).parent / "data" / "stage2_train.pt"

# 模型配置
VOCAB = 65536       # RWKV vocab
D_TARGET = 768      # target hidden 维度
N_TARGET_LAYERS = 3 # 采 3 层 target hidden
D_DRAFT = 256       # draft hidden 维度
N_DRAFT_LAYERS = 2  # 并行主干层数
BLOCK = 4           # block_size
N_HEADS = 8         # attention heads
LR = 1e-4
N_STEPS = 1500
BS = 64             # 训练 batch
SQRT_E = math.sqrt(math.e)


def load_data():
    d = torch.load(DATA_PATH, map_location="cpu", weights_only=True)
    tokens = d["tokens"]          # [N, T+1]
    hids = torch.cat([d["hidden_0"], d["hidden_6"], d["hidden_11"]], dim=-1)  # [N, T, 3*768]
    return tokens, hids


# ---------- 并行主干 ----------
class CrossAttnLayer(nn.Module):
    """self-attn + cross-attn + ffn"""
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln3 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d)
        )

    def forward(self, x, target_hidden):
        # x: [B, T, d], target_hidden: [B, 1, d]（anchor 位置）
        # self-attn（block 内位置间）
        h = self.ln1(x)
        a, _ = self.self_attn(h, h, h)
        x = x + a
        # cross-attn 到 target hidden
        h = self.ln2(x)
        a, _ = self.cross_attn(h, target_hidden, target_hidden)
        x = x + a
        # ffn
        h = self.ln3(x)
        x = x + self.ffn(h)
        return x


class ParallelTrunk(nn.Module):
    """并行主干：noise embedding + N 层 CrossAttnLayer。
    输入 anchor token，输出 block 内每个位置的 hidden + base_logits。
    """
    def __init__(self, vocab, d_draft, n_layers, n_heads, block_size, d_target):
        super().__init__()
        self.block_size = block_size
        self.d_draft = d_draft
        # token embedding（anchor 用）
        self.token_emb = nn.Embedding(vocab, d_draft)
        # mask token（block 内非首位用）
        self.mask_emb = nn.Parameter(torch.randn(1, 1, d_draft) * 0.02)
        # 位置编码
        self.pos_emb = nn.Parameter(torch.randn(1, block_size, d_draft) * 0.02)
        # target hidden 融合：3*768 -> d_draft
        self.target_fc = nn.Linear(d_target * N_TARGET_LAYERS, d_draft)
        self.target_ln = nn.LayerNorm(d_draft)
        # N 层 cross-attn
        self.layers = nn.ModuleList([
            CrossAttnLayer(d_draft, n_heads) for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_draft)
        # base lm_head
        self.lm_head = nn.Linear(d_draft, vocab, bias=False)

    def forward(self, anchor_token, target_hidden):
        """anchor_token: [B], target_hidden: [B, 3*768]
        -> base_logits [B, block, V], hidden [B, block, d]
        """
        B = anchor_token.shape[0]
        # noise embedding: 位置 0 = anchor emb, 位置 1+ = mask emb
        anchor_emb = self.token_emb(anchor_token).unsqueeze(1)  # [B,1,d]
        mask = self.mask_emb.expand(B, self.block_size - 1, -1)  # [B, block-1, d]
        x = torch.cat([anchor_emb, mask], dim=1)  # [B, block, d]
        x = x + self.pos_emb
        # target hidden 融合
        th = self.target_fc(target_hidden).unsqueeze(1)  # [B, 1, d]
        th = self.target_ln(th)
        # N 层 cross-attn
        for layer in self.layers:
            x = layer(x, th)
        x = self.final_ln(x)
        base_logits = self.lm_head(x)  # [B, block, V]
        return base_logits, x


# ---------- 顺序头 ----------
class GruHead(nn.Module):
    """DSpark GRU 顺序头"""
    def __init__(self, vocab, d_draft, rank=128):
        super().__init__()
        self.rank = rank
        self.token_emb = nn.Embedding(vocab, rank)
        self.joint = nn.Linear(2 * rank + d_draft, 3 * rank)
        self.w_out = nn.Linear(rank, vocab, bias=False)

    def forward_block(self, base_logits, prev_tokens, hidden):
        B, T, V = base_logits.shape
        prev_emb = self.token_emb(prev_tokens)
        s = torch.zeros(B, self.rank, device=base_logits.device)
        biases = []
        for t in range(T):
            z = torch.cat([s, prev_emb[:, t], hidden[:, t]], dim=-1)
            g_raw, c_raw, o_raw = self.joint(z).chunk(3, dim=-1)
            g = torch.sigmoid(g_raw)
            s = g * s + (1 - g) * torch.tanh(c_raw)
            biases.append(self.w_out(torch.tanh(o_raw)))
        return torch.stack(biases, dim=1)


class Rwkv7Head(nn.Module):
    """RWKV-7 DPLR 顺序头（完整版：token shift + LayerNorm + 正确公式）"""
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


# ---------- Draft 完整模型 ----------
class DSparkDraft(nn.Module):
    def __init__(self, head_cls):
        super().__init__()
        self.trunk = ParallelTrunk(VOCAB, D_DRAFT, N_DRAFT_LAYERS, N_HEADS, BLOCK, D_TARGET)
        self.head = head_cls(VOCAB, D_DRAFT)
        # 置信度头
        self.conf_head = nn.Linear(D_DRAFT, 1)

    def forward(self, anchor_token, target_hidden, prev_tokens):
        base_logits, hidden = self.trunk(anchor_token, target_hidden)
        bias = self.head.forward_block(base_logits, prev_tokens, hidden)
        draft_logits = base_logits + bias
        # 置信度
        conf = torch.sigmoid(self.conf_head(hidden))  # [B, block, 1]
        return draft_logits, conf.squeeze(-1)


# ---------- 训练 ----------
def sample_batch(tokens, hids, bs):
    """采 anchor 位置。tokens: [N, T+1], hids: [N, T, 3*768]
    anchor 位置 i 的 block = tokens[i, anchor+1 : anchor+1+BLOCK]（下一 token 起的 BLOCK 个）
    target_hidden = hids[i, anchor]（anchor 位置的 hidden）
    prev_tokens: teacher forcing，位置 0 = anchor token，位置 k = block[k-1]
    """
    N, T1 = tokens.shape
    max_anchor = T1 - BLOCK - 1  # anchor 从 0 到 max_anchor
    idx = torch.randint(0, N, (bs,))
    anc = torch.randint(0, max_anchor + 1, (bs,))
    # target hidden
    target_hidden = hids[idx, anc]  # [bs, 3*768]
    # anchor token
    anchor_token = tokens[idx, anc]  # [bs]
    # block 内 target tokens（下一 token 起 BLOCK 个）
    block_tokens = torch.stack([tokens[idx, anc + 1 + k] for k in range(BLOCK)], dim=1)  # [bs, BLOCK]
    # prev tokens（teacher forcing）
    prev = torch.zeros_like(block_tokens)
    prev[:, 0] = anchor_token
    prev[:, 1:] = block_tokens[:, :-1]
    return anchor_token, target_hidden, prev, block_tokens


def ce_loss(draft_logits, target_tokens):
    return F.cross_entropy(draft_logits.reshape(-1, VOCAB), target_tokens.reshape(-1))


def run(name, head_cls):
    print(f"\n{'='*60}\n=== {name} ===\n{'='*60}")
    tokens, hids = load_data()
    tokens = tokens.to(DEVICE)
    hids = hids.to(DEVICE)
    print(f"数据: tokens={tokens.shape}, hids={hids.shape}")

    model = DSparkDraft(head_cls).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params/1e6:.1f}M")

    for step in range(N_STEPS):
        anc, th, prev, tgt = sample_batch(tokens, hids, BS)
        draft_logits, conf = model(anc, th, prev)
        loss = ce_loss(draft_logits, tgt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0:
            with torch.no_grad():
                acc = (draft_logits.argmax(-1) == tgt).float()
                pos_acc = [acc[:, t].mean().item() for t in range(BLOCK)]
                avg_acc = acc.mean().item()
            print(f"  step {step+1:4d}  loss={loss.item():.4f}  acc={avg_acc:.3f}  pos={[f'{a:.2f}' for a in pos_acc]}")

    # 评估
    model.eval()
    pos_acc = [0.0] * BLOCK
    pos_cnt = [0] * BLOCK
    with torch.no_grad():
        N_eval = 512
        for i in range(0, N_eval, BS):
            anc, th, prev, tgt = sample_batch(tokens, hids, min(BS, N_eval - i))
            dl, _ = model(anc, th, prev)
            acc = (dl.argmax(-1) == tgt).float()
            for t in range(BLOCK):
                pos_acc[t] += acc[:, t].sum().item()
                pos_cnt[t] += acc[:, t].numel()
    pos_rate = [pos_acc[t] / max(pos_cnt[t], 1) for t in range(BLOCK)]
    avg = sum(pos_rate) / BLOCK
    print(f"\n  最终位置接受率: {[f'{r:.3f}' for r in pos_rate]}")
    print(f"  平均接受率: {avg:.4f}")
    return pos_rate, avg


def main():
    print(f"DEVICE={DEVICE} VOCAB={VOCAB} D_DRAFT={D_DRAFT} BLOCK={BLOCK}")
    print(f"target: RWKV-7 0.1B (L=12 C=768 H=12 N=64)")
    results = {}
    for name, cls in [("GRU 顺序头", GruHead), ("RWKV-7 顺序头", Rwkv7Head)]:
        torch.manual_seed(42)
        pos_rate, avg = run(name, cls)
        results[name] = (pos_rate, avg)
    print(f"\n{'='*60}\n汇总\n{'='*60}")
    print(f"{'头':<16} {'平均':<8} 各位置")
    for name, (pos_rate, avg) in results.items():
        print(f"{name:<16} {avg:.4f}   {[f'{r:.3f}' for r in pos_rate]}")


if __name__ == "__main__":
    main()
