"""DSpark 顺序头 RWKV-7 复现 · 阶段1v2：用正确 RWKV-7 公式重跑

基于 参考/dspark_rwkv/RWKV7公式参考.md 的正确公式重写 Rwkv7Head：
- S = S*w + (S@kk)*(-kk*a) + v⊗k   （DPLR 右乘，b=-kk*a）
- w = exp(-sigmoid(w_raw) / sqrt(e))，范围 [0.545, 1]
- kk = L2_normalize(k * k_k)        （对 kk 归一化，不是 S）
- k = LERP(k, k*a, k_a)             （k 修正）
- 输出：y = group_norm(S@r) + (r*k*r_k).sum * v

测试两种任务：
- 单 key：block[k] = (key + k) % V
- 双 key：block[0]=key_a, block[1]=key_b, block[k]=(key_a+key_b+k)%V
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

VOCAB = 256
DHID = 64
RANK = 32
BLOCK = 8
NOISE = 0.8
N_TRAIN = 2000
N_EVAL = 2000
LR = 3e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SQRT_E = math.sqrt(math.e)


# ---------- 任务数据 ----------
def make_data_single(n, block_size=BLOCK):
    keys = torch.randint(0, VOCAB, (n,))
    pos = torch.arange(block_size).unsqueeze(0)
    tokens = (keys.unsqueeze(1) + pos) % VOCAB
    key_oh = F.one_hot(keys, VOCAB).float()
    proj = torch.randn(VOCAB, DHID) * 0.3
    weak = key_oh @ proj
    weak = weak + torch.randn_like(weak) * NOISE
    weak = weak.unsqueeze(1).expand(-1, block_size, -1)
    tgt_logits = F.one_hot(tokens, VOCAB).float() * 10.0
    return (keys,), tokens, weak, tgt_logits


def make_data_double(n, block_size=BLOCK, noise=1.2):
    key_a = torch.randint(0, VOCAB, (n,))
    key_b = torch.randint(0, VOCAB, (n,))
    tokens = torch.zeros(n, block_size, dtype=torch.long)
    tokens[:, 0] = key_a
    tokens[:, 1] = key_b
    for k in range(2, block_size):
        tokens[:, k] = (key_a + key_b + k) % VOCAB
    ka_oh = F.one_hot(key_a, VOCAB).float()
    kb_oh = F.one_hot(key_b, VOCAB).float()
    proj = torch.randn(VOCAB, DHID) * 0.3
    weak = ka_oh @ proj + kb_oh @ proj
    weak = weak + torch.randn_like(weak) * noise
    weak = weak.unsqueeze(1).expand(-1, block_size, -1)
    tgt_logits = F.one_hot(tokens, VOCAB).float() * 10.0
    return (key_a, key_b), tokens, weak, tgt_logits


def make_prev(keys, tokens):
    """teacher forcing：位置 0 无前缀填 keys[0]，位置 k≥1 用 tokens[k-1]。"""
    prev = tokens.clone()
    prev[:, 1:] = tokens[:, :-1].clone()
    prev[:, 0] = keys[0]
    return prev


# ---------- 顺序头 ----------
class NoHead(nn.Module):
    def __init__(self, vocab_size, rank, hidden_size):
        super().__init__()

    def forward_block(self, base_logits, prev_token_ids, hidden_states):
        return base_logits


class GruHead(nn.Module):
    """DSpark RNNHead (GRU)。"""
    def __init__(self, vocab_size, rank, hidden_size):
        super().__init__()
        self.rank = rank
        self.token_emb = nn.Embedding(vocab_size, rank)
        self.joint = nn.Linear(2 * rank + hidden_size, 3 * rank)
        self.w_out = nn.Linear(rank, vocab_size, bias=False)

    def forward_block(self, base_logits, prev_token_ids, hidden_states):
        B, T, V = base_logits.shape
        prev_emb = self.token_emb(prev_token_ids)
        s = torch.zeros(B, self.rank, device=base_logits.device)
        biases = []
        for t in range(T):
            z = torch.cat([s, prev_emb[:, t], hidden_states[:, t]], dim=-1)
            proj = self.joint(z)
            g_raw, c_raw, o_raw = proj.chunk(3, dim=-1)
            g = torch.sigmoid(g_raw)
            s = g * s + (1 - g) * torch.tanh(c_raw)
            bias = self.w_out(torch.tanh(o_raw))
            biases.append(bias)
        return base_logits + torch.stack(biases, dim=1)


class Rwkv7HeadV2(nn.Module):
    """RWKV-7 DPLR 顺序头（带消融开关：use_shift / use_layernorm）。

    正确公式：S = S*w + (S@kk)*(-kk*a) + v⊗k
    消融开关：
    - use_shift: 是否用 RWKV-7 token shift（6 路 LERP(x, prev, x_param)）
    - use_layernorm: 是否在 time shift 前对 x 做 LayerNorm（RWKV-7 官方用 LN 不是 RMSNorm）
    """
    def __init__(self, vocab_size, rank, hidden_size, use_shift=True, use_layernorm=True):
        super().__init__()
        self.rank = rank
        self.hidden_size = hidden_size
        self.use_shift = use_shift
        self.use_layernorm = use_layernorm
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        # token shift 的 6 路插值参数（use_shift=False 时不使用）
        self.x_r = nn.Parameter(torch.zeros(hidden_size))
        self.x_w = nn.Parameter(torch.zeros(hidden_size))
        self.x_k = nn.Parameter(torch.zeros(hidden_size))
        self.x_v = nn.Parameter(torch.zeros(hidden_size))
        self.x_a = nn.Parameter(torch.zeros(hidden_size))
        self.x_g = nn.Parameter(torch.zeros(hidden_size))
        self.ln = nn.LayerNorm(hidden_size)
        # 从 hidden 投影出 r/k/v/w/a（use_shift=True 时输入=time-mixed，False 时输入=原始 hidden）
        self.r_proj = nn.Linear(hidden_size, rank, bias=False)
        self.k_proj = nn.Linear(hidden_size, rank, bias=False)
        self.v_proj = nn.Linear(hidden_size, rank, bias=False)
        self.w_proj = nn.Linear(hidden_size, rank, bias=False)
        self.a_proj = nn.Linear(hidden_size, rank, bias=False)
        self.w1 = nn.Linear(hidden_size, rank, bias=False)
        self.w2 = nn.Linear(rank, rank, bias=False)
        self.a0 = nn.Parameter(torch.zeros(rank))
        self.w0 = nn.Parameter(torch.zeros(rank))
        self.k_k = nn.Parameter(torch.ones(rank) * 0.1)
        self.k_a = nn.Parameter(torch.zeros(rank))
        self.r_k = nn.Parameter(torch.ones(rank) * 0.1)
        self.g1 = nn.Linear(hidden_size, rank, bias=False)
        self.g2 = nn.Linear(rank, rank, bias=False)
        self.w_out = nn.Linear(rank, vocab_size, bias=False)

    def forward_block(self, base_logits, prev_token_ids, hidden_states):
        B, T, V = base_logits.shape
        x = hidden_states
        # LayerNorm（可选）
        if self.use_layernorm:
            x = self.ln(x)
        # token shift（可选）：use_shift=False 时 6 路都等于 x（无 time mix）
        if self.use_shift:
            prev_emb = self.token_emb(prev_token_ids)
            xr = x + self.x_r * (prev_emb - x)
            xw = x + self.x_w * (prev_emb - x)
            xk = x + self.x_k * (prev_emb - x)
            xv = x + self.x_v * (prev_emb - x)
            xa = x + self.x_a * (prev_emb - x)
            xg = x + self.x_g * (prev_emb - x)
        else:
            xr = xw = xk = xv = xa = xg = x
        r = self.r_proj(xr)
        k = self.k_proj(xk)
        v = self.v_proj(xv)
        w_raw = self.w_proj(xw) + self.w2(torch.tanh(self.w1(xw)))
        a = torch.sigmoid(self.a0 + self.a_proj(xa))
        g = torch.sigmoid(self.g1(xg)) @ self.g2.weight

        S = torch.zeros(B, self.rank, self.rank, device=base_logits.device)
        biases = []
        eps = 1e-12
        for t in range(T):
            rt, kt, vt = r[:, t], k[:, t], v[:, t]
            at = a[:, t]
            wt_raw = w_raw[:, t]
            w = torch.exp(-torch.sigmoid(self.w0 + wt_raw) / SQRT_E)
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
            bias = self.w_out(y * g[:, t])
            biases.append(bias)
        return base_logits + torch.stack(biases, dim=1)


# ---------- Draft ----------
class Draft(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.lm_head = nn.Linear(DHID, VOCAB, bias=False)
        self.head = head  # 接受实例

    def forward(self, hidden, prev_tokens):
        base = self.lm_head(hidden)
        return self.head.forward_block(base, prev_tokens, hidden)


# ---------- 训练 + 评估 ----------
def ce_loss(d, t):
    return F.cross_entropy(d.reshape(-1, VOCAB), t.argmax(-1).reshape(-1))


def l1_loss(d, t):
    return (F.softmax(d.float(), -1) - F.softmax(t.float(), -1)).abs().sum(-1).mean()


def accept_rate(d, t):
    return (d.argmax(-1) == t.argmax(-1)).float()


def run(name, head_factory, train_data, n_steps=N_TRAIN):
    keys, tokens, weak, tgt = train_data
    n = keys[0].shape[0]
    prev_all = make_prev(keys, tokens)
    # 数据搬到 DEVICE
    weak = weak.to(DEVICE)
    prev_all = prev_all.to(DEVICE)
    tgt = tgt.to(DEVICE)
    draft = Draft(head_factory()).to(DEVICE)
    opt = torch.optim.Adam(draft.parameters(), lr=LR)
    print(f"\n=== {name} === (device={DEVICE})")
    bs = 256
    for step in range(n_steps):
        idx = torch.randint(0, n, (bs,))
        out = draft(weak[idx], prev_all[idx])
        loss = ce_loss(out, tgt[idx]) + 0.9 * l1_loss(out, tgt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 200 == 0:
            with torch.no_grad():
                acc = accept_rate(out, tgt[idx]).mean().item()
            print(f"  step {step+1:4d}  loss={loss.item():.4f}  acc={acc:.3f}")

    draft.eval()
    pos_acc = [0.0] * BLOCK
    pos_cnt = [0] * BLOCK
    with torch.no_grad():
        idx = torch.arange(min(N_EVAL, n))
        for i in range(0, idx.shape[0], 256):
            sub = idx[i:i + 256]
            out = draft(weak[sub], prev_all[sub])
            acc = accept_rate(out, tgt[sub])
            for t in range(BLOCK):
                pos_acc[t] += acc[:, t].sum().item()
                pos_cnt[t] += acc[:, t].numel()
    pos_rate = [pos_acc[t] / max(pos_cnt[t], 1) for t in range(BLOCK)]
    avg = sum(pos_rate) / BLOCK
    print(f"  最终各位置接受率: {[f'{r:.3f}' for r in pos_rate]}")
    print(f"  平均接受率: {avg:.4f}")
    return pos_rate, avg


def run_task(task_name, make_data_fn):
    print(f"\n{'='*60}")
    print(f"任务: {task_name}")
    print(f"{'='*60}")
    train_data = make_data_fn(4096)
    results = {}
    # 消融变体：name -> head_factory
    variants = [
        ("None(baseline)", lambda: NoHead(VOCAB, RANK, DHID)),
        ("GRU(DSpark)", lambda: GruHead(VOCAB, RANK, DHID)),
        ("RWKV baseline(无shift无LN)", lambda: Rwkv7HeadV2(VOCAB, RANK, DHID, use_shift=False, use_layernorm=False)),
        ("实验1: +shift", lambda: Rwkv7HeadV2(VOCAB, RANK, DHID, use_shift=True, use_layernorm=False)),
        ("实验1+2: shift+LN", lambda: Rwkv7HeadV2(VOCAB, RANK, DHID, use_shift=True, use_layernorm=True)),
    ]
    for name, factory in variants:
        torch.manual_seed(42)
        pos_rate, avg = run(name, factory, train_data)
        results[name] = (pos_rate, avg)
    print(f"\n--- {task_name} 汇总 ---")
    print(f"{'头':<28} {'平均':<8} 各位置")
    for name, (pos_rate, avg) in results.items():
        print(f"{name:<28} {avg:.4f}   {[f'{r:.3f}' for r in pos_rate]}")
    return results


def main():
    print(f"VOCAB={VOCAB} DHID={DHID} RANK={RANK} BLOCK={BLOCK}")
    print(f"RWKV-7 DPLR 公式: S=S*w+(S@kk)*(-kk*a)+v⊗k, w=exp(-sig/sqrt(e))")
    print(f"消融: baseline(无shift无LN) / 实验1(+shift) / 实验1+2(shift+LN)")
    run_task("单 key: block[k]=(key+k)%V", lambda n: make_data_single(n))
    run_task("双 key: block[0]=a,block[1]=b,block[k]=(a+b+k)%V", lambda n: make_data_double(n))


if __name__ == "__main__":
    main()
