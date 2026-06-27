"""DSpark 顺序头 RWKV-7 复现 · 阶段1b：双 key 累积任务（调难版）

目标：验证 RWKV-7 矩阵状态相对 GRU 向量状态在"需多步累积多 key"场景下的优势。

任务设计：
- 两个独立 key：key_a, key_b（weak hidden 无法精确定位）
- block[0] = key_a
- block[1] = key_b
- block[k] = (key_a + key_b + k) mod VOCAB  for k>=2
- 位置 2+ 必须同时知道 key_a 和 key_b 才能预测
- key_a 只在位置 0/1 的 prev_token 出现，位置 2 的 prev=block[1]=key_b
- 顺序头必须把 key_a 保留在状态里跨过 key_b 的覆盖

假设：
- GRU 向量状态在位置 1 处理 key_b 时会覆盖/衰减 key_a → 位置 2+ 接受率下降
- RWKV 矩阵状态容量大 rank 倍，能用不同子空间分别记 key_a/key_b → 位置 2+ 更稳
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

VOCAB = 256
DHID = 64
RANK = 32
BLOCK = 8
NOISE = 1.2  # 更强噪声，让 base 更弱
N_TRAIN = 3000
N_EVAL = 2000
LR = 3e-3
DEVICE = "cpu"


# ---------- 任务数据 ----------
def make_data(n, block_size=BLOCK):
    """生成 (keys, tokens, weak_hidden, target_logits)。
    block[0]=key_a, block[1]=key_b, block[k]=(key_a+key_b+k)%V for k>=2
    """
    key_a = torch.randint(0, VOCAB, (n,))
    key_b = torch.randint(0, VOCAB, (n,))
    pos = torch.arange(block_size, device=DEVICE).unsqueeze(0)  # [1,B]
    # block[0]=key_a, block[1]=key_b, block[k]=key_a+key_b+k
    tokens = torch.zeros(n, block_size, dtype=torch.long)
    tokens[:, 0] = key_a
    tokens[:, 1] = key_b
    for k in range(2, block_size):
        tokens[:, k] = (key_a + key_b + k) % VOCAB

    # weak hidden：同时含 key_a/key_b 的弱信号 + 强噪声
    ka_oh = F.one_hot(key_a, VOCAB).float()
    kb_oh = F.one_hot(key_b, VOCAB).float()
    proj = torch.randn(VOCAB, DHID) * 0.3
    weak = ka_oh @ proj + kb_oh @ proj  # [n, DHID]
    weak = weak + torch.randn_like(weak) * NOISE
    weak = weak.unsqueeze(1).expand(-1, block_size, -1)  # [n, block, DHID]

    tgt_logits = F.one_hot(tokens, VOCAB).float() * 10.0
    return (key_a, key_b), tokens, weak, tgt_logits


def make_prev(keys, tokens):
    """顺序头 teacher-forcing 输入。
    位置 0：无前缀，用 key_a（无信号）
    位置 k>=1：用 tokens[k-1]
    """
    key_a, _ = keys
    prev = tokens.clone()
    prev[:, 1:] = tokens[:, :-1].clone()
    prev[:, 0] = key_a
    return prev


# ---------- 顺序头（与 experiment.py 一致）----------
class NoHead(nn.Module):
    def __init__(self, vocab_size, rank, hidden_size):
        super().__init__()

    def forward_block(self, base_logits, prev_token_ids, hidden_states):
        return base_logits


class GruHead(nn.Module):
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


class Rwkv7Head(nn.Module):
    """RWKV-7 Delta Rule + L2 数值稳定（与 experiment.py 一致）。"""
    def __init__(self, vocab_size, rank, hidden_size):
        super().__init__()
        self.rank = rank
        self.token_emb = nn.Embedding(vocab_size, rank)
        jin = rank + hidden_size
        self.x_k = nn.Linear(jin, rank, bias=False)
        self.x_v = nn.Linear(jin, rank, bias=False)
        self.x_a = nn.Linear(jin, rank, bias=False)
        self.x_w = nn.Linear(jin, rank, bias=False)
        self.x_r = nn.Linear(jin, rank, bias=False)
        self.w_out = nn.Linear(rank, vocab_size, bias=False)

    def forward_block(self, base_logits, prev_token_ids, hidden_states):
        B, T, V = base_logits.shape
        prev_emb = self.token_emb(prev_token_ids)
        jin = torch.cat([prev_emb, hidden_states], dim=-1)
        k = self.x_k(jin); v = self.x_v(jin); a = self.x_a(jin)
        w = torch.sigmoid(self.x_w(jin)) * 0.9 + 0.05
        r = self.x_r(jin)
        s = torch.zeros(B, self.rank, self.rank, device=base_logits.device)
        biases = []
        eps = 1e-6
        for t in range(T):
            kt, vt, at, wt = k[:, t], v[:, t], a[:, t], w[:, t]
            s = s * wt.unsqueeze(1)
            s = s + kt.unsqueeze(2) * vt.unsqueeze(1)
            sa = (at.unsqueeze(1) * s).sum(dim=1)
            s = s + sa.unsqueeze(2) * kt.unsqueeze(1)
            s = s / (s.norm(dim=(1, 2), keepdim=True) + eps)
            sk = (s * kt.unsqueeze(1)).sum(dim=2)
            bias = self.w_out(torch.sigmoid(r[:, t]) * sk)
            biases.append(bias)
        return base_logits + torch.stack(biases, dim=1)


# ---------- Draft ----------
class Draft(nn.Module):
    def __init__(self, head_cls):
        super().__init__()
        self.lm_head = nn.Linear(DHID, VOCAB, bias=False)
        self.head = head_cls(VOCAB, RANK, DHID)

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


def run(name, head_cls, train_data, n_steps=N_TRAIN):
    keys, tokens, weak, tgt = train_data
    n = keys[0].shape[0]
    prev_all = make_prev(keys, tokens)
    draft = Draft(head_cls).to(DEVICE)
    opt = torch.optim.Adam(draft.parameters(), lr=LR)
    print(f"\n=== {name} ===")
    bs = 256
    for step in range(n_steps):
        idx = torch.randint(0, n, (bs,))
        out = draft(weak[idx], prev_all[idx])
        loss = ce_loss(out, tgt[idx]) + 0.9 * l1_loss(out, tgt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 300 == 0:
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
    # 关键指标：位置 2+（需双 key 累积）
    late = sum(pos_rate[2:]) / (BLOCK - 2)
    print(f"  位置2+平均(需双key): {late:.4f}")
    return pos_rate, avg, late


def main():
    print(f"VOCAB={VOCAB} DHID={DHID} RANK={RANK} BLOCK={BLOCK} NOISE={NOISE}")
    print(f"任务: block[0]=key_a, block[1]=key_b, block[k]=(key_a+key_b+k)%V for k>=2")
    print(f"      位置2+必须同时保留 key_a(被key_b覆盖) 和 key_b")
    train_data = make_data(4096)

    results = {}
    for name, cls in [("None", NoHead), ("GRU(DSpark)", GruHead), ("RWKV-7 Delta", Rwkv7Head)]:
        torch.manual_seed(42)
        pos_rate, avg, late = run(name, cls, train_data)
        results[name] = (pos_rate, avg, late)

    print("\n========== 汇总 ==========")
    print(f"{'头':<16} {'平均':<8} {'位置2+':<8} 各位置")
    for name, (pos_rate, avg, late) in results.items():
        print(f"{name:<16} {avg:.4f}   {late:.4f}   {[f'{r:.3f}' for r in pos_rate]}")
    print("\n假设验证: 位置2+ RWKV 应优于 GRU（矩阵状态容量 >> 向量状态）")


if __name__ == "__main__":
    main()
