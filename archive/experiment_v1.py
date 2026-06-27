"""DSpark 顺序头 RWKV-7 复现 · 阶段1：合成数据单测（v2 调难版）

目标：验证 RWKV-7 Delta Rule cell 作为 DSpark 顺序头，相比 GRU(RNNHead)/无顺序头，
能否在 block 内各位置取得更高 top-1 接受率。

任务设计（让前缀依赖成为关键信号）：
- 每个 block 由一个"key token"决定：block[k] = (key + k) mod VOCAB
- draft 主干 hidden 仅含 key 的"粗方向"（投影到低秩 + 噪声），无法精确定位 key
- 顺序头从"block 内已生成的真实 token"（teacher forcing）累积推断 key
  - None 头：只能靠弱 hidden，接受率上限低
  - 顺序头：从 x_{k-1}=(key+k-1) 反推 key，再算 x_k=(key+k)，精确命中
- RWKV-7 Delta 的矩阵状态天然适合"记一个 key 向量"，理论应优于 GRU
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

VOCAB = 256
DHID = 64
RANK = 32
BLOCK = 8
KEY_BITS = 5  # key 投影后的 bit 数，控制主干信息量
NOISE = 0.8  # 主干 hidden 噪声强度
N_TRAIN = 2000
N_EVAL = 2000
LR = 3e-3
DEVICE = "cpu"


# ---------- 任务数据 ----------
def make_data(n, block_size=BLOCK):
    """生成 (key, block_tokens, weak_hidden, target_logits)。
    block[k] = (key + k) mod VOCAB
    weak_hidden = noisy low-rank encoding of key（draft 主干输入）
    """
    keys = torch.randint(0, VOCAB, (n,))
    pos = torch.arange(block_size, device=DEVICE).unsqueeze(0)  # [1,B]
    tokens = (keys.unsqueeze(1) + pos) % VOCAB  # [n, block]
    # 主干 weak hidden：key 投影到 DHID 维 + 强噪声，让 lm_head 无法直接精确定位
    key_onehot = F.one_hot(keys, VOCAB).float()  # [n,V]
    proj = torch.randn(VOCAB, DHID) * 0.3
    weak = key_onehot @ proj  # [n, DHID]
    weak = weak + torch.randn_like(weak) * NOISE
    weak = weak.unsqueeze(1).expand(-1, block_size, -1)  # [n, block, DHID]
    # target logits：one-hot of target token
    tgt_logits = F.one_hot(tokens, VOCAB).float() * 10.0  # [n, block, VOCAB]
    return keys, tokens, weak, tgt_logits


# ---------- 顺序头 ----------
class NoHead(nn.Module):
    """无顺序头：base_logits 不变。"""
    def __init__(self, vocab_size, rank, hidden_size):
        super().__init__()

    def forward_block(self, base_logits, prev_token_ids, hidden_states):
        return base_logits


class GruHead(nn.Module):
    """DSpark RNNHead（GRU 式）。"""
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
    """RWKV-7 Delta Rule 作为顺序头。

    状态 s 是 [B, rank, rank] 矩阵。每步：
      s = s * w + k ⊗ v + (a · s) ⊗ k     (Delta Rule)
      bias = W_out( sigmoid(r) ⊙ (s · k) )
    对应 参考/web-rwkv time_mix_v7.wgsl: ss[c] = ss[c]*w[c] + kk[c]*vv + sa*bb[c]。

    k/v/a/w/r 从 [prev_token_emb ; hidden] 联合投影（与 GRU 头对齐，
    也符合 RWKV 实际使用 token 输入的 time-mix 语义）。
    """
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
        prev_emb = self.token_emb(prev_token_ids)  # [B,T,rank]
        jin = torch.cat([prev_emb, hidden_states], dim=-1)  # [B,T,rank+hid]
        k = self.x_k(jin)
        v = self.x_v(jin)
        a = self.x_a(jin)
        w = torch.sigmoid(self.x_w(jin)) * 0.9 + 0.05
        r = self.x_r(jin)

        s = torch.zeros(B, self.rank, self.rank, device=base_logits.device)
        biases = []
        eps = 1e-6
        for t in range(T):
            kt = k[:, t]
            vt = v[:, t]
            at = a[:, t]
            wt = w[:, t]
            s = s * wt.unsqueeze(1)                       # 衰减
            s = s + kt.unsqueeze(2) * vt.unsqueeze(1)     # k⊗v
            sa = (at.unsqueeze(1) * s).sum(dim=1)          # a·s
            s = s + sa.unsqueeze(2) * kt.unsqueeze(1)      # (a·s)⊗k
            # 数值稳定：对状态做 L2 归一化（对应 web-rwkv 的 group norm 思路）
            s = s / (s.norm(dim=(1, 2), keepdim=True) + eps)
            sk = (s * kt.unsqueeze(1)).sum(dim=2)          # s·k
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
def ce_loss(draft_logits, target_logits):
    tgt_tok = target_logits.argmax(dim=-1)
    return F.cross_entropy(draft_logits.reshape(-1, VOCAB), tgt_tok.reshape(-1))


def l1_loss(draft_logits, target_logits):
    p = F.softmax(draft_logits.float(), dim=-1)
    q = F.softmax(target_logits.float(), dim=-1)
    return (p - q).abs().sum(dim=-1).mean()


def accept_rate(draft_logits, target_logits):
    d = draft_logits.argmax(dim=-1)
    t = target_logits.argmax(dim=-1)
    return (d == t).float()


def make_prev(keys, tokens):
    """DSpark 顺序头的 teacher-forcing 输入。
    位置 0：无前缀，用 key 自身（无顺序头信号，靠 base logits）
    位置 k≥1：用 tokens[k-1]（block 内前一个真实 token）
    顺序头从 prev_token 反推 key，再算当前位置 token。
    """
    prev = tokens.clone()
    prev[:, 1:] = tokens[:, :-1].clone()
    prev[:, 0] = keys  # 第 0 位无前缀，填 key（顺序头在第 0 位无法用前缀）
    return prev


def run(head_name, head_cls, train_data, n_steps=N_TRAIN):
    keys, tokens, weak, tgt_logits = train_data
    n = keys.shape[0]
    prev_all = make_prev(keys, tokens)
    draft = Draft(head_cls).to(DEVICE)
    opt = torch.optim.Adam(draft.parameters(), lr=LR)
    print(f"\n=== {head_name} ===")
    bs = 256
    for step in range(n_steps):
        idx = torch.randint(0, n, (bs,))
        prev = prev_all[idx]
        hid = weak[idx]
        tgt = tgt_logits[idx]
        out = draft(hid, prev)
        loss = ce_loss(out, tgt) + 0.9 * l1_loss(out, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            with torch.no_grad():
                acc = accept_rate(out, tgt).mean().item()
            print(f"  step {step+1:4d}  loss={loss.item():.4f}  acc={acc:.3f}")

    # 最终评估
    draft.eval()
    pos_acc = [0.0] * BLOCK
    pos_cnt = [0] * BLOCK
    with torch.no_grad():
        idx = torch.arange(min(N_EVAL, n))
        for i in range(0, idx.shape[0], 256):
            sub = idx[i:i + 256]
            prev = prev_all[sub]
            hid = weak[sub]
            tgt = tgt_logits[sub]
            out = draft(hid, prev)
            acc = accept_rate(out, tgt)
            for t in range(BLOCK):
                pos_acc[t] += acc[:, t].sum().item()
                pos_cnt[t] += acc[:, t].numel()
    pos_rate = [pos_acc[t] / max(pos_cnt[t], 1) for t in range(BLOCK)]
    avg = sum(pos_rate) / BLOCK
    print(f"  最终各位置接受率: {[f'{r:.3f}' for r in pos_rate]}")
    print(f"  平均接受率: {avg:.4f}")
    return pos_rate, avg


def main():
    print(f"VOCAB={VOCAB} DHID={DHID} RANK={RANK} BLOCK={BLOCK} NOISE={NOISE}")
    print(f"任务: block[k]=(key+k)%VOCAB, 主干仅含 key 噪声弱表示")
    train_data = make_data(4096)

    results = {}
    for name, cls in [("None", NoHead), ("GRU(DSpark)", GruHead), ("RWKV-7 Delta", Rwkv7Head)]:
        torch.manual_seed(42)  # 每个头同样的 draft lm_head 初始化
        pos_rate, avg = run(name, cls, train_data)
        results[name] = (pos_rate, avg)

    print("\n========== 汇总 ==========")
    print(f"{'头':<16} {'平均':<8} 各位置接受率")
    for name, (pos_rate, avg) in results.items():
        print(f"{name:<16} {avg:.4f}   {[f'{r:.3f}' for r in pos_rate]}")
    print("\n说明: 第 0 位无前缀(顺序头无用), 后续位置应靠前缀累积提升")


if __name__ == "__main__":
    main()
