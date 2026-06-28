"""DSpark → RWKV · 阶段9：Logit Fusion（0.1B draft + 2.9B target，大带小）

参考 rwkv7-state-fusion研究.md：0.4B + 2.93B 融合，α=0.3-0.7 时大模型纠正小模型错误。
本实验：用更小的 0.1B 作为 draft，看 2.9B 能否提升 0.1B 到可用水平。

权重处理（参考 albatross_ref/rwkv7.py）：
- 转置：att.g1/g2/a1/a2/w1/w2/v1/v2, ffn.value.weight
- 不转置：att.receptance/key/value/output.weight, ffn.key.weight, head.weight
- forward 全部用 F.linear(x, W)
"""
import math
import time
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from stage9_target_2p9b import RWKV7Target2p9B, DTYPE, DEVICE
from rwkv_tokenizer import TRIE_TOKENIZER

# ==================================================================
# 权重路径（本地已有）
# ==================================================================
TARGET_WEIGHTS = Path(r"C:\work\niceui\rwkv7-g1h_preview4533-2.9b-20260623-ctx8192.pth.pth")
DRAFT_WEIGHTS = Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-0.1b.pth")
LAMBADA_FILE = Path(__file__).parent / "data" / "lambada_test.jsonl"
HEAD_SIZE = 64


def lerp(x, y, w):
    return x + w * (y - x)


def layer_norm(x, w, b, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


def group_norm(x, w, b, H, eps=64e-5):
    *lead, C = x.shape
    x = x.reshape(*lead, H, HEAD_SIZE)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x = (x - mean) / (var + eps).sqrt()
    x = x.reshape(*lead, C)
    return x * w + b


class RWKV7Draft:
    """RWKV-7 0.1B（GPU FP16），支持完整或裁剪层数。"""
    def __init__(self, path, n_layers=None):
        print(f"加载 draft 权重: {path}")
        z = torch.load(path, map_location="cpu", weights_only=True)
        r_k_raw = z["blocks.0.att.r_k"].squeeze()
        self.H, self.N = r_k_raw.shape  # 0.1B: [12, 64]
        self.C = self.H * self.N  # 768
        self.V = z["emb.weight"].shape[0]  # 65536

        # 自动检测总层数
        max_layer = max(int(k.split('.')[1]) for k in z if k.startswith('blocks.'))
        total_layers = max_layer + 1
        self.n_layer = n_layers if n_layers is not None else total_layers
        print(f"  总层数 {total_layers}, 使用前 {self.n_layer} 层")

        # 权重转换（参考 albatross_ref/rwkv7.py line 119-120）
        # 判断是否需要转置：用 layer 0 的 w1 判断整个模型
        # - g1d-0.4B: w1 [C, mid] = [in, out] → 需要转置
        # - g1a-0.4B: w1 [mid, C] = [out, in] → 不需要转置
        C = self.C
        w1_key = "blocks.0.att.w1"
        if w1_key in z and z[w1_key].dim() == 2:
            need_transpose = (z[w1_key].shape[0] == C)
        else:
            need_transpose = True
        print(f"  权重转置判断: w1.shape={z[w1_key].shape}, C={C}, need_transpose={need_transpose}")
        keys = list(z.keys())
        for k in keys:
            v = z[k].squeeze().to(DTYPE)
            need_t_keys = ('att.g1', 'att.g2', 'att.a1', 'att.a2',
                          'att.w1', 'att.w2', 'att.v1', 'att.v2')
            if need_transpose and any(kw in k for kw in need_t_keys):
                v = v.t()
            if k.endswith("att.r_k"):
                v = v.flatten()
            z[k] = v.contiguous()
        z["emb.weight"] = layer_norm(z["emb.weight"], z["blocks.0.ln0.weight"], z["blocks.0.ln0.bias"])

        # 只保留前 n_layers 层，搬到 GPU
        self.z = {}
        for k, v in z.items():
            if k.startswith("blocks."):
                layer_idx = int(k.split(".")[1])
                if layer_idx < self.n_layer:
                    self.z[k] = v.to(DEVICE)
            else:
                self.z[k] = v.to(DEVICE)
        print(f"  Draft: L={self.n_layer} C={self.C} H={self.H} N={self.N} V={self.V}")

    def zero_state(self, B, device=DEVICE):
        return [
            torch.zeros(self.n_layer, 2, B, self.C, dtype=DTYPE, device=device),
            torch.zeros(self.n_layer, B, self.H, self.N, self.N, dtype=DTYPE, device=device),
        ]

    @torch.no_grad()
    def forward(self, tokens, state):
        z = self.z
        B, T = tokens.shape
        x = z["emb.weight"][tokens]
        v_first = torch.zeros_like(x)
        for i in range(self.n_layer):
            p = f"blocks.{i}."
            att = p + "att."
            ffn = p + "ffn."
            xx = layer_norm(x, z[p+"ln1.weight"], z[p+"ln1.bias"])
            xx, v_first = self.tmix(i, xx, state[0][i], v_first, state[1][i], att)
            x = x + xx
            xx = layer_norm(x, z[p+"ln2.weight"], z[p+"ln2.bias"])
            x = x + self.cmix(xx, state[0][i], ffn)
        x = layer_norm(x, z["ln_out.weight"], z["ln_out.bias"])
        logits = F.linear(x, z["head.weight"])
        return logits

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
            S_kk = (wkv_state * (-kkt).unsqueeze(-1)).sum(dim=-2)  # sa[v] = sum_k state[k,v] * (-kk[k])
            wkv_state.mul_(wt.unsqueeze(-1))  # state[k,v] *= w[k]
            wkv_state.add_(S_kk.unsqueeze(-2) * (kkt * at).unsqueeze(-1))  # + sa[v] * b[k]
            wkv_state.add_(kt.unsqueeze(-1) * vt.unsqueeze(-2))  # + k[k] * v[v] (outer)
            yt = (wkv_state * rt.unsqueeze(-1)).sum(dim=-2)  # y[v] = sum_k r[k] * state[k,v]
            y[:, t] = yt.reshape(B, H * N)
        y = group_norm(y, z[p+"ln_x.weight"], z[p+"ln_x.bias"], H)
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


def generate_single(model, tokenizer, prompt, n_tokens, is_target=False):
    """单模型生成"""
    ids = tokenizer.encode(prompt)
    state = model.zero_state(1)
    ctx_tensor = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    if is_target:
        logits, _ = model.forward(ctx_tensor, state, return_hidden_layers=[])
    else:
        logits = model.forward(ctx_tensor, state)
    out = list(ids)
    for _ in range(n_tokens):
        next_tok = logits[0, -1].argmax().item()
        out.append(next_tok)
        next_t = torch.tensor([[next_tok]], device=DEVICE, dtype=torch.long)
        if is_target:
            logits, _ = model.forward(next_t, state, return_hidden_layers=[])
        else:
            logits = model.forward(next_t, state)
    return tokenizer.decode(out, utf8_errors='replace')


def generate_fusion(draft, target, tokenizer, prompt, n_tokens, alpha, mode="prob"):
    """融合生成"""
    ids = tokenizer.encode(prompt)
    t_state = target.zero_state(1)
    d_state = draft.zero_state(1)
    ctx_tensor = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    t_logits, _ = target.forward(ctx_tensor, t_state, return_hidden_layers=[])
    d_logits = draft.forward(ctx_tensor, d_state)
    out = list(ids)
    for _ in range(n_tokens):
        t_logit = t_logits[:, -1, :]
        d_logit = d_logits[:, -1, :]
        if mode == "prob":
            t_prob = F.softmax(t_logit, dim=-1)
            d_prob = F.softmax(d_logit, dim=-1)
            fused = alpha * d_prob + (1 - alpha) * t_prob
        else:
            fused = alpha * d_logit + (1 - alpha) * t_logit
        next_tok = fused.argmax(dim=-1).item()
        out.append(next_tok)
        next_t = torch.tensor([[next_tok]], device=DEVICE, dtype=torch.long)
        t_logits, _ = target.forward(next_t, t_state, return_hidden_layers=[])
        d_logits = draft.forward(next_t, d_state)
    return tokenizer.decode(out, utf8_errors='replace')


def eval_alpha_scan(draft, target, tokenizer, texts, alphas, ctx_len=64, gen_len=16):
    """α 扫描：融合预测 vs target / vs draft 一致率"""
    results = {a: {"match_t": 0, "match_d": 0, "total": 0} for a in alphas}
    for text_idx, text in enumerate(texts):
        ids = tokenizer.encode(text)
        if len(ids) < ctx_len + gen_len + 10:
            continue
        ctx = ids[:ctx_len]
        gt_next = ids[ctx_len:ctx_len + gen_len]
        ctx_tensor = torch.tensor([ctx], device=DEVICE, dtype=torch.long)
        t_state = target.zero_state(1)
        d_state = draft.zero_state(1)
        t_logits, _ = target.forward(ctx_tensor, t_state, return_hidden_layers=[])
        d_logits = draft.forward(ctx_tensor, d_state)
        for t in range(gen_len):
            t_logit = t_logits[:, -1, :]
            d_logit = d_logits[:, -1, :]
            t_pred = t_logit.argmax(dim=-1).item()
            d_pred = d_logit.argmax(dim=-1).item()
            for a in alphas:
                t_prob = F.softmax(t_logit, dim=-1)
                d_prob = F.softmax(d_logit, dim=-1)
                fused = a * d_prob + (1 - a) * t_prob
                f_pred = fused.argmax(dim=-1).item()
                results[a]["total"] += 1
                if f_pred == t_pred:
                    results[a]["match_t"] += 1
                if f_pred == d_pred:
                    results[a]["match_d"] += 1
            next_tok = gt_next[t]
            next_t = torch.tensor([[next_tok]], device=DEVICE, dtype=torch.long)
            t_logits, _ = target.forward(next_t, t_state, return_hidden_layers=[])
            d_logits = draft.forward(next_t, d_state)
        if (text_idx + 1) % 20 == 0:
            print(f"  处理 {text_idx+1}/{len(texts)}")
    return {a: (results[a]["match_t"] / max(results[a]["total"], 1),
                results[a]["match_d"] / max(results[a]["total"], 1))
            for a in alphas}


def main():
    print("=" * 70)
    print("DSpark → RWKV · 阶段9: Logit Fusion (0.1B draft + 2.9B target)")
    print("  目的：验证 0.1B 是否能被 2.9B 通过 Logit Fusion 提升")
    print("  Fusion: prob 模式, α * draft_prob + (1-α) * target_prob")
    print("  α=0: 纯 target  α=1: 纯 draft")
    print("=" * 70)

    tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))

    print(f"\n[加载 2.9B target]")
    target = RWKV7Target2p9B(TARGET_WEIGHTS)

    print(f"\n[加载 0.1B draft]")
    draft = RWKV7Draft(DRAFT_WEIGHTS)
    print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # 实验 1：知识准确性
    print(f"\n{'='*70}")
    print(f"[实验 1: 知识准确性]")
    prompts = [
        "光速的数值是多少？",
        "What is the speed of light?",
        "中国的首都是哪里？",
        "1+1等于几？",
    ]
    for prompt in prompts:
        print(f"\n--- Prompt: {prompt} ---")
        print(f"[0.1B]: {generate_single(draft, tokenizer, prompt, 40)}")
        print(f"[2.9B]: {generate_single(target, tokenizer, prompt, 40, is_target=True)}")
        for alpha in [0.3, 0.5, 0.7]:
            print(f"[fusion α={alpha}]: {generate_fusion(draft, target, tokenizer, prompt, 40, alpha)}")

    # 实验 2：α 扫描
    print(f"\n{'='*70}")
    print(f"[实验 2: α 扫描]")
    texts = []
    with open(LAMBADA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    texts = texts[:80]
    print(f"  加载 {len(texts)} 条 LAMBADA")
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    t0 = time.time()
    scan = eval_alpha_scan(draft, target, tokenizer, texts, alphas, ctx_len=64, gen_len=16)
    elapsed = time.time() - t0

    print(f"\n{'α':>6}  {'vs target':>12}  {'vs draft':>12}  说明")
    print(f"{'-'*60}")
    for a in alphas:
        m_t, m_d = scan[a]
        note = ""
        if a == 0.0: note = "纯 target"
        elif a == 1.0: note = "纯 draft"
        else: note = f"融合 α={a}"
        print(f"{a:>6.2f}  {m_t:>12.4f}  {m_d:>12.4f}  {note}")
    print(f"\n耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
