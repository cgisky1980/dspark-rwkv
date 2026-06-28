"""DSpark → RWKV · 阶段9：Logit Fusion 正确评估（大带小）

之前的实验搞反了方向。正确方向（参考 rwkv7-state-fusion研究.md）：
- 0.4B + 2.9B 并行推理，logits 融合
- 大模型提升小模型的输出质量
- α 是小模型权重（α=0 纯大模型，α=1 纯小模型）

评估指标：
- 0.4B 单独正确率（vs target top-1 作为 ground truth）
- 2.9B 单独正确率（基线）
- 融合后正确率（看是否比 0.4B 单独好）

实验场景：
1. 知识准确性：光速、年份等事实
2. 长文本生成：看融合输出是否更接近大模型
3. Top-1 一致率：融合 vs target（α 扫描）
"""
import math
import time
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from stage9_target_2p9b import RWKV7Target2p9B, DTYPE, DEVICE
from rwkv_tokenizer import TRIE_TOKENIZER

DRAFT_WEIGHTS = Path(__file__).parent / "weights" / "rwkv7-g1d-0.4b-20260210-ctx8192.pth"
LAMBADA_FILE = Path(__file__).parent / "data" / "lambada_test.jsonl"
HEAD_SIZE = 64
N_DRAFT_LAYERS_FULL = 24  # 完整 0.4B 24 层


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
    """RWKV-7 0.4B（GPU FP16），支持完整或裁剪层数。"""
    def __init__(self, path, n_layers=N_DRAFT_LAYERS_FULL):
        print(f"加载 draft 权重: {path} (取前 {n_layers} 层)")
        z = torch.load(path, map_location="cpu", weights_only=True)
        r_k_raw = z["blocks.0.att.r_k"].squeeze()
        self.H, self.N = r_k_raw.shape
        self.C = self.H * self.N
        self.V = z["emb.weight"].shape[0]
        self.n_layer = n_layers

        keys = list(z.keys())
        for k in keys:
            v = z[k].squeeze().to(DTYPE)
            if "key.weight" in k or "value.weight" in k or "receptance.weight" in k or "output.weight" in k or "head.weight" in k:
                v = v.t()
            if k.endswith("att.r_k"):
                v = v.flatten()
            z[k] = v.contiguous()
        z["emb.weight"] = layer_norm(z["emb.weight"], z["blocks.0.ln0.weight"], z["blocks.0.ln0.bias"])

        self.z = {}
        for k, v in z.items():
            if k.startswith("blocks."):
                layer_idx = int(k.split(".")[1])
                if layer_idx < n_layers:
                    self.z[k] = v.to(DEVICE)
            else:
                self.z[k] = v.to(DEVICE)
        print(f"Draft: L={self.n_layer} C={self.C} H={self.H} N={self.N} V={self.V}")

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
        logits = x @ z["head.weight"]
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
        kk = kk.view(B, T, H, N)
        k = k.view(B, T, H, N)
        v = v.view(B, T, H, N)
        r = r.view(B, T, H, N)
        w = w.view(B, T, H, N)
        a = a.view(B, T, H, N)
        kk_norm = kk / (kk.norm(dim=-1, keepdim=True) + 1e-12)
        y = torch.zeros(B, T, H * N, dtype=DTYPE, device=x.device)
        for t in range(T):
            kt, vt, rt, wt, at = k[:, t], v[:, t], r[:, t], w[:, t], a[:, t]
            kkt = kk_norm[:, t]
            wkv_state = wkv_state * wt.unsqueeze(-1)
            S_kk = (wkv_state * kkt.unsqueeze(-2)).sum(dim=-1)
            wkv_state = wkv_state + S_kk.unsqueeze(-1) * (-kkt * at).unsqueeze(-2)
            wkv_state = wkv_state + vt.unsqueeze(-2) * kt.unsqueeze(-1)
            yt = (wkv_state * rt.unsqueeze(-2)).sum(dim=-1)
            y[:, t] = yt.reshape(B, H * N)
        y = group_norm(y, z[p+"ln_x.weight"], z[p+"ln_x.bias"], H)
        rkr = (r.reshape(B, T, H, N) * k * z[p+"r_k"].view(1, 1, H, N)).sum(dim=-1, keepdim=True)
        y = y + (rkr * v.reshape(B, T, H, N)).reshape(B, T, H * N)
        return (y * g) @ z[p+"output.weight"], v_first

    def cmix(self, x, shift_state, p):
        z = self.z
        prev = torch.cat([shift_state[1].unsqueeze(1), x[:, :-1, :]], dim=1)
        shift_state[1] = x[:, -1, :]
        k = lerp(x, prev, z[p+"x_k"])
        k = torch.relu(k @ z[p+"key.weight"]) ** 2
        return k @ z[p+"value.weight"]


def generate_with_fusion(draft, target, tokenizer, prompt, n_tokens=40, alpha=0.5, mode="prob"):
    """融合生成：draft 和 target 并行 forward，logits 融合。"""
    ids = tokenizer.encode(prompt)
    print(f"\n[Prompt] {prompt[:80]}...")
    print(f"  tokens: {len(ids)}")

    # 初始化两个模型的 state
    t_state = target.zero_state(1)
    d_state = draft.zero_state(1)

    # 用 prompt 预热两个模型
    ctx_tensor = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    t_logits, _ = target.forward(ctx_tensor, t_state, return_hidden_layers=[])
    d_logits = draft.forward(ctx_tensor, d_state)

    out_ids = list(ids)
    for step in range(n_tokens):
        t_logit = t_logits[:, -1, :]  # [1, V]
        d_logit = d_logits[:, -1, :]

        if mode == "prob":
            t_prob = F.softmax(t_logit, dim=-1)
            d_prob = F.softmax(d_logit, dim=-1)
            fused_prob = alpha * d_prob + (1 - alpha) * t_prob
            next_tok = fused_prob.argmax(dim=-1).item()
        else:  # logit
            fused_logit = alpha * d_logit + (1 - alpha) * t_logit
            next_tok = fused_logit.argmax(dim=-1).item()

        out_ids.append(next_tok)

        # 两个模型各自 forward
        next_t = torch.tensor([[next_tok]], device=DEVICE, dtype=torch.long)
        t_logits, _ = target.forward(next_t, t_state, return_hidden_layers=[])
        d_logits = draft.forward(next_t, d_state)

    out_text = tokenizer.decode(out_ids, utf8_errors='replace')
    return out_text


def eval_alpha_scan(draft, target, tokenizer, texts, alphas, ctx_len=64, gen_len=16):
    """α 扫描：融合预测 vs target 独立预测的 top-1 一致率。

    α=0 (纯 target) 时应 100%。
    α 越大，draft 影响越大。
    关键看：α=0.3-0.7 时一致率是否仍高（说明大模型仍主导）。
    """
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
                # prob 模式融合
                t_prob = F.softmax(t_logit, dim=-1)
                d_prob = F.softmax(d_logit, dim=-1)
                fused_prob = a * d_prob + (1 - a) * t_prob
                f_pred = fused_prob.argmax(dim=-1).item()
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
    print("DSpark → RWKV · 阶段9: Logit Fusion 正确评估（大带小）")
    print("  Draft: 0.4B 完整 24 层")
    print("  Target: 2.9B 完整")
    print("  Fusion: prob 模式, α * draft_prob + (1-α) * target_prob")
    print("  α=0: 纯 target  α=1: 纯 draft")
    print("=" * 70)

    tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))

    print(f"\n[加载 2.9B target]")
    target = RWKV7Target2p9B()

    print(f"\n[加载 0.4B draft (24 层完整)]")
    draft = RWKV7Draft(DRAFT_WEIGHTS, n_layers=N_DRAFT_LAYERS_FULL)
    print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # 实验 1：知识准确性（参考 rwkv7-state-fusion研究.md）
    print(f"\n{'='*70}")
    print(f"[实验 1: 知识准确性 - 光速]")
    prompt = "光速的数值是多少？"

    print(f"\n--- 0.4B 单独 ---")
    d_state = draft.zero_state(1)
    ids = tokenizer.encode(prompt)
    ctx_tensor = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    d_logits = draft.forward(ctx_tensor, d_state)
    d_out = list(ids)
    for _ in range(40):
        next_tok = d_logits[0, -1].argmax().item()
        d_out.append(next_tok)
        d_logits = draft.forward(torch.tensor([[next_tok]], device=DEVICE), d_state)
    print(f"  {tokenizer.decode(d_out, utf8_errors='replace')}")

    print(f"\n--- 2.9B 单独 ---")
    t_state = target.zero_state(1)
    t_logits, _ = target.forward(ctx_tensor, t_state, return_hidden_layers=[])
    t_out = list(ids)
    for _ in range(40):
        next_tok = t_logits[0, -1].argmax().item()
        t_out.append(next_tok)
        t_logits, _ = target.forward(torch.tensor([[next_tok]], device=DEVICE), t_state, return_hidden_layers=[])
    print(f"  {tokenizer.decode(t_out, utf8_errors='replace')}")

    for alpha in [0.3, 0.5, 0.7]:
        print(f"\n--- 融合 α={alpha} (prob 模式) ---")
        out = generate_with_fusion(draft, target, tokenizer, prompt, n_tokens=40, alpha=alpha, mode="prob")
        print(f"  {out}")

    # 实验 2：α 扫描
    print(f"\n{'='*70}")
    print(f"[实验 2: α 扫描 - 融合预测 vs target/draft 一致率]")
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
    print(f"\n解读:")
    print(f"  - vs target 高 = 融合预测与 target 一致（大模型主导）")
    print(f"  - vs draft 高 = 融合预测与 draft 一致（小模型主导）")
    print(f"  - 之前研究: α≤0.7 时大模型仍能主导知识准确性")


if __name__ == "__main__":
    main()
