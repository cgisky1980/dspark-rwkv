"""DSpark → RWKV · 阶段9：Logit Fusion（裁剪 0.4B draft + 2.9B target）

方案：
- Draft: 0.4B 裁剪到 12 层（取前 12 层，C=1024 保持）
- Target: 2.9B 完整 forward
- Fusion: final_logits = α * draft_logits + (1-α) * target_logits
- 评估: 不同 α 下的 top-1 命中率（vs 纯 target）

关键教训（来自 project_memory）：
- RWKV-7 state 是非线性递归积，state 层面融合无法绕过 Delta Rule 非线性
- Logit Fusion 是唯一稳定有效的方案：保持 state 独立，输出层线性组合

无需训练，只需扫描 α 找最优值。
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
# 配置
# ==================================================================
DRAFT_WEIGHTS = Path(__file__).parent / "weights" / "rwkv7-g1d-0.4b-20260210-ctx8192.pth"
LAMBADA_FILE = Path(__file__).parent / "data" / "lambada_test.jsonl"
HEAD_SIZE = 64
N_DRAFT_LAYERS = 12  # 从 24 层取前 12 层


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
    """RWKV-7 0.4B 裁剪到 12 层（GPU FP16）。

    forward(tokens, state) -> logits [B, T, V]
    """
    def __init__(self, path, n_layers=N_DRAFT_LAYERS):
        print(f"加载 draft 权重: {path}")
        z = torch.load(path, map_location="cpu", weights_only=True)
        r_k_raw = z["blocks.0.att.r_k"].squeeze()
        self.H, self.N = r_k_raw.shape  # [16, 64]
        self.C = self.H * self.N  # 1024
        self.V = z["emb.weight"].shape[0]  # 65536
        self.n_layer = n_layers  # 裁剪到 12 层

        # 权重转换（参考 stage9_target_2p9b.py）
        keys = list(z.keys())
        for k in keys:
            v = z[k].squeeze().to(DTYPE)
            if "key.weight" in k or "value.weight" in k or "receptance.weight" in k or "output.weight" in k or "head.weight" in k:
                v = v.t()
            if k.endswith("att.r_k"):
                v = v.flatten()
            z[k] = v.contiguous()
        # emb 预处理
        z["emb.weight"] = layer_norm(z["emb.weight"], z["blocks.0.ln0.weight"], z["blocks.0.ln0.bias"])
        # 搬到 GPU（只保留前 n_layers 层）
        self.z = {}
        for k, v in z.items():
            if k.startswith("blocks."):
                layer_idx = int(k.split(".")[1])
                if layer_idx < n_layers:
                    self.z[k] = v.to(DEVICE)
            else:
                self.z[k] = v.to(DEVICE)
        print(f"Draft: L={self.n_layer} C={self.C} H={self.H} N={self.N} V={self.V}")
        print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def zero_state(self, B, device=DEVICE):
        return [
            torch.zeros(self.n_layer, 2, B, self.C, dtype=DTYPE, device=device),
            torch.zeros(self.n_layer, B, self.H, self.N, self.N, dtype=DTYPE, device=device),
        ]

    @torch.no_grad()
    def forward(self, tokens, state):
        """tokens: [B, T] -> logits [B, T, V]"""
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
        logits = x @ z["head.weight"]  # [B, T, V]，head.weight 已转置为 [V,C] -> 实际是 [C,V]
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


def load_lambada(n=200):
    """加载 LAMBADA 测试集"""
    seqs = []
    with open(LAMBADA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            seqs.append(text)
    return seqs[:n]


def evaluate_alpha(draft, target, tokenizer, texts, alphas, ctx_len=64, gen_len=16):
    """评估不同 α 下的 top-1 命中率。

    对每个文本：
    1. 用前 ctx_len token 做 context
    2. target 自回归生成 gen_len 个 token（ground truth）
    3. draft 也自回归生成 gen_len 个 token
    4. 融合 logits = α*draft + (1-α)*target，取 argmax
    5. 对比融合 argmax vs target argmax（top-1 命中率）

    注：这里 target 和 draft 独立 forward，各自维护自己的 state
    """
    results = {a: {"match": 0, "total": 0} for a in alphas}

    for text_idx, text in enumerate(texts):
        ids = tokenizer.encode(text)
        if len(ids) < ctx_len + gen_len + 10:
            continue

        ctx = ids[:ctx_len]
        gt_next = ids[ctx_len:ctx_len + gen_len]  # 真实下一个 token

        # 用 ctx 做 prefix，target 和 draft 各自 forward
        ctx_tensor = torch.tensor([ctx], device=DEVICE, dtype=torch.long)

        # target forward ctx
        t_state = target.zero_state(1)
        t_logits, _ = target.forward(ctx_tensor, t_state, return_hidden_layers=[])

        # draft forward ctx
        d_state = draft.zero_state(1)
        d_logits = draft.forward(ctx_tensor, d_state)

        # 逐 token 生成并对比
        for t in range(gen_len):
            t_logit = t_logits[:, -1, :]  # [1, V]
            d_logit = d_logits[:, -1, :]  # [1, V]

            # target 的预测（ground truth）
            t_pred = t_logit.argmax(dim=-1).item()

            # 融合预测
            for a in alphas:
                fused = a * d_logit + (1 - a) * t_logit
                f_pred = fused.argmax(dim=-1).item()
                results[a]["total"] += 1
                if f_pred == t_pred:
                    results[a]["match"] += 1

            # 用真实 token 继续（teacher forcing，确保对齐）
            next_tok = torch.tensor([[gt_next[t]]], device=DEVICE, dtype=torch.long)
            t_logits, _ = target.forward(next_tok, t_state, return_hidden_layers=[])
            d_logits = draft.forward(next_tok, d_state)

        if (text_idx + 1) % 20 == 0:
            print(f"  处理 {text_idx+1}/{len(texts)} 文本")

    return {a: results[a]["match"] / max(results[a]["total"], 1) for a in alphas}


def main():
    print("=" * 60)
    print("DSpark → RWKV · 阶段9: Logit Fusion 评估")
    print("  Draft: 0.4B 裁剪到 12 层")
    print("  Target: 2.9B 完整")
    print("  Fusion: α * draft + (1-α) * target")
    print("=" * 60)

    # 1. 加载 tokenizer
    tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))
    print(f"tokenizer 加载完成")

    # 2. 加载 target (2.9B)
    print(f"\n[加载 2.9B target]")
    target = RWKV7Target2p9B()
    print(f"target GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # 3. 加载 draft (0.4B 裁剪到 12 层)
    print(f"\n[加载 0.4B draft (12层)]")
    draft = RWKV7Draft(DRAFT_WEIGHTS, n_layers=N_DRAFT_LAYERS)
    print(f"draft+target GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # 4. smoke test
    print(f"\n[smoke test]")
    tokens = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
    t_state = target.zero_state(1)
    d_state = draft.zero_state(1)
    t_logits, _ = target.forward(tokens, t_state, return_hidden_layers=[])
    d_logits = draft.forward(tokens, d_state)
    print(f"  target logits: {t_logits.shape} top-5: {t_logits[0,-1].topk(5).indices.tolist()}")
    print(f"  draft logits:  {d_logits.shape} top-5: {d_logits[0,-1].topk(5).indices.tolist()}")

    # 5. 评估不同 α
    print(f"\n[评估 Logit Fusion]")
    texts = load_lambada(n=100)
    print(f"  加载 {len(texts)} 条 LAMBADA 文本")
    alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    t0 = time.time()
    match_rates = evaluate_alpha(draft, target, tokenizer, texts, alphas, ctx_len=64, gen_len=16)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"结果（top-1 命中率：融合预测 vs target 独立预测）")
    print(f"{'α':>6}  {'match_rate':>12}  {'说明'}")
    print(f"{'-'*40}")
    for a in alphas:
        rate = match_rates[a]
        note = ""
        if a == 0.0:
            note = "纯 target（基线，应为 100%）"
        elif a == 1.0:
            note = "纯 draft（无融合）"
        else:
            note = f"融合 α={a}"
        print(f"{a:>6.2f}  {rate:>12.4f}  {note}")
    print(f"\n耗时: {elapsed:.1f}s")
    print(f"\n说明:")
    print(f"  - match_rate = 融合预测与 target 独立预测的一致率")
    print(f"  - α=0 时应为 100%（融合 = 纯 target）")
    print(f"  - α=1 时为纯 draft 的预测能力")
    print(f"  - 中间 α 越高，draft 影响越大，但命中率下降说明 draft 预测与 target 不一致")

    # 6. 评估 draft 独立预测 vs ground truth
    print(f"\n[补充评估：draft 独立 vs ground truth]")
    texts_gt = load_lambada(n=50)
    d_correct = 0
    t_correct = 0
    total = 0
    for text in texts_gt:
        ids = tokenizer.encode(text)
        if len(ids) < 80:
            continue
        ctx = ids[:64]
        gt_next = ids[64:80]
        ctx_tensor = torch.tensor([ctx], device=DEVICE)
        d_state = draft.zero_state(1)
        t_state = target.zero_state(1)
        d_logits = draft.forward(ctx_tensor, d_state)
        t_logits, _ = target.forward(ctx_tensor, t_state, return_hidden_layers=[])
        for tok in gt_next:
            d_pred = d_logits[0, -1].argmax().item()
            t_pred = t_logits[0, -1].argmax().item()
            if d_pred == tok:
                d_correct += 1
            if t_pred == tok:
                t_correct += 1
            total += 1
            next_t = torch.tensor([[tok]], device=DEVICE)
            d_logits = draft.forward(next_t, d_state)
            t_logits, _ = target.forward(next_t, t_state, return_hidden_layers=[])
    print(f"  draft (12层) 正确率: {d_correct}/{total} = {d_correct/max(total,1):.4f}")
    print(f"  target (2.9B) 正确率: {t_correct}/{total} = {t_correct/max(total,1):.4f}")


if __name__ == "__main__":
    main()
