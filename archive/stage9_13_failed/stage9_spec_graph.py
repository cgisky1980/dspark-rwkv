"""DSpark → RWKV · Speculative Decoding with CUDA Graph

用 CUDA graph 加速 forward，消除 Python 调用开销：
- draft T=1: 32ms → 6.8ms (4.74x)
- target T=4: 61ms → 22.9ms (2.68x)

预录制 graphs：
- draft_step_graph: draft T=1 forward
- target_verify_graph: target T=K forward
- target_replay_graphs[R]: target T=R forward (R=1..K, 用于 rollback replay)
- draft_replay_graphs[R]: draft T=R forward
"""
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from stage9_target_2p9b import RWKV7Target2p9B, DTYPE, DEVICE
from stage9_logit_fusion_01b import RWKV7Draft
from rwkv_tokenizer import TRIE_TOKENIZER

G1A_TARGET = Path(r"C:\work\niceui\g1a-2.9B.pth")
DRAFT_WEIGHTS = Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1a-0.4b-20250905-ctx4096.pth")


def softmax_cpu(logits):
    m = logits.max()
    e = torch.exp(logits - m)
    return e / e.sum()


def rejection_sample(candidate, small_probs, large_probs):
    """标准 Speculative Decoding 拒绝采样 (Leviathan 2023)"""
    idx = candidate
    q = small_probs[idx].item()
    p = large_probs[idx].item()
    large_argmax = large_probs.argmax().item()
    if large_argmax == candidate:
        return True, candidate
    if q <= 0:
        accept_prob = 0.0
    else:
        accept_prob = min(1.0, p / q)
    r = torch.rand(1).item()
    if r < accept_prob:
        return True, candidate
    else:
        residual = torch.clamp(large_probs - small_probs, min=0.0)
        s = residual.sum().item()
        if s <= 0:
            return False, large_argmax
        r2 = torch.rand(1).item() * s
        acc = 0.0
        for i in range(residual.shape[0]):
            acc += residual[i].item()
            if acc >= r2:
                return False, i
        return False, residual.shape[0] - 1


class GraphSpecDecoder:
    def __init__(self, draft, target, K):
        self.draft = draft
        self.target = target
        self.K = K

        # buffer
        self.d_tok_buf = torch.zeros([1, 1], device=DEVICE, dtype=torch.long)
        self.t_verify_buf = torch.zeros([1, K], device=DEVICE, dtype=torch.long)

        # state（所有 graph 共用）
        self.d_state = draft.zero_state(1)
        self.t_state = target.zero_state(1)

        # 预录制 graphs
        print(f"录制 CUDA graphs (K={K})...", flush=True)
        self._record_draft_step()
        self._record_target_verify()
        self._record_replay_graphs()
        print(f"CUDA graphs 录制完成", flush=True)

    def _record_draft_step(self):
        for _ in range(3):
            self.draft.forward(self.d_tok_buf, self.d_state)
        torch.cuda.synchronize()
        self.d_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.d_graph):
            self.d_out = self.draft.forward(self.d_tok_buf, self.d_state)
        torch.cuda.synchronize()

    def _record_target_verify(self):
        for _ in range(3):
            self.target.forward(self.t_verify_buf, self.t_state, return_hidden_layers=[])
        torch.cuda.synchronize()
        self.t_verify_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.t_verify_graph):
            self.t_verify_out, _ = self.target.forward(self.t_verify_buf, self.t_state, return_hidden_layers=[])
        torch.cuda.synchronize()

    def _record_replay_graphs(self):
        self.t_replay_graphs = {}
        self.d_replay_graphs = {}
        self.t_replay_bufs = {}
        self.d_replay_bufs = {}
        self.t_replay_outs = {}
        self.d_replay_outs = {}

        for R in range(1, self.K + 1):
            # target replay graph
            t_buf = torch.zeros([1, R], device=DEVICE, dtype=torch.long)
            for _ in range(3):
                self.target.forward(t_buf, self.t_state, return_hidden_layers=[])
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out, _ = self.target.forward(t_buf, self.t_state, return_hidden_layers=[])
            torch.cuda.synchronize()
            self.t_replay_graphs[R] = g
            self.t_replay_bufs[R] = t_buf
            self.t_replay_outs[R] = out

            # draft replay graph
            d_buf = torch.zeros([1, R], device=DEVICE, dtype=torch.long)
            for _ in range(3):
                self.draft.forward(d_buf, self.d_state)
            torch.cuda.synchronize()
            g2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g2):
                out2 = self.draft.forward(d_buf, self.d_state)
            torch.cuda.synchronize()
            self.d_replay_graphs[R] = g2
            self.d_replay_bufs[R] = d_buf
            self.d_replay_outs[R] = out2

    def reset_state(self):
        self.d_state[0].zero_()
        self.d_state[1].zero_()
        self.t_state[0].zero_()
        self.t_state[1].zero_()

    @torch.no_grad()
    def decode(self, tokenizer, prompt, n_generate, alpha):
        ids = tokenizer.encode(prompt)
        out = list(ids)

        # 重置 state
        self.reset_state()

        # prompt forward（普通，T 不固定不能用 graph）
        ctx = torch.tensor([ids], device=DEVICE, dtype=torch.long)
        t_logits_full, _ = self.target.forward(ctx, self.t_state, return_hidden_layers=[])
        d_logits_full = self.draft.forward(ctx, self.d_state)
        t_logits = t_logits_full[:, -1, :]  # [1, V]
        d_logits = d_logits_full[:, -1, :]  # [1, V]

        stats = {"accepted": 0, "rejected": 0, "total_draft": 0}

        while len(out) - len(ids) < n_generate:
            # === draft 生成 K 个候选（replay draft graph）===
            d_state_backup = [s.clone() for s in self.d_state]
            draft_tokens = []
            small_probs_list = []
            cur_d_logits = d_logits
            large_logits_for_fusion = t_logits.clone()

            for i in range(self.K):
                if i == 0:
                    fused = cur_d_logits + alpha * large_logits_for_fusion
                    probs = softmax_cpu(fused[0])
                else:
                    probs = softmax_cpu(cur_d_logits[0])

                tok = probs.argmax().item()
                small_probs_list.append(probs)
                draft_tokens.append(tok)

                # replay draft graph
                self.d_tok_buf.copy_(torch.tensor([[tok]], device=DEVICE, dtype=torch.long))
                self.d_graph.replay()
                cur_d_logits = self.d_out[:, -1, :]

            stats["total_draft"] += self.K

            # === target 验证（replay target verify graph）===
            t_state_backup = [s.clone() for s in self.t_state]
            self.t_verify_buf.copy_(torch.tensor([draft_tokens], device=DEVICE, dtype=torch.long))
            self.t_verify_graph.replay()

            # 构造 verify_logits_list
            verify_logits_list = [t_logits[0]]
            for k in range(self.K):
                verify_logits_list.append(self.t_verify_out[0, k])

            # === rejection sampling ===
            n_accept = 0
            correction_token = None
            for i in range(self.K):
                large_probs = softmax_cpu(verify_logits_list[i])
                accept, resampled = rejection_sample(draft_tokens[i], small_probs_list[i], large_probs)
                if accept:
                    n_accept += 1
                else:
                    correction_token = resampled
                    break

            stats["accepted"] += n_accept

            if n_accept == self.K:
                # 全部接受
                out.extend(draft_tokens)
                t_logits = self.t_verify_out[:, -1, :]
                d_logits = self.d_out[:, -1, :]
            else:
                # 中途拒绝：rollback + replay
                stats["rejected"] += 1
                out.extend(draft_tokens[:n_accept])
                out.append(correction_token)

                # rollback state
                self.t_state[0].copy_(t_state_backup[0])
                self.t_state[1].copy_(t_state_backup[1])
                self.d_state[0].copy_(d_state_backup[0])
                self.d_state[1].copy_(d_state_backup[1])

                # replay [accepted + correction]
                replay_tokens = draft_tokens[:n_accept] + [correction_token]
                R = len(replay_tokens)
                if R in self.t_replay_graphs:
                    t_buf = self.t_replay_bufs[R]
                    d_buf = self.d_replay_bufs[R]
                    t_buf.copy_(torch.tensor([replay_tokens], device=DEVICE, dtype=torch.long))
                    d_buf.copy_(torch.tensor([replay_tokens], device=DEVICE, dtype=torch.long))
                    self.t_replay_graphs[R].replay()
                    self.d_replay_graphs[R].replay()
                    t_logits = self.t_replay_outs[R][:, -1, :]
                    d_logits = self.d_replay_outs[R][:, -1, :]
                else:
                    # fallback: 普通 forward
                    replay_tensor = torch.tensor([replay_tokens], device=DEVICE, dtype=torch.long)
                    t_logits_full, _ = self.target.forward(replay_tensor, self.t_state, return_hidden_layers=[])
                    t_logits = t_logits_full[:, -1, :]
                    d_logits_full = self.draft.forward(replay_tensor, self.d_state)
                    d_logits = d_logits_full[:, -1, :]

            if out[-1] == 0 or len(out) - len(ids) >= n_generate:
                break

        return out, stats


def benchmark_baseline(target, tokenizer, prompt, n_generate=40):
    """基线：纯 target 自回归"""
    ids = tokenizer.encode(prompt)
    out = list(ids)
    state = target.zero_state(1)
    ctx = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    logits, _ = target.forward(ctx, state, return_hidden_layers=[])
    forwards = 1
    while len(out) - len(ids) < n_generate:
        next_tok = logits[0, -1].argmax().item()
        out.append(next_tok)
        nt = torch.tensor([[next_tok]], device=DEVICE, dtype=torch.long)
        logits, _ = target.forward(nt, state, return_hidden_layers=[])
        forwards += 1
        if next_tok == 0:
            break
    return out, {"target_forwards": forwards}


def main():
    print("=" * 70)
    print("DSpark → RWKV · Speculative Decoding with CUDA Graph")
    print(f"  Target: g1a-2.9B  |  Draft: g1a-0.4B-20250905")
    print(f"  CUDA Graph 加速: 消除 Python 调用开销")
    print("=" * 70)

    tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))
    print("加载 target...", flush=True)
    target = RWKV7Target2p9B(G1A_TARGET)
    print("加载 draft...", flush=True)
    draft = RWKV7Draft(DRAFT_WEIGHTS)
    print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    prompts = [
        "什么是人工智能",
        "中国的首都是哪里",
        "水的化学式是什么",
        "地球围绕太阳转吗",
        "一年有多少天",
        "太阳从哪个方向升起",
        "月亮是什么形状",
        "人类有几只手",
        "猫是动物吗",
        "苹果是什么颜色的",
    ]

    # 基线
    print(f"\n[基线: 纯 target 自回归, {len(prompts)} 条]", flush=True)
    t0 = time.time()
    total_base_fwd = 0
    for p in prompts:
        _, stats = benchmark_baseline(target, tokenizer, p, n_generate=40)
        total_base_fwd += stats["target_forwards"]
    base_time = (time.time() - t0) / len(prompts)
    base_fwd = total_base_fwd / len(prompts)
    print(f"  平均 {base_time:.2f}s/条, target forwards {base_fwd:.1f}", flush=True)

    # CUDA graph 版本
    K = 4
    alpha = 0.5
    print(f"\n[初始化 GraphSpecDecoder K={K}]", flush=True)
    decoder = GraphSpecDecoder(draft, target, K)

    print(f"\n[CUDA Graph Spec: K={K}, α={alpha}, {len(prompts)} 条]", flush=True)
    t0 = time.time()
    total_acc, total_draft = 0, 0
    for p in prompts:
        out, stats = decoder.decode(tokenizer, p, n_generate=40, alpha=alpha)
        total_acc += stats["accepted"]
        total_draft += stats["total_draft"]
    elapsed = time.time() - t0
    ar = total_acc / max(total_draft, 1)
    spec_time = elapsed / len(prompts)
    speedup = base_time / spec_time if spec_time > 0 else 0
    print(f"  接受率: {ar:.2%}", flush=True)
    print(f"  平均 {spec_time:.2f}s/条", flush=True)
    print(f"  加速比: {speedup:.2f}x", flush=True)

    # 对比：普通版本（无 CUDA graph）
    print(f"\n[普通 Spec (无 graph): K={K}, α={alpha}, {len(prompts)} 条]", flush=True)
    from stage9_batch_spec import speculative_decode_batch
    t0 = time.time()
    total_acc2, total_draft2 = 0, 0
    for p in prompts:
        out, stats = speculative_decode_batch(draft, target, tokenizer, p, n_generate=40, K=K, alpha=alpha)
        total_acc2 += stats["accepted"]
        total_draft2 += stats["total_draft"]
    elapsed2 = time.time() - t0
    ar2 = total_acc2 / max(total_draft2, 1)
    spec_time2 = elapsed2 / len(prompts)
    speedup2 = base_time / spec_time2 if spec_time2 > 0 else 0
    print(f"  接受率: {ar2:.2%}", flush=True)
    print(f"  平均 {spec_time2:.2f}s/条", flush=True)
    print(f"  加速比: {speedup2:.2f}x", flush=True)

    # 示例生成
    print(f"\n[示例生成: K={K}, α={alpha}]", flush=True)
    for p in prompts[:3]:
        out, _ = decoder.decode(tokenizer, p, n_generate=40, alpha=alpha)
        gen = tokenizer.decode(out[len(tokenizer.encode(p)):])
        print(f"  {p} -> {gen[:80]}", flush=True)


if __name__ == "__main__":
    main()
