"""DSpark → RWKV · 阶段13: CUDA Graph 加速 draft 的 K-state 投机解码

基于 stage12,但 draft 的 T=1 forward 用 CUDA Graph 捕获,消除 kernel launch 开销。
- target: CUDA(fp32io16)
- draft: CUDA(fp32io16) + CUDA Graph(T=1)

理论分析:
- stage12 draft T=1 forward = 6.5ms,其中 launch 开销 4ms
- CUDA Graph 后 draft T=1 = 2.4ms
- K=4 每轮 draft 生成:3 × 2.4ms = 7.2ms(原 19ms)
- target 验证 [1,4] = 14ms
- 一轮总时间(无拒绝)~ 21ms(原 33ms)
"""
import math
import time
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from stage11_cuda_target import RWKV7Target2p9BCuda, DTYPE
from stage9_logit_fusion_01b import TARGET_WEIGHTS, LAMBADA_FILE, DEVICE
from rwkv_tokenizer import TRIE_TOKENIZER

DRAFT_WEIGHTS = Path(r"C:\work\niceui\test\dspark_rwkv\weights\rwkv7-g1d-0.4b-20260210-ctx8192.pth")


def clone_state(state):
    return [s.clone() for s in state]


class DraftGraphWrapper:
    """CUDA Graph 包装的 draft 模型

    - T=1 forward:用 CUDA Graph(高频路径)
    - T>1 forward:普通 forward(prompt 预填、replay)
    """

    def __init__(self, weights_path):
        self.model = RWKV7Target2p9BCuda(weights_path)
        self.graph = None
        self.static_tok = None
        self.static_state = None
        self.static_logits = None
        self._graph_ready = False

    def zero_state(self, B):
        return self.model.zero_state(B)

    def _capture_graph(self, init_state):
        """捕获 T=1 forward graph
        
        init_state:用于 warmup + capture 的 state(会被消费,但 capture 后用 static_state)
        """
        # 静态缓冲区
        self.static_tok = torch.zeros((1, 1), device=DEVICE, dtype=torch.long)
        self.static_state = init_state  # 直接用传入的 state 作为 capture 对象
        self.static_logits = torch.zeros((1, 1, 65536), device=DEVICE, dtype=torch.float16)

        # warmup(用独立 state 避免污染 static_state)
        warmup_state = self.model.zero_state(1)
        for _ in range(3):
            self.model.forward(self.static_tok, warmup_state, [])
        torch.cuda.synchronize()

        # capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            out, _ = self.model.forward(self.static_tok, self.static_state, [])
            self.static_logits.copy_(out)
        torch.cuda.synchronize()
        self._graph_ready = True

    def restore_state(self, new_state):
        """恢复 state(拒绝后用)"""
        for i in range(len(self.static_state)):
            self.static_state[i].copy_(new_state[i])

    def forward_t1(self, token_id):
        """T=1 graph forward(返回 [1, V] logits)
        
        会 in-place 更新 static_state(= self.model 的 state)
        """
        if not self._graph_ready:
            raise RuntimeError("graph not captured, call _capture_graph first")
        self.static_tok[0, 0] = token_id
        self.graph.replay()
        return self.static_logits[:, -1, :]  # [1, V]

    def forward(self, tokens, state, return_hidden_layers=None):
        """通用 forward(走普通路径)"""
        return self.model.forward(tokens, state, return_hidden_layers or [])


def speculative_decode(draft, target, tokenizer, prompt, n_generate=40, K=4, alpha=0.5):
    """K-state 并发验证投机解码(draft 用 CUDA Graph)

    draft: DraftGraphWrapper
    target: RWKV7Target2p9BCuda
    """
    ids = tokenizer.encode(prompt)
    out = list(ids)

    # 初始化 state
    t_state = target.zero_state(1)
    d_state = draft.zero_state(1)

    # forward prompt(普通路径,T>1)
    ctx_tensor = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    t_logits_full, _ = target.forward(ctx_tensor, t_state, return_hidden_layers=[])
    d_logits_full, _ = draft.forward(ctx_tensor, d_state, [])

    t_logits = t_logits_full[:, -1, :]
    d_logits = d_logits_full[:, -1, :]

    # 捕获 draft T=1 graph(用 d_state 作为 static_state)
    draft._capture_graph(d_state)

    stats = {"accepted": 0, "rejected": 0, "total_draft": 0,
             "target_forwards": 1, "draft_forwards": 1, "draft_graph_forwards": 0}

    while len(out) - len(ids) < n_generate:
        # === 阶段 1: draft 生成 K 个候选 ===
        d_state_backup = clone_state(d_state)

        draft_tokens = []
        # 候选 1:logits 融合
        fused = d_logits + alpha * t_logits
        tok = fused.argmax(dim=-1).item()
        draft_tokens.append(tok)

        # 候选 2..K:graph T=1 forward
        for k in range(1, K):
            d_l = draft.forward_t1(tok)  # [1, V],in-place 更新 d_state
            tok = d_l[0].argmax().item()
            draft_tokens.append(tok)
            stats["draft_graph_forwards"] += 1

        d_logits_after = d_l  # [1, V]

        stats["total_draft"] += K

        # === 阶段 2: target forward [1, K] ===
        t_state_backup = clone_state(t_state)
        draft_tensor = torch.tensor([draft_tokens], device=DEVICE, dtype=torch.long)
        t_logits_full, _ = target.forward(draft_tensor, t_state_backup, return_hidden_layers=[])
        stats["target_forwards"] += 1

        # === 阶段 3: 决策 ===
        accepted = 0
        t_pred = t_logits[0].argmax().item()
        if draft_tokens[0] == t_pred:
            accepted = 1
            out.append(draft_tokens[0])
            stats["accepted"] += 1
            for i in range(1, K):
                t_pred = t_logits_full[0, i-1].argmax().item()
                if draft_tokens[i] == t_pred:
                    accepted += 1
                    out.append(draft_tokens[i])
                    stats["accepted"] += 1
                else:
                    out.append(t_pred)
                    stats["rejected"] += 1
                    break
        else:
            out.append(t_pred)
            stats["rejected"] += 1

        # === 阶段 4: 同步 state ===
        if accepted == K:
            t_state = t_state_backup
            t_logits = t_logits_full[:, -1, :]
            d_logits = d_logits_after
            # d_state 已正确(graph forward in-place 更新)
        else:
            # 拒绝:target replay
            replay = draft_tokens[:accepted] + [out[-1]]
            replay_tensor = torch.tensor([replay], device=DEVICE, dtype=torch.long)
            t_logits_replay, _ = target.forward(replay_tensor, t_state, return_hidden_layers=[])
            t_logits = t_logits_replay[:, -1, :]
            stats["target_forwards"] += 1

            # draft:回退 state + replay(普通 forward,因为 T>1)
            d_logits_replay, _ = draft.forward(replay_tensor, d_state_backup, [])
            d_logits = d_logits_replay[:, -1, :]
            stats["draft_forwards"] += 1
            # d_state_backup 现在是 S_{j+1}(replay 后的正确状态)
            # 把它拷贝到 graph 的 static_state,后续 graph replay 基于此状态
            draft.restore_state(d_state_backup)

        if out[-1] == 0 or len(out) - len(ids) >= n_generate:
            break

    return out, stats


def benchmark_baseline(target, tokenizer, prompt, n_generate=40):
    ids = tokenizer.encode(prompt)
    out = list(ids)
    state = target.zero_state(1)
    ctx_tensor = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    logits, _ = target.forward(ctx_tensor, state, return_hidden_layers=[])
    forwards = 1
    while len(out) - len(ids) < n_generate:
        next_tok = logits[0, -1].argmax().item()
        out.append(next_tok)
        next_t = torch.tensor([[next_tok]], device=DEVICE, dtype=torch.long)
        logits, _ = target.forward(next_t, state, return_hidden_layers=[])
        forwards += 1
        if next_tok == 0:
            break
    return out, {"target_forwards": forwards}


def main():
    print("=" * 70)
    print("DSpark → RWKV · 阶段13: CUDA Graph 加速 draft")
    print("  新方案:target CUDA + draft CUDA + CUDA Graph")
    print("  0.4B g1d draft + 2.9B target")
    print("=" * 70)

    tokenizer = TRIE_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))
    print("\n加载 target 2.9B...")
    target = RWKV7Target2p9BCuda(TARGET_WEIGHTS)
    print("加载 draft 0.4B...")
    draft = DraftGraphWrapper(DRAFT_WEIGHTS)
    print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # === 简单 prompt 测试 ===
    simple_prompts = [
        "什么是人工智能",
        "中国的首都是哪里",
        "水的化学式是什么",
        "太阳从哪个方向升起",
        "月亮是什么形状",
        "人类有几只手",
        "猫是动物吗",
        "苹果是什么颜色的",
    ]

    N_SIMPLE = 8
    print(f"\n[阶段 1: 简单中文 prompt, {N_SIMPLE} 条]")
    configs = [
        {"K": 4, "alpha": 2.0},
    ]

    for cfg in configs:
        K, alpha = cfg["K"], cfg["alpha"]
        print(f"\n--- K={K}, α={alpha} ---")
        total_accepted = 0
        total_draft = 0
        total_forwards = 0
        total_graph = 0
        total_replay = 0
        t0 = time.time()
        for prompt in simple_prompts[:N_SIMPLE]:
            out, stats = speculative_decode(draft, target, tokenizer, prompt,
                                            n_generate=40, K=K, alpha=alpha)
            total_accepted += stats["accepted"]
            total_draft += stats["total_draft"]
            total_forwards += stats["target_forwards"]
            total_graph += stats.get("draft_graph_forwards", 0)
            total_replay += stats.get("draft_forwards", 0)
        elapsed = time.time() - t0
        accept_rate = total_accepted / max(total_draft, 1)
        avg_forwards = total_forwards / N_SIMPLE
        print(f"  接受率: {accept_rate:.4f} ({total_accepted}/{total_draft})")
        print(f"  平均 target forwards: {avg_forwards:.1f}")
        print(f"  draft graph forwards: {total_graph} (T=1,用 graph)")
        print(f"  draft replay forwards: {total_replay} (T>1,普通 forward)")
        print(f"  平均每条 {elapsed/N_SIMPLE:.2f}s")

    # === LAMBADA 基线 ===
    print(f"\n[阶段 2: LAMBADA 测试]")
    N_BASELINE = 10
    N_SPEC = 10
    with open(LAMBADA_FILE, "r", encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f]

    print(f"\n[基线: 纯 target, {N_BASELINE} 条]")
    t0 = time.time()
    total_forwards_base = 0
    for text in texts[:N_BASELINE]:
        _, stats = benchmark_baseline(target, tokenizer, text, n_generate=40)
        total_forwards_base += stats["target_forwards"]
    baseline_time = (time.time() - t0) / N_BASELINE
    avg_forwards_base = total_forwards_base / N_BASELINE
    print(f"  平均每条 {baseline_time:.2f}s, target forwards: {avg_forwards_base:.1f}")

    # target sanity check
    print(f"\n[target sanity check: 纯 target 生成]")
    for prompt in simple_prompts[:3]:
        out, _ = benchmark_baseline(target, tokenizer, prompt, n_generate=40)
        gen = tokenizer.decode(out[len(tokenizer.encode(prompt)):])
        print(f"  {prompt}")
        print(f"    → {gen[:80]}")

    # Speculative decoding
    for cfg in configs:
        K, alpha = cfg["K"], cfg["alpha"]
        print(f"\n[Speculative K={K}, α={alpha}, {N_SPEC} 条]")
        total_accepted = 0
        total_draft = 0
        total_forwards = 0
        t0 = time.time()
        for text in texts[:N_SPEC]:
            out, stats = speculative_decode(draft, target, tokenizer, text,
                                            n_generate=40, K=K, alpha=alpha)
            total_accepted += stats["accepted"]
            total_draft += stats["total_draft"]
            total_forwards += stats["target_forwards"]
        elapsed = time.time() - t0
        accept_rate = total_accepted / max(total_draft, 1)
        avg_forwards = total_forwards / N_SPEC
        avg_time = elapsed / N_SPEC
        print(f"  接受率: {accept_rate:.4f} ({total_accepted}/{total_draft})")
        print(f"  平均 target forwards: {avg_forwards:.1f} (vs 基线 {avg_forwards_base:.1f})")
        print(f"  forward 减少: {(1-avg_forwards/avg_forwards_base)*100:.1f}%")
        print(f"  平均每条 {avg_time:.2f}s (vs 基线 {baseline_time:.2f}s)")
        print(f"  时间加速比: {baseline_time/avg_time:.2f}x")

    # 示例输出
    print(f"\n[示例输出]")
    for prompt in simple_prompts[:3]:
        out, _ = speculative_decode(draft, target, tokenizer, prompt, n_generate=40, K=4, alpha=2.0)
        gen = tokenizer.decode(out[len(tokenizer.encode(prompt)):])
        print(f"  {prompt}")
        print(f"    → {gen[:80]}")


if __name__ == "__main__":
    main()
