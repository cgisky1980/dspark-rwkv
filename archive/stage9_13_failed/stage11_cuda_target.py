"""DSpark → RWKV · 阶段11: CUDA 加速 target(抄 Albatross faster3a)

直接 copy 参考/Albatross/faster3a_2605/rwkv7_fast_v3a.py 的实现,做最小改造:
- 类名改为 RWKV7Target2p9BCuda,与 stage9 对齐
- forward(tokens, state, return_hidden_layers=[]) 返回 (logits, hidden_layers)
  - 保持 [B, T, V] shape
  - return_hidden_layers 为空时返回空 list
- zero_state(B) 保持 stage9 接口

关键:CUDA kernel `wkv_seq_w0` 替代 Python for 循环,消除 T 维度的串行开销
"""
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

HEAD_SIZE = 64
DTYPE = torch.float16
THIS_DIR = Path(__file__).resolve().parent
CUDA_DIR = THIS_DIR / "cuda"

# 全局状态(对齐 v3a 的全局变量)
L, C, H, N, V = 0, 0, 0, HEAD_SIZE, 0
WKV_MODE = "fp32io16"
EMB_DEVICE = "cuda"
RKV_MODE = "off"
CMIX_SPARSE = "no-fc"
LOWRANK_WEIGHT = "both"
ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
PP_DEVICES: list[int] = []
LOWRANK_SUFFIXES = ("att.w1", "att.w2", "att.a1", "att.a2", "att.g1", "att.g2", "att.v1", "att.v2")
LOWRANK_IN_ROWS_T = 7
LOWRANK_OUT_ROWS_T = 4
LOWRANK_FUSED_MIN_C = 1024
CMIX_NOFC_ROW20_MAX_T = 5
CMIX_NOFC_T512_MIN_ROWS = 8
LN1_TMIX_FUSE = True
CMIX_B1T1_SPARSE = "b1t1_sparse"
CMIX_ROWS2_SPARSE = "rows2_sparse"
CMIX_B1T1_NOFC = "b1t1_nofc"
CMIX_ROWS2_NOFC = "rows2_nofc"
CMIX_DENSE = "dense"

# 默认 target 权重(stage9 的路径)
TARGET_WEIGHTS_DEFAULT = Path(r"C:\work\niceui\rwkv7-g1h_preview4533-2.9b-20260623-ctx8192.pth.pth")


def log(message: str) -> None:
    print(f"[stage11_cuda_target] {message}", flush=True)


def cuda_mem() -> str:
    if not torch.cuda.is_available():
        return "cuda=unavailable"
    free, total = torch.cuda.mem_get_info()
    used = total - free
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return f"gpu_mem used={used/2**30:.2f}GiB allocated={allocated/2**30:.2f}GiB reserved={reserved/2**30:.2f}GiB total={total/2**30:.2f}GiB"


def sync_all() -> None:
    if PP_DEVICES:
        for dev_id in PP_DEVICES:
            torch.cuda.synchronize(dev_id)
    else:
        torch.cuda.synchronize()


def pp_enabled() -> bool:
    return len(PP_DEVICES) > 1


def first_device() -> torch.device:
    return torch.device(f"cuda:{PP_DEVICES[0]}") if PP_DEVICES else torch.device("cuda")


def last_device() -> torch.device:
    return torch.device(f"cuda:{PP_DEVICES[-1]}") if PP_DEVICES else torch.device("cuda")


def layer_device_index(layer: int) -> int:
    if not pp_enabled():
        return 0
    return min(len(PP_DEVICES) - 1, layer * len(PP_DEVICES) // L)


def layer_device(layer: int) -> torch.device:
    return torch.device(f"cuda:{PP_DEVICES[layer_device_index(layer)]}") if PP_DEVICES else torch.device("cuda")


def pp_segments() -> list[tuple[int, int]]:
    if not pp_enabled():
        return [(0, L)]
    out = []
    start = 0
    while start < L:
        idx = layer_device_index(start)
        end = start + 1
        while end < L and layer_device_index(end) == idx:
            end += 1
        out.append((start, end))
        start = end
    return out


def key_device(key: str) -> torch.device:
    if key == "head.weight" or key.startswith("ln_out."):
        return last_device()
    if key == "emb.weight" or key.startswith("blocks.0.ln0."):
        return first_device()
    parts = key.split(".")
    if len(parts) > 2 and parts[0] == "blocks":
        return layer_device(int(parts[1]))
    return first_device()


class PathConfig:
    __slots__ = ("rows", "use_batched_rkv", "cmix_mode")

    def __init__(self, rows: int, use_batched_rkv: bool, cmix_mode: str):
        self.rows = rows
        self.use_batched_rkv = use_batched_rkv
        self.cmix_mode = cmix_mode


def select_path(B: int, T: int) -> PathConfig:
    rows = B * T
    if CMIX_SPARSE == "off":
        cmix_mode = CMIX_DENSE
    elif CMIX_SPARSE == "no-fc":
        use_nofc = rows <= cmix_nofc_max_rows() or (rows == 20 and T <= cmix_nofc_row20_max_t())
        cmix_mode = CMIX_B1T1_NOFC if rows == 1 else (CMIX_ROWS2_NOFC if use_nofc else CMIX_DENSE)
    elif rows == 1:
        cmix_mode = CMIX_B1T1_SPARSE
    elif rows == 2:
        cmix_mode = CMIX_ROWS2_NOFC
    else:
        cmix_mode = CMIX_DENSE
    if RKV_MODE == "auto":
        use_batched_rkv = (rows == 1) or (4 <= rows <= 64)
    elif RKV_MODE == "on":
        use_batched_rkv = True
    else:
        use_batched_rkv = False
    if use_orig_linear("att_c2c"):
        use_batched_rkv = False
    return PathConfig(rows=rows, use_batched_rkv=use_batched_rkv, cmix_mode=cmix_mode)


def cmix_nofc_max_rows() -> int:
    return 19


def cmix_nofc_row20_max_t() -> int:
    return CMIX_NOFC_ROW20_MAX_T


def use_orig_linear(group: str) -> bool:
    return group in ORIG_LINEAR_GROUPS


def is_lowrank_weight(key: str) -> bool:
    return key.endswith(LOWRANK_SUFFIXES)


def can_use_lowrank_fused(rows: int) -> bool:
    return C >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_IN_ROWS_T


def can_use_lowrank_out_fused(rows: int) -> bool:
    return C >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_OUT_ROWS_T


def is_att_c2c_weight(key: str) -> bool:
    return ".att." in key and key.endswith(("receptance.weight", "key.weight", "value.weight", "output.weight"))


def is_orig_linear_weight(key: str) -> bool:
    return (
        (use_orig_linear("att_c2c") and is_att_c2c_weight(key))
        or (use_orig_linear("ffn_key") and ".ffn.key.weight" in key)
        or (use_orig_linear("head") and key == "head.weight")
    )


def load_extensions(wkv_mode: str = "fp16") -> None:
    t0 = time.perf_counter()
    log(f"loading CUDA extensions v3a_ops + fast_ops + wkv={wkv_mode}")
    cuda_flags = ["-O3", "--use_fast_math", "--extra-device-vectorization"] + ([] if os.name == "nt" else ["-Xptxas", "-O3"])
    # Windows MSVC link.exe 不认 -l 前缀,直接传 .lib 文件名
    extra_ldflags = ["cublas.lib", "cublasLt.lib"] if os.name == "nt" else None
    load(name="rwkv7_v3a_ops", sources=[str(CUDA_DIR / "rwkv7_v3a_ops.cpp"), str(CUDA_DIR / "rwkv7_v3a_ops.cu")], is_python_module=False, verbose=False, extra_cflags=["-O3"], extra_cuda_cflags=cuda_flags, extra_ldflags=extra_ldflags)
    load(name="rwkv7_fast_ops_fp16", sources=[str(CUDA_DIR / "rwkv7_fast_ops_fp16.cpp"), str(CUDA_DIR / "rwkv7_fast_ops_fp16.cu")], is_python_module=False, verbose=False, extra_cflags=["-O3"], extra_cuda_cflags=cuda_flags, extra_ldflags=extra_ldflags)
    if wkv_mode == "fp16":
        load(name="rwkv7_wkv_fp16_v2", sources=[str(CUDA_DIR / "rwkv7_wkv_fp16_v2.cpp"), str(CUDA_DIR / "rwkv7_wkv_fp16_v2.cu")], is_python_module=False, verbose=False, extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-res-usage", "--extra-device-vectorization", "-Xptxas", "-O3"], extra_ldflags=extra_ldflags)
    elif wkv_mode == "fp32io16":
        load(name="rwkv7_wkv_fp32_v2", sources=[str(CUDA_DIR / "rwkv7_wkv_fp32_v2.cpp"), str(CUDA_DIR / "rwkv7_wkv_fp32_v2.cu")], is_python_module=False, verbose=False, extra_cflags=["-O3", "-D_IO_FP16_"], extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas", "-O3", "-D_IO_FP16_"], extra_ldflags=extra_ldflags)
    else:
        raise ValueError(f"unknown wkv_mode: {wkv_mode}")
    log(f"CUDA extensions loaded in {time.perf_counter() - t0:.3f}s")


class RWKV7Target2p9BCuda:
    """RWKV-7 2.9B target,使用 Albatross faster3a 的 CUDA 加速实现。

    与 stage9 的 RWKV7Target2p9B 接口对齐:
      forward(tokens, state, return_hidden_layers=[]) -> (logits [B,T,V], hidden_layers)
      zero_state(B) -> [shift_state, wkv_state, elapsed_t]
    """

    def __init__(self, path=TARGET_WEIGHTS_DEFAULT):
        global L, C, H, N, V
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch._C._jit_set_autocast_mode(False)

        # 加载 CUDA 扩展(只加载一次)
        if not hasattr(RWKV7Target2p9BCuda, "_ops_loaded"):
            load_extensions(WKV_MODE)
            RWKV7Target2p9BCuda._ops_loaded = True

        t0 = time.perf_counter()
        log(f"loading weights from {path}")
        z = torch.load(path, map_location="cpu", mmap=True)
        log(f"weights mmap loaded in {time.perf_counter() - t0:.3f}s tensors={len(z)}")

        H, N = z["blocks.0.att.r_k"].shape
        C, V = H * N, z["emb.weight"].shape[0]
        assert N == HEAD_SIZE
        max_layer = max(int(k.split(".")[1]) for k in z.keys() if k.startswith("blocks."))
        L = max_layer + 1
        log(f"detected model C={C} H={H} N={N} V={V} L={L}")

        emb_src = z["emb.weight"].squeeze()
        ln0_w_src = z["blocks.0.ln0.weight"].squeeze()
        ln0_b_src = z["blocks.0.ln0.bias"].squeeze()
        emb_cpu = emb_src if EMB_DEVICE == "cpu" else None
        t0 = time.perf_counter()
        log(f"moving and preprocessing weights to CUDA emb={EMB_DEVICE}")
        for key in list(z.keys()):
            if key == "emb.weight" and emb_cpu is not None:
                continue
            value = z[key].squeeze()
            dev = key_device(key)
            is_lowrank = is_lowrank_weight(key)
            if ".ffn.key.weight" in key and CMIX_SPARSE == "auto":
                z[key + ".fc"] = value.to(device=dev, dtype=DTYPE).contiguous()
            if (
                not is_lowrank
                and (("key.weight" in key and not is_orig_linear_weight(key))
                or ("value.weight" in key and not is_orig_linear_weight(key))
                or ("receptance.weight" in key and not is_orig_linear_weight(key))
                or ("output.weight" in key and not is_orig_linear_weight(key))
                or ("head.weight" in key and not is_orig_linear_weight(key)))
            ):
                value = value.t()
            value = value.to(device=dev, dtype=DTYPE).contiguous()
            if key.endswith("att.r_k"):
                value = value.flatten().contiguous()
            if is_lowrank:
                if LOWRANK_WEIGHT in ("orig", "both"):
                    z[key] = value
                else:
                    del z[key]
                if LOWRANK_WEIGHT in ("transpose", "both"):
                    z[key + ".t"] = value.t().contiguous()
            else:
                z[key] = value
        emb_dev = first_device()
        ln0_w_bf16 = ln0_w_src.to(device=emb_dev).contiguous()
        ln0_b_bf16 = ln0_b_src.to(device=emb_dev).contiguous()
        if emb_cpu is None:
            with torch.cuda.device(emb_dev):
                z["emb.weight"] = torch.ops.rwkv7_v3a_ops.emb_ln0_bf16_to_f16(
                    emb_src.to(device=emb_dev).contiguous(), ln0_w_bf16, ln0_b_bf16)
        else:
            emb = torch.empty((V, C), dtype=DTYPE, pin_memory=True)
            with torch.cuda.device(emb_dev):
                for start in range(0, V, 4096):
                    end = min(start + 4096, V)
                    chunk = emb_cpu[start:end].to(device=emb_dev).contiguous()
                    chunk = torch.ops.rwkv7_v3a_ops.emb_ln0_bf16_to_f16(chunk, ln0_w_bf16, ln0_b_bf16)
                    emb[start:end].copy_(chunk)
            z["emb.weight"] = emb
        if RKV_MODE != "off" and not use_orig_linear("att_c2c"):
            for layer in range(L):
                p = f"blocks.{layer}.att."
                z[p + "rkv.weight"] = torch.stack((z[p + "receptance.weight"], z[p + "key.weight"], z[p + "value.weight"])).contiguous()
        self.z = z
        self.emb_cpu = EMB_DEVICE == "cpu"
        self.emb_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        sync_all()
        # 保存实例维度(支持同进程多模型:forward 时切换全局变量)
        self.L, self.C, self.H, self.N, self.V = L, C, H, N, V
        log(f"model ready in {time.perf_counter() - t0:.3f}s L={L} C={C} H={H} N={N} V={V}")
        log(cuda_mem())

    def _set_globals(self):
        """切换全局 L/C/H/N/V 为本实例的维度(支持同进程多模型)"""
        global L, C, H, N, V
        L, C, H, N, V = self.L, self.C, self.H, self.N, self.V

    def zero_state(self, B: int):
        self._set_globals()
        if pp_enabled():
            shift = []
            wkv = []
            for layer in range(L):
                dev = layer_device(layer)
                shift.append(torch.zeros((2, B, C), dtype=DTYPE, device=dev))
                wkv.append(torch.zeros((B, H, N, N), dtype=torch.float32 if WKV_MODE == "fp32io16" else DTYPE, device=dev))
            elapsed = [torch.zeros((B,), dtype=torch.int32, device=torch.device(f"cuda:{d}")) for d in PP_DEVICES]
            return [shift, wkv, elapsed]
        return [
            torch.zeros((L, 2, B, C), dtype=DTYPE, device="cuda"),
            torch.zeros((L, B, H, N, N), dtype=torch.float32 if WKV_MODE == "fp32io16" else DTYPE, device="cuda"),
            torch.zeros((B,), dtype=torch.int32, device="cuda"),
        ]

    def forward(self, tokens: torch.Tensor, state, return_hidden_layers=None):
        """与 stage9 接口对齐:返回 (logits [B,T,V], hidden_layers)

        return_hidden_layers: list of layer indices to snapshot,空 list 表示不需要
        """
        self._set_globals()
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        path = select_path(B, T)
        x = self.embed(tokens)
        logits = self.forward_from_x(x, state, path, all_logits=True)
        # forward_from_x 返回 [B, T, V](all_logits=True)
        # 对齐 stage9:只取最后位置的 logits(如果 return_hidden_layers 为空)
        # 实际上 stage9 的 forward 返回 [B, T, V],调用方自己取 [:, -1, :]
        # 所以我们也返回完整 [B, T, V]
        return logits, []  # hidden_layers 暂不实现(stage10 不用)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.emb_cpu:
            if tokens.device != self.z["emb.weight"].device:
                tokens = tokens.to(self.z["emb.weight"].device, non_blocking=True)
            return self.z["emb.weight"][tokens]
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        host, dev = self.emb_cache.get((B, T), (None, None))
        if host is None:
            host = torch.empty((B * T, C), dtype=DTYPE, pin_memory=True)
            dev = torch.empty((B, T, C), dtype=DTYPE, device=first_device())
            self.emb_cache[(B, T)] = (host, dev)
        flat = tokens.reshape(-1)
        if flat.device.type != "cpu":
            flat = flat.cpu()
        torch.index_select(self.z["emb.weight"], 0, flat, out=host)
        dev.copy_(host.view(B, T, C), non_blocking=True)
        return dev

    def forward_from_x(self, x: torch.Tensor, state, path: PathConfig, all_logits: bool = False, last_indices=None) -> torch.Tensor:
        if pp_enabled():
            return self.forward_from_x_pp(x, state, path, all_logits, last_indices)
        z = self.z
        B, T, _ = x.shape
        v_first = x
        xx = self.ln(x, z["blocks.0.ln1.weight"], z["blocks.0.ln1.bias"])
        pre_mix = None

        for layer in range(L):
            p = f"blocks.{layer}."
            xx, v_first = self.tmix(layer, xx, state[0][layer], state[1][layer], state[2], v_first, p + "att.", path, pre_mix)
            pre_mix = None
            if T == 1 and path.cmix_mode not in (CMIX_B1T1_SPARSE, CMIX_ROWS2_SPARSE):
                x, mixed = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16(
                    x.contiguous(), xx.contiguous(), state[0][layer][1], z[p + "ln2.weight"], z[p + "ln2.bias"], z[p + "ffn.x_k"])
                xx = self.cmix_from_mixed(mixed, p + "ffn.", path)
            else:
                x, xx = self.add_ln(x, xx, z[p + "ln2.weight"], z[p + "ln2.bias"])
                xx = self.cmix(xx, state[0][layer], p + "ffn.", path)
            if layer + 1 < L:
                p_next = f"blocks.{layer + 1}."
                if LN1_TMIX_FUSE and B == 1 and T == 1:
                    outs = torch.ops.rwkv7_v3a_ops.add_layer_norm_tmix_mix6_f16(
                        x.contiguous(), xx.contiguous(), state[0][layer + 1][0],
                        z[p_next + "ln1.weight"], z[p_next + "ln1.bias"],
                        z[p_next + "att.x_r"], z[p_next + "att.x_w"], z[p_next + "att.x_k"],
                        z[p_next + "att.x_v"], z[p_next + "att.x_a"], z[p_next + "att.x_g"])
                    x, pre_mix = outs[0], outs[1:]
                    xx = x
                else:
                    x, xx = self.add_ln(x, xx, z[p_next + "ln1.weight"], z[p_next + "ln1.bias"])
            elif not all_logits:
                if last_indices is not None:
                    x = self.ln(self.add(x, xx), z["ln_out.weight"], z["ln_out.bias"])
                    x = x[torch.arange(B, device=x.device), last_indices].contiguous()
                else:
                    x = self.add_last_ln(x, xx, z["ln_out.weight"], z["ln_out.bias"])
                torch.ops.rwkv7_v3a_ops.advance_i32(state[2], T)
                return self.linear_head(x)
            else:
                x = self.add(x, xx)

        x = self.ln(x, z["ln_out.weight"], z["ln_out.bias"])
        torch.ops.rwkv7_v3a_ops.advance_i32(state[2], T)
        return self.linear_head(x)

    def forward_from_x_pp(self, x: torch.Tensor, state, path: PathConfig, all_logits: bool = False, last_indices=None) -> torch.Tensor:
        B, T, _ = x.shape
        v_first = None
        v_first_by_stage: dict[int, torch.Tensor] = {}
        x = x.to(first_device())
        segments = pp_segments()
        for stage, (start, end) in enumerate(segments):
            dev = layer_device(start)
            if x.device != dev:
                x = x.to(dev)
            with torch.cuda.device(dev):
                v_in = None if start == 0 else v_first_by_stage[stage]
                x, v_first = self.forward_pp_segment(x, state, path, start, end, v_in)
            if start == 0 and v_first is not None:
                for next_stage, (next_start, _) in enumerate(segments[1:], 1):
                    next_dev = layer_device(next_start)
                    v_first_by_stage[next_stage] = v_first if next_dev == v_first.device else v_first.to(next_dev)
        with torch.cuda.device(last_device()):
            return self.forward_pp_tail(x, state, T, all_logits, last_indices, advance=True)

    def forward_pp_segment(self, x: torch.Tensor, state, path: PathConfig, start: int, end: int, v_first):
        z = self.z
        B, T, _ = x.shape
        out_v_first = None
        xx = self.ln(x, z[f"blocks.{start}.ln1.weight"], z[f"blocks.{start}.ln1.bias"])
        pre_mix = None
        for layer in range(start, end):
            p = f"blocks.{layer}."
            v_in = x if layer == 0 else v_first
            xx, v_out = self.tmix(layer, xx, state[0][layer], state[1][layer], state[2][layer_device_index(layer)], v_in, p + "att.", path, pre_mix)
            pre_mix = None
            if layer == 0:
                v_first = v_out
                out_v_first = v_out
            if T == 1 and path.cmix_mode not in (CMIX_B1T1_SPARSE, CMIX_ROWS2_SPARSE):
                x, mixed = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16(
                    x.contiguous(), xx.contiguous(), state[0][layer][1], z[p + "ln2.weight"], z[p + "ln2.bias"], z[p + "ffn.x_k"])
                xx = self.cmix_from_mixed(mixed, p + "ffn.", path)
            else:
                x, xx = self.add_ln(x, xx, z[p + "ln2.weight"], z[p + "ln2.bias"])
                xx = self.cmix(xx, state[0][layer], p + "ffn.", path)
            if layer + 1 < end:
                p_next = f"blocks.{layer + 1}."
                if LN1_TMIX_FUSE and B == 1 and T == 1:
                    outs = torch.ops.rwkv7_v3a_ops.add_layer_norm_tmix_mix6_f16(
                        x.contiguous(), xx.contiguous(), state[0][layer + 1][0],
                        z[p_next + "ln1.weight"], z[p_next + "ln1.bias"],
                        z[p_next + "att.x_r"], z[p_next + "att.x_w"], z[p_next + "att.x_k"],
                        z[p_next + "att.x_v"], z[p_next + "att.x_a"], z[p_next + "att.x_g"])
                    x, pre_mix = outs[0], outs[1:]
                    xx = x
                else:
                    x, xx = self.add_ln(x, xx, z[p_next + "ln1.weight"], z[p_next + "ln1.bias"])
            else:
                x = self.add(x, xx)
        return x, out_v_first

    def forward_pp_tail(self, x: torch.Tensor, state, T: int, all_logits: bool = False, last_indices=None, advance: bool = True) -> torch.Tensor:
        B = x.size(0)
        if not all_logits:
            if last_indices is None:
                x = x[:, -1].contiguous()
            else:
                x = x[torch.arange(B, device=x.device), last_indices].contiguous()
        x = self.ln(x, self.z["ln_out.weight"], self.z["ln_out.bias"])
        if advance:
            self.advance_pp_elapsed(state, T)
        return self.linear_head(x)

    def advance_pp_elapsed(self, state, T: int) -> None:
        for idx, dev_id in enumerate(PP_DEVICES):
            with torch.cuda.device(dev_id):
                torch.ops.rwkv7_v3a_ops.advance_i32(state[2][idx], T)

    def ln(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.layer_norm_f16(x.contiguous(), weight, bias)

    def tmix(self, layer: int, x: torch.Tensor, shift_state, wkv_state, elapsed_t, v_first, p: str, path: PathConfig, pre_mix=None):
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = x.shape
        if pre_mix is not None:
            xr, xw, xk, xv, xa, xg = pre_mix
        else:
            xr, xw, xk, xv, xa, xg = ops.tmix_mix6(B, T, C, x.contiguous(), shift_state[0], z[p + "x_r"], z[p + "x_w"], z[p + "x_k"], z[p + "x_v"], z[p + "x_a"], z[p + "x_g"])
        if pre_mix is not None:
            if path.use_batched_rkv:
                flat = torch.stack((xr.reshape(-1, C), xk.reshape(-1, C), xv.reshape(-1, C)))
                rkv = torch.bmm(flat, z[p + "rkv.weight"])
                r, k, v = [t.view(B, T, C) for t in rkv.unbind(0)]
            else:
                r = self.linear_orig_layout(xr, z[p + "receptance.weight"], path, "att_c2c")
                k = self.linear_orig_layout(xk, z[p + "key.weight"], path, "att_c2c")
                v = self.linear_orig_layout(xv, z[p + "value.weight"], path, "att_c2c")
        else:
            if path.use_batched_rkv:
                flat = torch.stack((xr.reshape(-1, C), xk.reshape(-1, C), xv.reshape(-1, C)))
                rkv = torch.bmm(flat, z[p + "rkv.weight"])
                r, k, v = [t.view(B, T, C) for t in rkv.unbind(0)]
            else:
                r = self.linear_orig_layout(xr, z[p + "receptance.weight"], path, "att_c2c")
                k = self.linear_orig_layout(xk, z[p + "key.weight"], path, "att_c2c")
                v = self.linear_orig_layout(xv, z[p + "value.weight"], path, "att_c2c")

        v1 = None
        if LOWRANK_WEIGHT != "orig" and can_use_lowrank_fused(path.rows) and can_use_lowrank_out_fused(path.rows) and layer != 0:
            w1, a1, g1, v1 = torch.ops.rwkv7_v3a_ops.linear_wagv_rank_in_f16(
                xw.contiguous(), xa.contiguous(), xg.contiguous(), xv.contiguous(),
                z[p + "w1.t"], z[p + "a1.t"], z[p + "g1.t"], z[p + "v1.t"])
        elif LOWRANK_WEIGHT != "orig" and can_use_lowrank_fused(path.rows):
            w1, a1, g1 = torch.ops.rwkv7_v3a_ops.linear_wag_rank_in_f16(
                xw.contiguous(), xa.contiguous(), xg.contiguous(), z[p + "w1.t"], z[p + "a1.t"], z[p + "g1.t"])
        else:
            w1 = self.linear_rank_in(xw, z.get(p + "w1"), z.get(p + "w1.t"), path.rows)
            a1 = self.linear_rank_in(xa, z.get(p + "a1"), z.get(p + "a1.t"), path.rows)
            g1 = self.linear_rank_in(xg, z.get(p + "g1"), z.get(p + "g1.t"), path.rows)
        v_done = False
        if LOWRANK_WEIGHT != "orig" and can_use_lowrank_out_fused(path.rows) and layer != 0 and v1 is not None:
            w, a, g, v = torch.ops.rwkv7_v3a_ops.linear_wagv_rank_out_f16(
                w1.contiguous(), a1.contiguous(), g1.contiguous(), v1.contiguous(),
                z[p + "w2.t"], z[p + "a2.t"], z[p + "g2.t"], z[p + "v2.t"],
                v.contiguous(), v_first.contiguous(), z[p + "v0"])
            v_done = True
        elif LOWRANK_WEIGHT != "orig" and can_use_lowrank_out_fused(path.rows):
            w, a, g = torch.ops.rwkv7_v3a_ops.linear_wag_rank_out_f16(
                w1.contiguous(), a1.contiguous(), g1.contiguous(), z[p + "w2.t"], z[p + "a2.t"], z[p + "g2.t"])
        else:
            w = self.linear_rank_out_act(w1, z.get(p + "w2"), z.get(p + "w2.t"), path.rows, 1)
            a = self.linear_rank_out(a1, z.get(p + "a2"), z.get(p + "a2.t"), path.rows)
            g = self.linear_rank_out_act(g1, z.get(p + "g2"), z.get(p + "g2.t"), path.rows, 2)
        k, neg_kk, kka = ops.tmix_kk_a_gate(B, T, C, H, k.contiguous(), z[p + "k_k"], z[p + "a0"], a.contiguous(), z[p + "k_a"])

        if layer == 0:
            v_first = v
        elif not v_done:
            if LOWRANK_WEIGHT != "orig" and can_use_lowrank_out_fused(path.rows):
                if v1 is None:
                    v1 = self.linear_rank_in(xv, z.get(p + "v1"), z.get(p + "v1.t"), path.rows)
                v = torch.ops.rwkv7_v3a_ops.linear_t_vres_f16(v1.contiguous(), z[p + "v2.t"], v.contiguous(), v_first.contiguous(), z[p + "v0"])
            else:
                v12 = self.linear_rank_out(self.linear_rank_in(xv, z.get(p + "v1"), z.get(p + "v1.t"), path.rows), z.get(p + "v2"), z.get(p + "v2.t"), path.rows)
                v = ops.tmix_vres_gate(B, T, C, v.contiguous(), v_first.contiguous(), z[p + "v0"], v12.contiguous())

        y = torch.empty_like(r)
        if WKV_MODE == "fp32io16":
            w_raw = ops.add_vec(C, w.contiguous(), z[p + "w0"])
            torch.ops.rwkv7_wkv_fp32_v2.forward(B, T, C, H, wkv_state, r.contiguous(), w_raw.contiguous(), k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y)
        elif T <= 16:
            torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_w0(B, T, C, H, wkv_state, r.contiguous(), w.contiguous(), z[p + "w0"], k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y, elapsed_t)
        else:
            w_raw = ops.add_vec(C, w.contiguous(), z[p + "w0"])
            torch.ops.rwkv7_wkv_fp16_v2.wkv_seq(B, T, C, H, wkv_state, r.contiguous(), w_raw.contiguous(), k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y, elapsed_t)
        y = ops.tmix_lnx_rkvres_xg(B, T, C, H, y.contiguous(), r.contiguous(), k.contiguous(), v.contiguous(), z[p + "r_k"], z[p + "ln_x.weight"], z[p + "ln_x.bias"], g.contiguous())
        return self.linear_orig_layout(y, z[p + "output.weight"], path, "att_c2c"), v_first

    def cmix(self, x: torch.Tensor, shift_state, p: str, path: PathConfig) -> torch.Tensor:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = x.shape

        if path.cmix_mode == CMIX_B1T1_SPARSE:
            return ops.cmix_sparse_one(C, z[p + "key.weight.fc"].size(0), x.contiguous(), shift_state[1], z[p + "x_k"], z[p + "key.weight.fc"], z[p + "value.weight"])
        if path.cmix_mode == CMIX_ROWS2_SPARSE:
            return ops.cmix_sparse_rows(B, T, C, z[p + "key.weight.fc"].size(0), x.contiguous(), shift_state[1], z[p + "x_k"], z[p + "key.weight.fc"], z[p + "value.weight"])

        mixed = ops.cmix_mix(B, T, C, x.contiguous(), shift_state[1], z[p + "x_k"])
        return self.cmix_from_mixed(mixed, p, path)

    def cmix_from_mixed(self, mixed: torch.Tensor, p: str, path: PathConfig) -> torch.Tensor:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = mixed.shape
        hid = self.linear_orig_layout(mixed, z[p + "key.weight"], path, "ffn_key")
        if path.cmix_mode == CMIX_B1T1_NOFC:
            return ops.cmix_sparse_down_relu_one(C, z[p + "value.weight"].size(0), hid.view(-1).contiguous(), z[p + "value.weight"])
        if path.cmix_mode == CMIX_ROWS2_NOFC:
            F = z[p + "value.weight"].size(0)
            if path.rows >= CMIX_NOFC_T512_MIN_ROWS and C % 512 == 0 and F % 512 == 0:
                return ops.cmix_sparse_down_relu_rows_t512(B, T, C, F, hid.contiguous(), z[p + "value.weight"])
            return ops.cmix_sparse_down_relu_rows(B, T, C, F, hid.contiguous(), z[p + "value.weight"])

        k = ops.relu_square(hid.contiguous())
        return self.linear(k, z[p + "value.weight"])

    def linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if x.numel() == x.size(-1) and weight.size(1) % 64 == 0:
            return torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk(x.contiguous(), weight)
        return torch.ops.rwkv7_v3a_ops.linear_f16(x.contiguous(), weight)

    def linear_head(self, x: torch.Tensor) -> torch.Tensor:
        z = self.z
        if not use_orig_linear("head"):
            return self.linear(x, z["head.weight"])
        rows = x.numel() // C
        return self.linear_orig_layout(x, z["head.weight"], PathConfig(rows, False, CMIX_DENSE), "head")

    def linear_orig_layout(self, x: torch.Tensor, weight: torch.Tensor, path: PathConfig, group: str) -> torch.Tensor:
        if not use_orig_linear(group):
            return self.linear(x, weight)
        if path.rows == 1:
            if group == "ffn_key":
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, True)
                return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, C <= 1024)
            return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, group != "att_c2c" or C < 2048)
        if path.rows == 2:
            if group == "att_c2c":
                return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 64, 2, True)
            if group == "ffn_key":
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, False)
                if C < 4096:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 64, 2, True)
                return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, False)
            if group == "head" and C == 2560:
                return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, False)
            return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 64, 2, True)
        if path.rows == 3:
            if group == "head":
                if C <= 2048:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
                return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(x.contiguous(), weight, 3, 2)
            if group == "ffn_key":
                if C <= 1024:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_cfg_f16(x.contiguous(), weight, 64, 3, 4)
                if C == 2048:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if group == "att_c2c":
                if C == 768:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(x.contiguous(), weight, 1, 2)
                if C == 1024:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(x.contiguous(), weight, 2, 2)
                if C == 2048:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(x.contiguous(), weight, 3, 4)
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(x.contiguous(), weight, 3, 2)
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
            return torch.ops.rwkv7_v3a_ops.linear_orig_rows_cfg_f16(x.contiguous(), weight, 64, 3, 4)
        if path.rows == 4:
            if group == "ffn_key":
                if C <= 1024:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_cfg_f16(x.contiguous(), weight, 64, 2, 4)
                if C == 2048:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if group == "att_c2c":
                if C <= 1024:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(x.contiguous(), weight, 2, 2)
                if C == 2048:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(x.contiguous(), weight, 4, 2)
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(x.contiguous(), weight, 4, 2)
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
        if group == "head":
            if C == 768:
                if 192 <= path.rows < 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 3)
                if 96 <= path.rows < 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
            if C == 1024:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
                if 192 <= path.rows < 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
                if 96 <= path.rows < 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
            if C == 2048:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
                if 192 <= path.rows < 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 6)
                if 128 <= path.rows < 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
                if 96 <= path.rows < 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if C == 2560:
                if path.rows >= 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
                if path.rows >= 192:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 5)
                if path.rows >= 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 5)
                if path.rows >= 128:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
                if path.rows >= 96:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
                if path.rows >= 80:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
                if path.rows >= 72:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
            if path.rows >= 1024:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 0)
            if path.rows >= 512:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
            if path.rows >= 384:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 2)
            if path.rows >= 256:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
            if path.rows >= 192:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 0)
            if path.rows >= 160:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
            if path.rows >= 128:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 0)
            if path.rows >= 112:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
            if path.rows >= 96:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
            if path.rows >= 80:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 2)
            if path.rows >= 72:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 2)
        if group == "att_c2c":
            if C == 2560 and 17 <= path.rows <= 20:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if C == 768:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 1)
                if 96 <= path.rows < 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 3)
            if C == 1024:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 0)
                if 96 <= path.rows < 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 6)
            if C == 2048:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 3)
                if 192 <= path.rows < 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 0)
                if 96 <= path.rows < 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 4)
            if C == 2560:
                if path.rows >= 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
                if path.rows >= 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
                if path.rows >= 128:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 2)
                if path.rows >= 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 3)
                if path.rows >= 96:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 2)
                if path.rows >= 72:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 2)
                if path.rows >= 5:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
            if path.rows >= 1024:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 4)
            if path.rows >= 768:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
            if path.rows >= 512:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
            if path.rows >= 384:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 2)
            if path.rows >= 256:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 4)
            if path.rows >= 192:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if path.rows >= 160:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 1)
            if path.rows >= 112:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
            if path.rows >= 96:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 5)
            if path.rows >= 72:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
            if path.rows >= 48:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 6)
            if path.rows >= 32:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if path.rows >= 24:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 6)
            if path.rows >= 12:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if path.rows >= 5:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
        if group == "ffn_key":
            if C == 2560 and 17 <= path.rows <= 20:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if C == 768:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
                if 96 <= path.rows < 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
            if C == 1024:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 2)
                if 192 <= path.rows < 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
                if 96 <= path.rows < 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 2)
            if C == 2048 and 128 <= path.rows < 160:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 3)
            if C == 2560:
                if path.rows >= 192:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 5)
                if path.rows >= 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 4)
                if path.rows >= 128:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 5)
                if path.rows >= 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 4)
                if path.rows >= 96:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 4)
                if path.rows >= 80:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 3)
                if path.rows >= 72:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 4)
                if path.rows >= 3:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
            if path.rows >= 1024:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if path.rows >= 768:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
            if path.rows >= 512:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 3)
            if path.rows >= 384:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
            if path.rows >= 256:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 4)
            if path.rows >= 192:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
            if path.rows >= 160:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
            if path.rows >= 128:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
            if path.rows >= 112:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 3)
            if path.rows >= 96:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
            if path.rows >= 72:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 1)
            if path.rows >= 48:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
            if path.rows >= 12:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if path.rows in (5, 6):
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
        return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)

    def linear_rank_in(self, x: torch.Tensor, weight: torch.Tensor, weight_t: torch.Tensor, rows: int) -> torch.Tensor:
        if weight_t is not None and rows <= LOWRANK_IN_ROWS_T:
            return torch.ops.rwkv7_v3a_ops.linear_t_f16(x.contiguous(), weight_t)
        return self.linear_lowrank_orig(x, weight) if weight is not None else self.linear_t_orig(x, weight_t)

    def linear_rank_out(self, x: torch.Tensor, weight: torch.Tensor, weight_t: torch.Tensor, rows: int) -> torch.Tensor:
        if weight_t is not None and C >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_OUT_ROWS_T:
            return torch.ops.rwkv7_v3a_ops.linear_t_f16(x.contiguous(), weight_t)
        return self.linear_lowrank_orig(x, weight) if weight is not None else self.linear_t_orig(x, weight_t)

    def linear_rank_out_act(self, x: torch.Tensor, weight: torch.Tensor, weight_t: torch.Tensor, rows: int, act: int) -> torch.Tensor:
        if weight_t is not None and C >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_OUT_ROWS_T:
            return torch.ops.rwkv7_v3a_ops.linear_t_act_f16(x.contiguous(), weight_t, act)
        ops = torch.ops.rwkv7_fast_ops_fp16
        x = ops.act_tanh(x.contiguous()) if act == 1 else ops.act_sigmoid(x.contiguous())
        return self.linear_lowrank_orig(x.contiguous(), weight) if weight is not None else self.linear_t_orig(x, weight_t)

    def linear_lowrank_orig(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.linear_f16(x.contiguous(), weight)

    def linear_t_orig(self, x: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight_t)

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.add_f16(x.contiguous(), y.contiguous())

    def add_ln(self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        outs = torch.ops.rwkv7_v3a_ops.add_layer_norm_f16(x.contiguous(), residual.contiguous(), weight, bias)
        return outs[0], outs[1]

    def add_last_ln(self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.add_last_layer_norm_f16(x.contiguous(), residual.contiguous(), weight, bias)
