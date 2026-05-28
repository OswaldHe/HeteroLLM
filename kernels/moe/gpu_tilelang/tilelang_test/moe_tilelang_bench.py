"""End-to-end DeepSeek V3.2 MoE built on TileLang HIP kernels (MI210).

Validates against the PyTorch reference in deepseek_v32_moe_bench.py and
benchmarks the TileLang implementation.

Usage:
  conda activate rmt
  # tilelang (built from ~/tilelang for ROCm 6.2) needs the system libstdc++
  # because conda's libstdc++.so.6 lacks GLIBCXX_3.4.30. Preload it:
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \\
      python moe_tilelang_bench.py --L 64
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \\
      python moe_tilelang_bench.py --L 128 --iters 100 --skip-check
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import os

# ROCm 6.2's hipBLASLt fails on tiny (n=1) fp32 GEMVs like the gate
# projection (HIPBLAS_STATUS_INTERNAL_ERROR). Force the legacy hipBLAS
# path before importing torch so addmm/matmul fall back to a working
# kernel. Must precede ``import torch``.
os.environ.setdefault("DISABLE_ADDMM_HIP_LT", "1")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLAS_LT", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the reference module importable
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from moe_tilelang_kernel import (  # noqa: E402
    build_group_descriptors,
    moe_routed_kernel,
    moe_shared_kernel,
)
from moe_tilelang_decode import (  # noqa: E402
    moe_routed_decode_stage1_int8_kernel,
    moe_routed_decode_stage1_kernel,
    moe_routed_decode_stage2_int8_kernel,
    moe_routed_decode_stage2_kernel,
    moe_shared_decode_stage1_kernel,
    moe_shared_decode_stage2_kernel,
)


# ---------------------------------------------------------------------------
# DeepSeek V3.2 config — matches deepseek_v32_moe_bench.py
# ---------------------------------------------------------------------------
@dataclass
class ModelArgs:
    dim:               int = 7168
    moe_inter_dim:     int = 2048
    n_routed_experts:  int = 256
    n_shared_experts:  int = 1
    n_activated_experts: int = 8
    n_expert_groups:   int = 8
    n_limited_groups:  int = 4
    route_scale:       float = 2.5
    score_func:        Literal["softmax", "sigmoid"] = "sigmoid"


# ---------------------------------------------------------------------------
# Reference PyTorch implementation (verbatim from deepseek_v32_moe_bench.py)
# ---------------------------------------------------------------------------
class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.topk        = args.n_activated_experts
        self.n_groups    = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func  = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias   = nn.Parameter(torch.zeros(args.n_routed_experts, dtype=torch.float32))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            group_scores  = scores.topk(2, dim=-1)[0].sum(dim=-1)
            group_indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=torch.bool)
            mask.scatter_(1, group_indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(
            (F.silu(self.w1(x).float()) * self.w3(x).float()).to(x.dtype)
        )


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(
            (F.silu(self.w1(x).float()) * self.w3(x).float()).to(x.dtype)
        )


class MoERef(nn.Module):
    """PyTorch reference — used for correctness check only."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim                = args.dim
        self.n_routed_experts   = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.gate    = Gate(args)
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim)
            for _ in range(args.n_routed_experts)
        ])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x, dtype=torch.float32)
        # ROCm 6.2 lacks small-input kernels for both ``bincount`` and
        # ``where`` on this device, so do the per-expert dispatch lookup on
        # CPU. The expert MLP itself stays on GPU.
        indices_cpu = indices.cpu()
        counts = torch.bincount(
            indices_cpu.flatten(), minlength=self.n_routed_experts,
        ).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            idx_cpu, top_cpu = torch.where(indices_cpu == i)
            idx = idx_cpu.to(indices.device)
            top = top_cpu.to(indices.device)
            y[idx] += self.experts[i](x[idx]).float() * weights[idx, top, None]
        # NOTE: shared expert disabled for routed-only benchmark
        return y.to(x.dtype).view(shape)


# ---------------------------------------------------------------------------
# TileLang-backed MoE
# ---------------------------------------------------------------------------
class MoETileLangDecode(nn.Module):
    """L=1 decode-specialized TileLang MoE.

    Four split-K GEMV kernels (no MFMA padding waste):
      * routed stage 1: per (k, di_block) → up_logits[k, di_block]
      * routed stage 2: per dh_block → sum_k W_down[idx[k]] @ up_logits[k] * w[k]
      * shared stage 1: per di_block → up_logits[di_block]
      * shared stage 2: per dh_block → W_down @ up_logits
    Routed and shared writes are then summed in PyTorch.
    """

    def __init__(
        self,
        args: ModelArgs,
        ref: MoERef,
        BLOCK_N: int = 64,
        reduce_threads: int = 64,
    ):
        super().__init__()
        self.args  = args
        self.gate  = ref.gate
        device = torch.device("cuda")
        dtype  = torch.float16

        # ROCm 6.2 fails the very first dispatch of small fp32 elementwise
        # / GEMV kernels ("no kernel image is available"). Run a couple of
        # tiny ops at construction time so the kernel images are cached
        # before the first gate call hits them at L=1.
        with torch.no_grad():
            torch.randn(1, 256, device=device, dtype=torch.float32).sigmoid().sum()
            (torch.randn(1, 16, device=device, dtype=torch.float32) @
             torch.randn(16, 8, device=device, dtype=torch.float32)).sum()

        Wg = torch.stack([ref.experts[i].w1.weight.detach() for i in range(args.n_routed_experts)], dim=0)
        Wu = torch.stack([ref.experts[i].w3.weight.detach() for i in range(args.n_routed_experts)], dim=0)
        Wd = torch.stack([ref.experts[i].w2.weight.detach() for i in range(args.n_routed_experts)], dim=0)
        self.W_routed_gate = Wg.to(device=device, dtype=dtype).contiguous()
        self.W_routed_up   = Wu.to(device=device, dtype=dtype).contiguous()
        self.W_routed_down = Wd.to(device=device, dtype=dtype).contiguous()

        self.W_shared_gate = ref.shared_experts.w1.weight.detach().to(device=device, dtype=dtype).contiguous()
        self.W_shared_up   = ref.shared_experts.w3.weight.detach().to(device=device, dtype=dtype).contiguous()
        self.W_shared_down = ref.shared_experts.w2.weight.detach().to(device=device, dtype=dtype).contiguous()

        d_inter_shared = args.n_shared_experts * args.moe_inter_dim
        self.routed_stage1 = moe_routed_decode_stage1_kernel(
            d_hidden=args.dim, d_inter=args.moe_inter_dim,
            n_routed_experts=args.n_routed_experts,
            topk=args.n_activated_experts,
            BLOCK_N=BLOCK_N, reduce_threads=reduce_threads,
        )
        self.routed_stage2 = moe_routed_decode_stage2_kernel(
            d_hidden=args.dim, d_inter=args.moe_inter_dim,
            n_routed_experts=args.n_routed_experts,
            topk=args.n_activated_experts,
            BLOCK_N=BLOCK_N, reduce_threads=reduce_threads,
        )
        self.shared_stage1 = moe_shared_decode_stage1_kernel(
            d_hidden=args.dim, d_inter=d_inter_shared,
            BLOCK_N=BLOCK_N, reduce_threads=reduce_threads,
        )
        self.shared_stage2 = moe_shared_decode_stage2_kernel(
            d_hidden=args.dim, d_inter=d_inter_shared,
            BLOCK_N=BLOCK_N, reduce_threads=reduce_threads,
        )

        self.up_logits_shared = torch.empty((d_inter_shared,),                              device=device, dtype=dtype)
        self.shared_out       = torch.empty((args.dim,),                                    device=device, dtype=dtype)
        self.up_logits_routed = torch.empty((args.n_activated_experts, args.moe_inter_dim), device=device, dtype=dtype)
        self.routed_out       = torch.empty((args.dim,),                                    device=device, dtype=dtype)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        args = self.args
        orig_shape = x.shape
        x_flat = x.view(-1, args.dim)
        assert x_flat.size(0) == 1, "decode kernels only support L=1"
        x_vec = x_flat.view(-1).contiguous()

        weights, indices = self.gate(x_flat)
        idx_int32 = indices.view(-1).to(torch.int32).contiguous()
        w_fp32    = weights.view(-1).to(torch.float32).contiguous()

        self.routed_stage1(
            x_vec, self.W_routed_gate, self.W_routed_up,
            idx_int32, self.up_logits_routed,
        )
        self.routed_stage2(
            self.W_routed_down, idx_int32, w_fp32,
            self.up_logits_routed, self.routed_out,
        )
        # NOTE: shared expert disabled for routed-only benchmark
        # self.shared_stage1(x_vec, self.W_shared_gate, self.W_shared_up, self.up_logits_shared)
        # self.shared_stage2(self.W_shared_down, self.up_logits_shared, self.shared_out)

        return self.routed_out.to(x_flat.dtype).view(orig_shape)


def _quantize_per_expert_int8(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-expert symmetric int8 quantization for ``W`` of shape (E, *).

    Stays in the input dtype (fp16) for the abs/amax pass — casting the
    entire tensor to fp32 first costs ~14 GiB on DeepSeek V3.2 routed
    weights and OOMs the GPU.

    Returns (W_q [int8, same shape], scale [fp32, (E,)]).
    """
    E = W.size(0)
    amax  = W.reshape(E, -1).abs().amax(dim=1).float().clamp_min(1e-8)  # (E,)
    scale = (amax / 127.0).contiguous()
    inv   = (1.0 / scale).to(W.dtype).view((E,) + (1,) * (W.dim() - 1))
    W_q   = (W * inv).round().clamp(-128, 127).to(torch.int8).contiguous()
    return W_q, scale


def _quantize_per_tensor_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric int8 quantization. Returns (x_q [int8], scale [fp32, (1,)])."""
    amax  = x.abs().amax().float().clamp_min(1e-8)
    scale = (amax / 127.0).reshape(1).contiguous()
    inv   = (1.0 / scale).to(x.dtype)
    x_q   = (x * inv).round().clamp(-128, 127).to(torch.int8).contiguous()
    return x_q, scale


class MoETileLangDecodeInt8(nn.Module):
    """L=1 decode-specialized TileLang MoE with int8 weights and activations.

    Storage layout:
      * routed weights (W_gate / W_up / W_down): int8, per-expert symmetric
        scale (one fp32 scalar per expert, per matrix).
      * x (per-token input): int8, per-call dynamic per-tensor symmetric
        scale.
    All accumulators run in fp32; up_logits stay in fp16 between stages.
    """

    def __init__(
        self,
        args: ModelArgs,
        ref: MoERef,
        BLOCK_N: int = 64,
        reduce_threads: int = 64,
    ):
        super().__init__()
        self.args = args
        self.gate = ref.gate
        device = torch.device("cuda")

        # Stack and quantize routed expert weights (per-expert scale).
        Wg = torch.stack([ref.experts[i].w1.weight.detach() for i in range(args.n_routed_experts)], dim=0)
        Wu = torch.stack([ref.experts[i].w3.weight.detach() for i in range(args.n_routed_experts)], dim=0)
        Wd = torch.stack([ref.experts[i].w2.weight.detach() for i in range(args.n_routed_experts)], dim=0)
        Wg_q, Wg_s = _quantize_per_expert_int8(Wg.to(device))
        Wu_q, Wu_s = _quantize_per_expert_int8(Wu.to(device))
        Wd_q, Wd_s = _quantize_per_expert_int8(Wd.to(device))
        # The kernels read packed int32 views (4 int8 lanes per word) to
        # avoid tilelang's HIP int8 vector-broadcast codegen bug.
        self.W_routed_gate    = Wg_q.view(torch.int32)
        self.W_routed_up      = Wu_q.view(torch.int32)
        self.W_routed_down    = Wd_q.view(torch.int32)
        self.W_routed_gate_s  = Wg_s
        self.W_routed_up_s    = Wu_s
        self.W_routed_down_s  = Wd_s

        # Compile int8 decode kernels (routed only — shared expert disabled
        # in the routed-only correctness path, same as MoETileLangDecode).
        self.routed_stage1 = moe_routed_decode_stage1_int8_kernel(
            d_hidden=args.dim, d_inter=args.moe_inter_dim,
            n_routed_experts=args.n_routed_experts,
            topk=args.n_activated_experts,
            BLOCK_N=BLOCK_N, reduce_threads=reduce_threads,
        )
        self.routed_stage2 = moe_routed_decode_stage2_int8_kernel(
            d_hidden=args.dim, d_inter=args.moe_inter_dim,
            n_routed_experts=args.n_routed_experts,
            topk=args.n_activated_experts,
            BLOCK_N=BLOCK_N, reduce_threads=reduce_threads,
        )

        # Persistent staging buffers.
        self.up_logits_routed = torch.empty(
            (args.n_activated_experts, args.moe_inter_dim),
            device=device, dtype=torch.float16,
        )
        self.routed_out = torch.empty(
            (args.dim,), device=device, dtype=torch.float16,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        args = self.args
        orig_shape = x.shape
        x_flat = x.view(-1, args.dim)
        assert x_flat.size(0) == 1, "decode kernels only support L=1"
        x_vec_fp = x_flat.view(-1).contiguous()

        # Per-call dynamic activation quantization.
        x_q, x_scale = _quantize_per_tensor_int8(x_vec_fp)
        x_q_packed = x_q.view(torch.int32)

        weights, indices = self.gate(x_flat)
        idx_int32 = indices.view(-1).to(torch.int32).contiguous()
        w_fp32    = weights.view(-1).to(torch.float32).contiguous()

        self.routed_stage1(
            x_q_packed, self.W_routed_gate, self.W_routed_up,
            x_scale, self.W_routed_gate_s, self.W_routed_up_s,
            idx_int32, self.up_logits_routed,
        )
        self.routed_stage2(
            self.W_routed_down, self.W_routed_down_s,
            idx_int32, w_fp32,
            self.up_logits_routed, self.routed_out,
        )
        return self.routed_out.to(x_flat.dtype).view(orig_shape)


class MoETileLang(nn.Module):
    """General TileLang-backed MoE (any L).  Used for L > 1."""

    def __init__(
        self,
        args: ModelArgs,
        ref: MoERef,
        num_tokens: int,
        block_token: int = 64,
        block_dhidden: int = 64,
        block_dinter: int = 64,
        threads: int = 256,
    ):
        super().__init__()
        self.args         = args
        self.num_tokens   = num_tokens
        self.block_token  = block_token
        self.block_dh     = block_dhidden
        self.block_di     = block_dinter
        self.threads      = threads
        self.gate         = ref.gate  # PyTorch gate (small op, keep on torch)

        device = torch.device("cuda")
        dtype  = torch.float16

        # Stack per-expert routed weights into [E, K, D] tensors
        Wg_list = [ref.experts[i].w1.weight.detach() for i in range(args.n_routed_experts)]
        Wu_list = [ref.experts[i].w3.weight.detach() for i in range(args.n_routed_experts)]
        Wd_list = [ref.experts[i].w2.weight.detach() for i in range(args.n_routed_experts)]

        # Shapes follow upstream conv: w1/w3 -> (inter, dim); w2 -> (dim, inter)
        # In the kernel we expect transpose_B style:
        #   W_gate[E, d_inter, d_hidden],  W_up[E, d_inter, d_hidden],
        #   W_down[E, d_hidden, d_inter]
        self.W_routed_gate = torch.stack(Wg_list, dim=0).to(device=device, dtype=dtype).contiguous()
        self.W_routed_up   = torch.stack(Wu_list, dim=0).to(device=device, dtype=dtype).contiguous()
        self.W_routed_down = torch.stack(Wd_list, dim=0).to(device=device, dtype=dtype).contiguous()

        # Shared expert
        self.W_shared_gate = ref.shared_experts.w1.weight.detach().to(device=device, dtype=dtype).contiguous()
        self.W_shared_up   = ref.shared_experts.w3.weight.detach().to(device=device, dtype=dtype).contiguous()
        self.W_shared_down = ref.shared_experts.w2.weight.detach().to(device=device, dtype=dtype).contiguous()

        d_inter_shared = args.n_shared_experts * args.moe_inter_dim
        self.shared_kernel = moe_shared_kernel(
            d_hidden=args.dim,
            d_inter=d_inter_shared,
            num_tokens=num_tokens,
            dtype="float16",
            block_token=block_token,
            block_dhidden=block_dhidden,
            block_dinter=block_dinter,
            threads=threads,
        )

        # Routed kernel sized for the worst-case dispatch buffer.
        group_sum = num_tokens * args.n_activated_experts
        M_blocks  = math.ceil(group_sum / block_token) + args.n_routed_experts
        self.group_sum = group_sum
        self.M_blocks  = M_blocks
        self.routed_kernel = moe_routed_kernel(
            d_hidden=args.dim,
            d_inter=args.moe_inter_dim,
            n_routed_experts=args.n_routed_experts,
            group_sum=group_sum,
            M_blocks=M_blocks,
            dtype="float16",
            block_token=block_token,
            block_dhidden=block_dhidden,
            block_dinter=block_dinter,
            threads=threads,
        )

        # Persistent staging buffers
        self.x_dispatch        = torch.empty((group_sum, args.dim),       device=device, dtype=dtype)
        self.tok_weights_disp  = torch.empty((group_sum,),                 device=device, dtype=dtype)
        self.tok_idx_disp      = torch.empty((group_sum,),                 device=device, dtype=torch.int64)
        self.up_logits_routed  = torch.empty((group_sum, args.moe_inter_dim), device=device, dtype=dtype)
        self.routed_out        = torch.empty((group_sum, args.dim),       device=device, dtype=dtype)
        self.up_logits_shared  = torch.empty((num_tokens, d_inter_shared), device=device, dtype=dtype)
        self.shared_out        = torch.empty((num_tokens, args.dim),      device=device, dtype=dtype)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        args = self.args
        orig_shape = x.shape
        x_flat = x.view(-1, args.dim)
        T = x_flat.size(0)
        assert T == self.num_tokens, f"expected {self.num_tokens} tokens, got {T}"

        weights, indices = self.gate(x_flat)               # [T, topk] (fp32)
        flat_idx = indices.view(-1)
        flat_w   = weights.view(-1).to(x_flat.dtype)

        # Sort by expert to build the grouped GEMM input
        order = flat_idx.argsort()
        token_idxs = order // args.n_activated_experts     # which token does each slot belong to

        # Counts per expert (CPU for descriptor build)
        counts_t = torch.bincount(flat_idx, minlength=args.n_routed_experts)
        counts   = counts_t.cpu().tolist()

        # Pack dispatched tokens & per-slot routing weights
        self.x_dispatch.copy_(x_flat[token_idxs])
        self.tok_idx_disp.copy_(token_idxs)
        self.tok_weights_disp.copy_(flat_w[order])

        # Build per-block descriptors for grouped GEMM
        gs, go, gpo, gix, M_blocks = build_group_descriptors(
            counts, self.block_token, args.n_routed_experts,
        )
        assert M_blocks == self.M_blocks, "M_blocks mismatch — recompile kernel for this L"
        device = x.device
        group_sizes       = torch.tensor(gs,  dtype=torch.int32, device=device)
        group_offsets     = torch.tensor(go,  dtype=torch.int32, device=device)
        group_pad_offsets = torch.tensor(gpo, dtype=torch.int32, device=device)
        group_idx_for_bx  = torch.tensor(gix, dtype=torch.int32, device=device)

        # Routed kernel (writes into self.up_logits_routed and self.routed_out)
        self.routed_kernel(
            self.x_dispatch,
            self.W_routed_gate,
            self.W_routed_up,
            self.W_routed_down,
            self.tok_weights_disp,
            group_sizes,
            group_offsets,
            group_pad_offsets,
            group_idx_for_bx,
            self.up_logits_routed,
            self.routed_out,
        )

        # Scatter-reduce dispatched outputs back into per-token rows
        y = torch.zeros_like(x_flat, dtype=torch.float32)
        idx_expanded = self.tok_idx_disp.view(-1, 1).expand(-1, args.dim)
        y.scatter_add_(0, idx_expanded, self.routed_out.float())

        # Shared kernel (always all tokens)
        self.shared_kernel(
            x_flat,
            self.W_shared_gate,
            self.W_shared_up,
            self.W_shared_down,
            self.up_logits_shared,
            self.shared_out,
        )
        y = y + self.shared_out.float()
        return y.to(x_flat.dtype).view(orig_shape)


# ---------------------------------------------------------------------------
# Bench / correctness driver
# ---------------------------------------------------------------------------
def sync():
    torch.cuda.synchronize()


def run_bench(fn, x, warmup, iters):
    for _ in range(warmup):
        fn(x)
    sync()
    times = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn(x)
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
    return sorted(times)


def pct(vals, p):
    return vals[int(len(vals) * p / 100)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L",      type=int, default=64)
    parser.add_argument("--iters",  type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--skip-check", action="store_true",
                        help="skip the (slow) PyTorch reference correctness check")
    parser.add_argument("--block_token",  type=int, default=64)
    parser.add_argument("--block_dhidden", type=int, default=64)
    parser.add_argument("--block_dinter",  type=int, default=64)
    parser.add_argument("--threads",       type=int, default=256)
    parser.add_argument("--decode_block_n",        type=int, default=64,
                        help="L=1 split-K GEMV: output rows per block")
    parser.add_argument("--decode_reduce_threads", type=int, default=16,
                        help="L=1 split-K GEMV: reduction threads per output (BLOCK_N*this <= 1024)")
    parser.add_argument("--dtype", choices=["fp16", "int8"], default="fp16",
                        help="MoE storage / compute dtype (decode path supports both; "
                             "L>1 dense path is fp16-only).")
    parser.add_argument("--int8", dest="dtype", action="store_const", const="int8",
                        help="Shortcut for --dtype int8.")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = ModelArgs()
    device = torch.device("cuda")
    dtype  = torch.float16

    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"Tokens L : {args.L}")
    print(f"Dtype    : {args.dtype}")
    print(f"Block    : token={args.block_token} dh={args.block_dhidden} di={args.block_dinter} threads={args.threads}")
    print()

    print("Building reference MoE on GPU (random weights)...")
    ref = MoERef(cfg).to(device=device, dtype=dtype)
    ref.eval()

    print("Building TileLang MoE wrapper...")
    if args.L == 1:
        if args.dtype == "int8":
            moe = MoETileLangDecodeInt8(
                cfg, ref,
                BLOCK_N=args.decode_block_n,
                reduce_threads=args.decode_reduce_threads,
            )
        else:
            moe = MoETileLangDecode(
                cfg, ref,
                BLOCK_N=args.decode_block_n,
                reduce_threads=args.decode_reduce_threads,
            )
    else:
        if args.dtype == "int8":
            raise NotImplementedError(
                "int8 path is currently only implemented for the L=1 decode kernels"
            )
        moe = MoETileLang(
            cfg, ref, num_tokens=args.L,
            block_token=args.block_token,
            block_dhidden=args.block_dhidden,
            block_dinter=args.block_dinter,
            threads=args.threads,
        )

    x = torch.randn(args.L, cfg.dim, device=device, dtype=dtype)

    if not args.skip_check:
        print("Correctness check vs PyTorch reference...")
        with torch.no_grad():
            if args.L == 1:
                # ROCm 6.2 lacks small-tensor kernels for advanced indexing
                # / scatter, so run the reference on CPU at L=1. Move ref
                # in place rather than deep-copying — fp16 mode's weights
                # are large enough that a GPU-side deepcopy OOMs the 64 GB
                # MI210. moe holds its own weight copies plus the gate
                # reference, so we restore ref to GPU before calling moe.
                ref.cpu()
                y_ref = ref(x.cpu()).float().to(device)
                ref.to(device)
            else:
                y_ref = ref(x).float()
            y_til = moe(x).float()
        max_abs = (y_ref - y_til).abs().max().item()
        ref_max = y_ref.abs().max().item()
        rel = max_abs / max(ref_max, 1e-6)
        print(f"  max abs err = {max_abs:.4f}   (ref max = {ref_max:.4f}, rel = {rel:.2e})")

    print("\nBenchmark (TileLang MoE)...")
    times = run_bench(lambda z: moe(z), x, args.warmup, args.iters)
    mean = sum(times) / len(times)
    print(f"  iters={args.iters}  min={times[0]:.3f} ms  P50={pct(times,50):.3f}  P90={pct(times,90):.3f}  max={times[-1]:.3f}  mean={mean:.3f} ms")
    print(f"  throughput = {args.L / (mean * 1e-3):.1f} tokens/sec")


if __name__ == "__main__":
    main()
