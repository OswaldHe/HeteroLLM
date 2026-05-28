"""TileLang fused-MoE kernels for DeepSeek V3.2 on AMD MI210 (gfx90a).

Two HIP kernels:
  1. ``moe_shared_kernel``  — fused (W_gate, W_up, SiLU * up, W_down) for the
     shared expert (dense MLP applied to every token).
  2. ``moe_routed_kernel``  — grouped GEMM that dispatches sorted-by-expert
     tokens through their assigned routed experts (W_gate, W_up, SiLU*up,
     W_down) and multiplies the output by the per-token routing weight.

Adapted from tile-ai/tilelang's ``examples/fusedmoe/example_fusedmoe_tilelang``
and tuned for DeepSeek V3.2 (dim=7168, inter=2048, 256 routed experts, top-8).
All datatypes are fp16 — MI210 has no FP8.
"""

import math
from typing import Tuple

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# Shared expert: dense fused gated MLP applied to every token.
# ---------------------------------------------------------------------------
@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True}, target="hip")
def moe_shared_kernel(
    d_hidden: int,
    d_inter: int,
    num_tokens: int,
    dtype: str = "float16",
    block_token: int = 64,
    block_dhidden: int = 64,
    block_dinter: int = 64,
    threads: int = 256,
    num_stages: int = 1,
):
    """Shared-expert kernel:  out[t,h] = (SiLU(x@Wg^T) * x@Wu^T) @ Wd^T.

    Two sub-kernels exchange ``up_logits[num_tokens, d_inter]`` through HBM.
    """
    accum_dtype = "float"
    log2_e = 1.44269504  # SiLU via exp2

    @T.prim_func
    def kernel(
        x:       T.Tensor((num_tokens, d_hidden), dtype),
        W_gate:  T.Tensor((d_inter,  d_hidden), dtype),
        W_up:    T.Tensor((d_inter,  d_hidden), dtype),
        W_down:  T.Tensor((d_hidden, d_inter ), dtype),
        up_logits: T.Tensor((num_tokens, d_inter), dtype),
        out:     T.Tensor((num_tokens, d_hidden), dtype),
    ):
        # Stage 1: gate & up projections, fuse with SiLU * up
        with T.Kernel(
            T.ceildiv(num_tokens, block_token),
            T.ceildiv(d_inter,    block_dinter),
            threads=threads,
        ) as (bx, by):
            x_shared      = T.alloc_shared((block_token,  block_dhidden), dtype)
            Wg_shared     = T.alloc_shared((block_dinter, block_dhidden), dtype)
            Wu_shared     = T.alloc_shared((block_dinter, block_dhidden), dtype)
            gate_local    = T.alloc_fragment((block_token, block_dinter), accum_dtype)
            up_local      = T.alloc_fragment((block_token, block_dinter), accum_dtype)

            T.use_swizzle(8)
            T.clear(gate_local)
            T.clear(up_local)

            for k in T.Pipelined(T.ceildiv(d_hidden, block_dhidden), num_stages=num_stages):
                T.copy(x[bx * block_token,    k * block_dhidden], x_shared)
                T.copy(W_gate[by * block_dinter, k * block_dhidden], Wg_shared)
                T.copy(W_up  [by * block_dinter, k * block_dhidden], Wu_shared)
                T.gemm(x_shared, Wg_shared, gate_local, transpose_B=True)
                T.gemm(x_shared, Wu_shared, up_local,   transpose_B=True)

            for i, j in T.Parallel(block_token, block_dinter):
                gate_local[i, j] = gate_local[i, j] * (
                    1.0 / (1.0 + T.exp2(-gate_local[i, j] * log2_e))
                )
                up_local[i, j] = up_local[i, j] * gate_local[i, j]

            T.copy(up_local, up_logits[bx * block_token, by * block_dinter])

        # Stage 2: down projection
        with T.Kernel(
            T.ceildiv(num_tokens, block_token),
            T.ceildiv(d_hidden,   block_dhidden),
            threads=threads,
        ) as (bx, by):
            up_shared    = T.alloc_shared((block_token,   block_dinter), dtype)
            Wd_shared    = T.alloc_shared((block_dhidden, block_dinter), dtype)
            out_local    = T.alloc_fragment((block_token, block_dhidden), accum_dtype)

            T.use_swizzle(8)
            T.clear(out_local)

            for k in T.Pipelined(T.ceildiv(d_inter, block_dinter), num_stages=num_stages):
                T.copy(up_logits[bx * block_token,   k * block_dinter], up_shared)
                T.copy(W_down  [by * block_dhidden, k * block_dinter], Wd_shared)
                T.gemm(up_shared, Wd_shared, out_local, transpose_B=True)

            T.copy(out_local, out[bx * block_token, by * block_dhidden])

    return kernel


# ---------------------------------------------------------------------------
# Routed experts: grouped GEMM (M-axis grouped) + SiLU*up + down + weight scale
# ---------------------------------------------------------------------------
@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True}, target="hip")
def moe_routed_kernel(
    d_hidden: int,
    d_inter: int,
    n_routed_experts: int,
    group_sum: int,        # = total dispatched tokens (== L * topk after gate)
    M_blocks: int,         # = ceildiv(group_sum, block_token) + n_routed_experts (padded)
    dtype: str = "float16",
    block_token: int = 64,
    block_dhidden: int = 64,
    block_dinter: int = 64,
    threads: int = 256,
    num_stages: int = 1,
):
    accum_dtype = "float"
    log2_e = 1.44269504

    in_shape    = (group_sum, d_hidden)
    inter_shape = (group_sum, d_inter)
    Wg_shape    = (n_routed_experts, d_inter,  d_hidden)
    Wu_shape    = (n_routed_experts, d_inter,  d_hidden)
    Wd_shape    = (n_routed_experts, d_hidden, d_inter )

    @T.prim_func
    def kernel(
        x:                 T.Tensor(in_shape, dtype),
        W_gate:            T.Tensor(Wg_shape, dtype),
        W_up:              T.Tensor(Wu_shape, dtype),
        W_down:            T.Tensor(Wd_shape, dtype),
        token_weights:     T.Tensor((group_sum,), dtype),
        group_sizes:       T.Tensor((n_routed_experts,), "int32"),
        group_offsets:     T.Tensor((n_routed_experts,), "int32"),
        group_pad_offsets: T.Tensor((n_routed_experts,), "int32"),
        group_idx_for_bx:  T.Tensor((M_blocks,),         "int32"),
        up_logits:         T.Tensor(inter_shape, dtype),
        out:               T.Tensor(in_shape,   dtype),
    ):
        # Stage 1: gate / up / fuse SiLU
        with T.Kernel(
            M_blocks,
            T.ceildiv(d_inter, block_dinter),
            threads=threads,
        ) as (bx, by):
            x_shared   = T.alloc_shared((block_token,  block_dhidden), dtype)
            Wg_shared  = T.alloc_shared((block_dinter, block_dhidden), dtype)
            Wu_shared  = T.alloc_shared((block_dinter, block_dhidden), dtype)
            gate_local = T.alloc_fragment((block_token, block_dinter), accum_dtype)
            up_local   = T.alloc_fragment((block_token, block_dinter), accum_dtype)

            T.use_swizzle(8)

            m_start_padded = bx * block_token
            cur_group = group_idx_for_bx[bx]
            cur_size  = group_sizes[cur_group]
            m_start   = m_start_padded - group_pad_offsets[cur_group] + group_offsets[cur_group]
            actual_rows = T.max(
                0,
                T.min(block_token, cur_size - (m_start_padded - group_pad_offsets[cur_group])),
            )

            T.clear(gate_local)
            T.clear(up_local)

            for k in T.Pipelined(T.ceildiv(d_hidden, block_dhidden), num_stages=num_stages):
                T.copy(
                    x[m_start : m_start + block_token,
                      k * block_dhidden : (k + 1) * block_dhidden],
                    x_shared,
                )
                T.copy(
                    W_gate[cur_group,
                           by * block_dinter : (by + 1) * block_dinter,
                           k  * block_dhidden : (k + 1) * block_dhidden],
                    Wg_shared,
                )
                T.gemm(x_shared, Wg_shared, gate_local, transpose_B=True)
                T.copy(
                    W_up[cur_group,
                         by * block_dinter : (by + 1) * block_dinter,
                         k  * block_dhidden : (k + 1) * block_dhidden],
                    Wu_shared,
                )
                T.gemm(x_shared, Wu_shared, up_local, transpose_B=True)

            for i, j in T.Parallel(block_token, block_dinter):
                gate_local[i, j] = gate_local[i, j] * (
                    1.0 / (1.0 + T.exp2(-gate_local[i, j] * log2_e))
                )
                up_local[i, j] = up_local[i, j] * gate_local[i, j]

            for i, j in T.Parallel(block_token, block_dinter):
                if i < actual_rows:
                    up_logits[m_start + i, by * block_dinter + j] = up_local[i, j]

        # Stage 2: down projection + per-token routing-weight scale
        with T.Kernel(
            M_blocks,
            T.ceildiv(d_hidden, block_dhidden),
            threads=threads,
        ) as (bx, by):
            up_shared = T.alloc_shared((block_token,   block_dinter), dtype)
            Wd_shared = T.alloc_shared((block_dhidden, block_dinter), dtype)
            out_local = T.alloc_fragment((block_token, block_dhidden), accum_dtype)

            T.use_swizzle(8)

            m_start_padded = bx * block_token
            cur_group = group_idx_for_bx[bx]
            cur_size  = group_sizes[cur_group]
            m_start   = m_start_padded - group_pad_offsets[cur_group] + group_offsets[cur_group]
            actual_rows = T.max(
                0,
                T.min(block_token, cur_size - (m_start_padded - group_pad_offsets[cur_group])),
            )

            T.clear(out_local)

            for k in T.Pipelined(T.ceildiv(d_inter, block_dinter), num_stages=num_stages):
                T.copy(
                    up_logits[m_start : m_start + block_token,
                              k * block_dinter : (k + 1) * block_dinter],
                    up_shared,
                )
                T.copy(
                    W_down[cur_group,
                           by * block_dhidden : (by + 1) * block_dhidden,
                           k  * block_dinter  : (k + 1) * block_dinter],
                    Wd_shared,
                )
                T.gemm(up_shared, Wd_shared, out_local, transpose_B=True)

            for i, j in T.Parallel(block_token, block_dhidden):
                if i < actual_rows:
                    out[m_start + i, by * block_dhidden + j] = (
                        out_local[i, j] * token_weights[m_start + i]
                    )

    return kernel


# ---------------------------------------------------------------------------
# Per-block group descriptors for the routed kernel (built in PyTorch).
# ---------------------------------------------------------------------------
def build_group_descriptors(
    counts: list[int],
    block_token: int,
    n_routed_experts: int,
) -> Tuple[list[int], list[int], list[int], list[int], int]:
    """Returns (group_sizes, group_offsets, group_pad_offsets,
    group_idx_for_bx, M_blocks).

    Same scheme as upstream tilelang fusedmoe: each expert's block range is
    rounded up to ``block_token``; ``group_idx_for_bx`` records which expert
    each padded block index belongs to.
    """
    assert len(counts) == n_routed_experts
    group_sizes   = list(counts)
    group_offsets = [0] * n_routed_experts
    for i in range(1, n_routed_experts):
        group_offsets[i] = group_offsets[i - 1] + counts[i - 1]

    group_pad_offsets = [0] * n_routed_experts
    for i in range(1, n_routed_experts):
        group_pad_offsets[i] = group_pad_offsets[i - 1] + math.ceil(
            (counts[i - 1] + 1) / block_token
        ) * block_token

    M_blocks = math.ceil(sum(counts) / block_token) + n_routed_experts
    group_idx_for_bx = [0] * M_blocks
    for bx in range(M_blocks):
        m_start_padded = bx * block_token
        for e in range(n_routed_experts):
            if m_start_padded >= group_pad_offsets[e]:
                group_idx_for_bx[bx] = e

    return (
        group_sizes,
        group_offsets,
        group_pad_offsets,
        group_idx_for_bx,
        M_blocks,
    )
