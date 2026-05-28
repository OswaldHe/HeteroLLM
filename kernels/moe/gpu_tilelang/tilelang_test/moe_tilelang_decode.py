"""Decode-specialized TileLang MoE for DeepSeek V3.2 on AMD MI210 (gfx90a).

L=1 specialization. Each kernel is structured as a split-K GEMV with
warp-level reduction (T.tvm_thread_allreduce) — no MFMA, no MFMA-min-M
padding waste.

Three kernels:
  - moe_routed_decode_stage1_kernel : per (k_blk, di_block) compute
    up_logits[k_blk, di_block] = SiLU(W_gate[idx[k], di_block, :] @ x)
                                 * (W_up [idx[k], di_block, :] @ x)
  - moe_routed_decode_stage2_kernel : per dh_block accumulate
    out[dh_block] = sum_k (W_down[idx[k], dh_block, :] @ up_logits[k]) * w[k]
  - moe_shared_decode_kernel        : single fused MLP (gate, up, SiLU,
    down) for the shared expert at L=1.

For ROCm 6.2 / MI210 (gfx90a) we use fp16 storage with fp32 accumulators.
Wavefront width is 64; default reduce_threads = 64 maps to one warp-level
allreduce.
"""

import tilelang
import tilelang.language as T
from tvm import DataType


# ---------------------------------------------------------------------------
# Routed stage 1: gate & up GEMV with fused SiLU * up → up_logits[topk, d_inter]
# ---------------------------------------------------------------------------
@tilelang.jit(target="hip")
def moe_routed_decode_stage1_kernel(
    d_hidden: int,
    d_inter: int,
    n_routed_experts: int,
    topk: int,
    BLOCK_N: int = 64,
    reduce_threads: int = 64,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    TILE_K  = 128 // DataType(dtype).bits   # 8 for fp16 (16-byte vector load)
    BLOCK_K = reduce_threads * TILE_K       # 512 default

    @T.prim_func
    def kernel(
        x:           T.Tensor((d_hidden,), dtype),
        W_gate:      T.Tensor((n_routed_experts, d_inter, d_hidden), dtype),
        W_up:        T.Tensor((n_routed_experts, d_inter, d_hidden), dtype),
        expert_idx:  T.Tensor((topk,), "int32"),
        up_logits:   T.Tensor((topk, d_inter), dtype),
    ):
        with T.Kernel(
            topk,
            T.ceildiv(d_inter, BLOCK_N),
            threads=(BLOCK_N, reduce_threads),
        ) as (k_blk, bn):
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local  = T.alloc_local((TILE_K,), dtype)
            Bg_local = T.alloc_local((TILE_K,), dtype)
            Bu_local = T.alloc_local((TILE_K,), dtype)
            G_acc    = T.alloc_local((1,), accum_dtype)
            U_acc    = T.alloc_local((1,), accum_dtype)

            T.clear(G_acc)
            T.clear(U_acc)

            for bk in T.serial(T.ceildiv(d_hidden, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    A_local[k]  = x[bk * BLOCK_K + tk * TILE_K + k]
                    Bg_local[k] = W_gate[expert_idx[k_blk], bn * BLOCK_N + tn,
                                         bk * BLOCK_K + tk * TILE_K + k]
                    Bu_local[k] = W_up  [expert_idx[k_blk], bn * BLOCK_N + tn,
                                         bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    G_acc[0] += T.Cast(accum_dtype, A_local[k]) * \
                                T.Cast(accum_dtype, Bg_local[k])
                    U_acc[0] += T.Cast(accum_dtype, A_local[k]) * \
                                T.Cast(accum_dtype, Bu_local[k])

            G_red = T.alloc_local((1,), accum_dtype)
            U_red = T.alloc_local((1,), accum_dtype)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b,
                               [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), G_acc[0], True, G_red[0], tk, dtype="handle"))
            with T.attr(
                T.comm_reducer(lambda a, b: a + b,
                               [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), U_acc[0], True, U_red[0], tk, dtype="handle"))

            sig = T.alloc_local((1,), accum_dtype)
            sig[0] = G_red[0] / (1.0 + T.exp(-G_red[0]))
            up_logits[k_blk, bn * BLOCK_N + tn] = T.Cast(
                dtype, sig[0] * U_red[0]
            )

    return kernel


# ---------------------------------------------------------------------------
# Routed stage 2: down projection + per-expert routing-weight scaling
# ---------------------------------------------------------------------------
@tilelang.jit(target="hip")
def moe_routed_decode_stage2_kernel(
    d_hidden: int,
    d_inter: int,
    n_routed_experts: int,
    topk: int,
    BLOCK_N: int = 64,
    reduce_threads: int = 64,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    TILE_K  = 128 // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def kernel(
        W_down:      T.Tensor((n_routed_experts, d_hidden, d_inter), dtype),
        expert_idx:  T.Tensor((topk,), "int32"),
        weights:     T.Tensor((topk,), accum_dtype),
        up_logits:   T.Tensor((topk, d_inter), dtype),
        out:         T.Tensor((d_hidden,), dtype),
    ):
        with T.Kernel(
            T.ceildiv(d_hidden, BLOCK_N),
            threads=(BLOCK_N, reduce_threads),
        ) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            U_local  = T.alloc_local((TILE_K,), dtype)
            Bd_local = T.alloc_local((TILE_K,), dtype)
            O_acc    = T.alloc_local((1,), accum_dtype)
            K_acc    = T.alloc_local((1,), accum_dtype)

            T.clear(O_acc)

            for k_blk in T.serial(topk):
                T.clear(K_acc)
                for bk in T.serial(T.ceildiv(d_inter, BLOCK_K)):
                    for k in T.vectorized(TILE_K):
                        U_local[k]  = up_logits[k_blk, bk * BLOCK_K + tk * TILE_K + k]
                        Bd_local[k] = W_down[expert_idx[k_blk], bn * BLOCK_N + tn,
                                             bk * BLOCK_K + tk * TILE_K + k]
                    for k in T.serial(TILE_K):
                        K_acc[0] += T.Cast(accum_dtype, U_local[k]) * \
                                    T.Cast(accum_dtype, Bd_local[k])
                O_acc[0] += K_acc[0] * weights[k_blk]

            O_red = T.alloc_local((1,), accum_dtype)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b,
                               [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), O_acc[0], True, O_red[0], tk, dtype="handle"))

            out[bn * BLOCK_N + tn] = T.Cast(dtype, O_red[0])

    return kernel


# ---------------------------------------------------------------------------
# Shared expert (L=1) — fused gate, up, SiLU*up, down via two stages.
# ---------------------------------------------------------------------------
@tilelang.jit(target="hip")
def moe_shared_decode_stage1_kernel(
    d_hidden: int,
    d_inter: int,
    BLOCK_N: int = 64,
    reduce_threads: int = 64,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    TILE_K  = 128 // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def kernel(
        x:         T.Tensor((d_hidden,), dtype),
        W_gate:    T.Tensor((d_inter, d_hidden), dtype),
        W_up:      T.Tensor((d_inter, d_hidden), dtype),
        up_logits: T.Tensor((d_inter,), dtype),
    ):
        with T.Kernel(
            T.ceildiv(d_inter, BLOCK_N),
            threads=(BLOCK_N, reduce_threads),
        ) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local  = T.alloc_local((TILE_K,), dtype)
            Bg_local = T.alloc_local((TILE_K,), dtype)
            Bu_local = T.alloc_local((TILE_K,), dtype)
            G_acc    = T.alloc_local((1,), accum_dtype)
            U_acc    = T.alloc_local((1,), accum_dtype)
            T.clear(G_acc)
            T.clear(U_acc)

            for bk in T.serial(T.ceildiv(d_hidden, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    A_local[k]  = x[bk * BLOCK_K + tk * TILE_K + k]
                    Bg_local[k] = W_gate[bn * BLOCK_N + tn,
                                         bk * BLOCK_K + tk * TILE_K + k]
                    Bu_local[k] = W_up  [bn * BLOCK_N + tn,
                                         bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    G_acc[0] += T.Cast(accum_dtype, A_local[k]) * \
                                T.Cast(accum_dtype, Bg_local[k])
                    U_acc[0] += T.Cast(accum_dtype, A_local[k]) * \
                                T.Cast(accum_dtype, Bu_local[k])

            G_red = T.alloc_local((1,), accum_dtype)
            U_red = T.alloc_local((1,), accum_dtype)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b,
                               [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), G_acc[0], True, G_red[0], tk, dtype="handle"))
            with T.attr(
                T.comm_reducer(lambda a, b: a + b,
                               [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), U_acc[0], True, U_red[0], tk, dtype="handle"))

            sig = T.alloc_local((1,), accum_dtype)
            sig[0] = G_red[0] / (1.0 + T.exp(-G_red[0]))
            up_logits[bn * BLOCK_N + tn] = T.Cast(dtype, sig[0] * U_red[0])

    return kernel


@tilelang.jit(target="hip")
def moe_shared_decode_stage2_kernel(
    d_hidden: int,
    d_inter: int,
    BLOCK_N: int = 64,
    reduce_threads: int = 64,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    TILE_K  = 128 // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def kernel(
        W_down:    T.Tensor((d_hidden, d_inter), dtype),
        up_logits: T.Tensor((d_inter,), dtype),
        out:       T.Tensor((d_hidden,), dtype),
    ):
        with T.Kernel(
            T.ceildiv(d_hidden, BLOCK_N),
            threads=(BLOCK_N, reduce_threads),
        ) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            U_local  = T.alloc_local((TILE_K,), dtype)
            Bd_local = T.alloc_local((TILE_K,), dtype)
            O_acc    = T.alloc_local((1,), accum_dtype)
            T.clear(O_acc)

            for bk in T.serial(T.ceildiv(d_inter, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    U_local[k]  = up_logits[bk * BLOCK_K + tk * TILE_K + k]
                    Bd_local[k] = W_down[bn * BLOCK_N + tn,
                                         bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    O_acc[0] += T.Cast(accum_dtype, U_local[k]) * \
                                T.Cast(accum_dtype, Bd_local[k])

            O_red = T.alloc_local((1,), accum_dtype)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b,
                               [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), O_acc[0], True, O_red[0], tk, dtype="handle"))

            out[bn * BLOCK_N + tn] = T.Cast(dtype, O_red[0])

    return kernel


# ---------------------------------------------------------------------------
# INT8 decode kernels (W4A8-style storage: int8 weights & activations,
# per-tensor symmetric scales, fp32 accumulators).
#
# Memory bandwidth is the bottleneck on a GEMV at L=1; storing weights as
# int8 halves the HBM traffic vs fp16. gfx90a has no int8 MFMA but we don't
# need it here — the dot is scalar/vector i32 += i8*i8, which the compiler
# maps to v_dot4 on gfx90a.
# ---------------------------------------------------------------------------
@tilelang.jit(target="hip")
def moe_routed_decode_stage1_int8_kernel(
    d_hidden: int,
    d_inter: int,
    n_routed_experts: int,
    topk: int,
    BLOCK_N: int = 64,
    reduce_threads: int = 64,
    out_dtype: str = "float16",
):
    """Routed stage 1 (int8 storage, viewed as packed int32):
        gate[k_blk, di] = SiLU(x_q @ W_gate_q[idx[k_blk], di]) * sx * sw_g
        up  [k_blk, di] = (x_q @ W_up_q  [idx[k_blk], di])     * sx * sw_u
        up_logits[k_blk, di] = SiLU(gate) * up   (cast back to ``out_dtype``)

    The int8 weights and activations are passed as int32 buffers — every
    word packs four consecutive int8 lanes (PyTorch ``.view(torch.int32)``
    reinterpret). This dodges tilelang's HIP int8 vector-broadcast codegen
    bug while keeping vectorized HBM loads.
    """
    assert d_hidden % 4 == 0, "d_hidden must be a multiple of 4 for packed int8"
    d_hidden_pack = d_hidden // 4
    TILE_W   = 4                        # 4 int32 words = 16 int8 lanes per load
    BLOCK_KW = reduce_threads * TILE_W  # words per K-block
    accum_dtype = "int32"
    fp32 = "float32"

    @T.prim_func
    def kernel(
        x:          T.Tensor((d_hidden_pack,), "int32"),
        W_gate:     T.Tensor((n_routed_experts, d_inter, d_hidden_pack), "int32"),
        W_up:       T.Tensor((n_routed_experts, d_inter, d_hidden_pack), "int32"),
        x_scale:    T.Tensor((1,), fp32),
        Wg_scale:   T.Tensor((n_routed_experts,), fp32),
        Wu_scale:   T.Tensor((n_routed_experts,), fp32),
        expert_idx: T.Tensor((topk,), "int32"),
        up_logits:  T.Tensor((topk, d_inter), out_dtype),
    ):
        with T.Kernel(
            topk,
            T.ceildiv(d_inter, BLOCK_N),
            threads=(BLOCK_N, reduce_threads),
        ) as (k_blk, bn):
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local  = T.alloc_local((TILE_W,), accum_dtype)
            Bg_local = T.alloc_local((TILE_W,), accum_dtype)
            Bu_local = T.alloc_local((TILE_W,), accum_dtype)
            G_acc    = T.alloc_local((1,), accum_dtype)
            U_acc    = T.alloc_local((1,), accum_dtype)

            T.clear(G_acc)
            T.clear(U_acc)

            for bk in T.serial(T.ceildiv(d_hidden_pack, BLOCK_KW)):
                for w in T.vectorized(TILE_W):
                    A_local[w]  = x[bk * BLOCK_KW + tk * TILE_W + w]
                    Bg_local[w] = W_gate[expert_idx[k_blk], bn * BLOCK_N + tn,
                                         bk * BLOCK_KW + tk * TILE_W + w]
                    Bu_local[w] = W_up  [expert_idx[k_blk], bn * BLOCK_N + tn,
                                         bk * BLOCK_KW + tk * TILE_W + w]
                # Unpack 4 signed int8 lanes per int32 word (cast int32 -> int8
                # truncates the relevant byte, cast back to int32 sign-extends).
                for w in T.serial(TILE_W):
                    a_b0  = T.Cast(accum_dtype, T.Cast("int8", A_local[w]))
                    a_b1  = T.Cast(accum_dtype, T.Cast("int8", A_local[w] >> 8))
                    a_b2  = T.Cast(accum_dtype, T.Cast("int8", A_local[w] >> 16))
                    a_b3  = T.Cast(accum_dtype, T.Cast("int8", A_local[w] >> 24))
                    bg_b0 = T.Cast(accum_dtype, T.Cast("int8", Bg_local[w]))
                    bg_b1 = T.Cast(accum_dtype, T.Cast("int8", Bg_local[w] >> 8))
                    bg_b2 = T.Cast(accum_dtype, T.Cast("int8", Bg_local[w] >> 16))
                    bg_b3 = T.Cast(accum_dtype, T.Cast("int8", Bg_local[w] >> 24))
                    bu_b0 = T.Cast(accum_dtype, T.Cast("int8", Bu_local[w]))
                    bu_b1 = T.Cast(accum_dtype, T.Cast("int8", Bu_local[w] >> 8))
                    bu_b2 = T.Cast(accum_dtype, T.Cast("int8", Bu_local[w] >> 16))
                    bu_b3 = T.Cast(accum_dtype, T.Cast("int8", Bu_local[w] >> 24))
                    G_acc[0] += a_b0 * bg_b0 + a_b1 * bg_b1 + a_b2 * bg_b2 + a_b3 * bg_b3
                    U_acc[0] += a_b0 * bu_b0 + a_b1 * bu_b1 + a_b2 * bu_b2 + a_b3 * bu_b3

            G_red = T.alloc_local((1,), accum_dtype)
            U_red = T.alloc_local((1,), accum_dtype)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b,
                               [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), G_acc[0], True, G_red[0], tk, dtype="handle"))
            with T.attr(
                T.comm_reducer(lambda a, b: a + b,
                               [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), U_acc[0], True, U_red[0], tk, dtype="handle"))

            sx  = x_scale[0]
            g_f = T.alloc_local((1,), fp32)
            u_f = T.alloc_local((1,), fp32)
            g_f[0] = T.Cast(fp32, G_red[0]) * sx * Wg_scale[expert_idx[k_blk]]
            u_f[0] = T.Cast(fp32, U_red[0]) * sx * Wu_scale[expert_idx[k_blk]]
            sig    = g_f[0] / (1.0 + T.exp(-g_f[0]))
            up_logits[k_blk, bn * BLOCK_N + tn] = T.Cast(out_dtype, sig * u_f[0])

    return kernel


@tilelang.jit(target="hip")
def moe_routed_decode_stage2_int8_kernel(
    d_hidden: int,
    d_inter: int,
    n_routed_experts: int,
    topk: int,
    BLOCK_N: int = 64,
    reduce_threads: int = 64,
    in_dtype: str = "float16",
):
    """Routed stage 2 (int8 W_down, fp16 up_logits):
        out[dh] = sum_k weights[k] * sw_d[idx[k]] * (W_down_q[idx[k], dh] @ up_logits[k])
    Mixed-precision scalar GEMV: fp32 accumulator, int8 W_down dequantized
    inline. up_logits stays in ``in_dtype`` (fp16) since it is small and
    re-quantizing it would cost an extra pass with no bandwidth win.
    """
    # int8 W_down passed as int32 (4 packed lanes per word). up_logits is
    # fp16. We pair a 2-int32 word load (8 int8 lanes) with a fp16 vector
    # load of width 8, so each thread covers 8 lanes per inner step.
    assert d_inter % 4 == 0, "d_inter must be a multiple of 4 for packed int8"
    d_inter_pack = d_inter // 4
    TILE_W   = 2                            # 2 int32 words = 8 int8 lanes
    TILE_FP  = TILE_W * 4                   # 8 fp16 lanes (matches int8 lanes)
    BLOCK_KW = reduce_threads * TILE_W      # words per K-block
    fp32 = "float32"

    @T.prim_func
    def kernel(
        W_down:     T.Tensor((n_routed_experts, d_hidden, d_inter_pack), "int32"),
        Wd_scale:   T.Tensor((n_routed_experts,), fp32),
        expert_idx: T.Tensor((topk,), "int32"),
        weights:    T.Tensor((topk,), fp32),
        up_logits:  T.Tensor((topk, d_inter), in_dtype),
        out:        T.Tensor((d_hidden,), in_dtype),
    ):
        with T.Kernel(
            T.ceildiv(d_hidden, BLOCK_N),
            threads=(BLOCK_N, reduce_threads),
        ) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            U_local  = T.alloc_local((TILE_FP,), in_dtype)
            Bd_local = T.alloc_local((TILE_W,),  "int32")
            O_acc    = T.alloc_local((1,), fp32)
            K_acc    = T.alloc_local((1,), fp32)

            T.clear(O_acc)

            for k_blk in T.serial(topk):
                T.clear(K_acc)
                for bk in T.serial(T.ceildiv(d_inter_pack, BLOCK_KW)):
                    for k in T.vectorized(TILE_FP):
                        U_local[k] = up_logits[k_blk,
                                               bk * BLOCK_KW * 4
                                               + tk * TILE_FP + k]
                    for w in T.vectorized(TILE_W):
                        Bd_local[w] = W_down[expert_idx[k_blk], bn * BLOCK_N + tn,
                                             bk * BLOCK_KW + tk * TILE_W + w]
                    for w in T.serial(TILE_W):
                        b0 = T.Cast(fp32, T.Cast("int8", Bd_local[w]))
                        b1 = T.Cast(fp32, T.Cast("int8", Bd_local[w] >> 8))
                        b2 = T.Cast(fp32, T.Cast("int8", Bd_local[w] >> 16))
                        b3 = T.Cast(fp32, T.Cast("int8", Bd_local[w] >> 24))
                        u0 = T.Cast(fp32, U_local[w * 4 + 0])
                        u1 = T.Cast(fp32, U_local[w * 4 + 1])
                        u2 = T.Cast(fp32, U_local[w * 4 + 2])
                        u3 = T.Cast(fp32, U_local[w * 4 + 3])
                        K_acc[0] += u0 * b0 + u1 * b1 + u2 * b2 + u3 * b3
                O_acc[0] += K_acc[0] * weights[k_blk] * Wd_scale[expert_idx[k_blk]]

            O_red = T.alloc_local((1,), fp32)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b,
                               [T.Cast(fp32, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), O_acc[0], True, O_red[0], tk, dtype="handle"))

            out[bn * BLOCK_N + tn] = T.Cast(in_dtype, O_red[0])

    return kernel
