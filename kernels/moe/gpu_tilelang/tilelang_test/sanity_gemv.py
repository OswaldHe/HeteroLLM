"""Sanity: split-K GEMV with allreduce on MI210.

We replicate the upstream `splitk_gemv_vectorized_tvm` and verify it
produces correct results for our typical decode shapes.
"""

import torch
import tilelang
import tilelang.language as T
from tvm import DataType


@tilelang.jit(out_idx=[-1], target="hip")
def splitk_gemv_vec_tvm(N, K, BLOCK_N, reduce_threads,
                        dtype=T.float16, accum_dtype=T.float32):
    TILE_K  = 128 // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def main(
        A: T.Tensor((K,), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_acc   = T.alloc_local((1,), accum_dtype)
            T.clear(C_acc)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn,
                                   bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_acc[0] += T.Cast(accum_dtype, A_local[k]) * \
                                T.Cast(accum_dtype, B_local[k])
            C_red = T.alloc_local((1,), accum_dtype)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), C_acc[0], True, C_red[0], tk, dtype="handle"))

            C[bn * BLOCK_N + tn] = T.Cast(dtype, C_red[0])

    return main


def main():
    N, K = 2048, 7168
    A = torch.randn(K,    device="cuda", dtype=torch.float16)
    B = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.01
    ref = (A.float() @ B.float().T).to(torch.float16)

    for BLOCK_N, RT in [(64, 16), (16, 64), (32, 32)]:
        kernel = splitk_gemv_vec_tvm(N, K, BLOCK_N, RT)
        C = kernel(A, B)
        err = (C.float() - ref.float()).abs().max().item()
        rel = err / max(ref.abs().max().item(), 1e-6)
        print(f"BLOCK_N={BLOCK_N} reduce_threads={RT}  max_abs_err={err:.5f}  rel={rel:.2e}")


if __name__ == "__main__":
    main()
