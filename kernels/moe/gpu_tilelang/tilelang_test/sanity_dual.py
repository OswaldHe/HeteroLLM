"""Test: a single block doing TWO independent split-K reductions."""
import torch
import tilelang
import tilelang.language as T
from tvm import DataType


@tilelang.jit(target="hip")
def dual_gemv(N, K, BLOCK_N, reduce_threads,
              dtype="float16", accum_dtype="float"):
    TILE_K  = 128 // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def main(
        A:  T.Tensor((K,), dtype),
        B1: T.Tensor((N, K), dtype),
        B2: T.Tensor((N, K), dtype),
        C1: T.Tensor((N,), dtype),
        C2: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N),
                       threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local  = T.alloc_local((TILE_K,), dtype)
            B1_local = T.alloc_local((TILE_K,), dtype)
            B2_local = T.alloc_local((TILE_K,), dtype)
            C1_acc   = T.alloc_local((1,), accum_dtype)
            C2_acc   = T.alloc_local((1,), accum_dtype)
            T.clear(C1_acc)
            T.clear(C2_acc)

            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    A_local[k]  = A[bk*BLOCK_K + tk*TILE_K + k]
                    B1_local[k] = B1[bn*BLOCK_N + tn, bk*BLOCK_K + tk*TILE_K + k]
                    B2_local[k] = B2[bn*BLOCK_N + tn, bk*BLOCK_K + tk*TILE_K + k]
                for k in T.serial(TILE_K):
                    C1_acc[0] += T.Cast(accum_dtype, A_local[k]) * \
                                 T.Cast(accum_dtype, B1_local[k])
                    C2_acc[0] += T.Cast(accum_dtype, A_local[k]) * \
                                 T.Cast(accum_dtype, B2_local[k])

            C1_red = T.alloc_local((1,), accum_dtype)
            C2_red = T.alloc_local((1,), accum_dtype)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), C1_acc[0], True, C1_red[0], tk, dtype="handle"))
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), C2_acc[0], True, C2_red[0], tk, dtype="handle"))

            C1[bn*BLOCK_N + tn] = T.Cast(dtype, C1_red[0])
            C2[bn*BLOCK_N + tn] = T.Cast(dtype, C2_red[0])

    return main


N, K = 2048, 7168
A  = torch.randn(K,    device="cuda", dtype=torch.float16)
B1 = (torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.01).contiguous()
B2 = (torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.01).contiguous()
ref1 = (A.float() @ B1.float().T).to(torch.float16)
ref2 = (A.float() @ B2.float().T).to(torch.float16)

C1 = torch.empty(N, device="cuda", dtype=torch.float16)
C2 = torch.empty(N, device="cuda", dtype=torch.float16)

kernel = dual_gemv(N, K, 16, 64)
kernel(A, B1, B2, C1, C2)

print(f"C1 err = {(C1-ref1).float().abs().max().item():.5f} (ref max={ref1.abs().max().item():.5f})")
print(f"C2 err = {(C2-ref2).float().abs().max().item():.5f} (ref max={ref2.abs().max().item():.5f})")
