"""Smoke test: TileLang GEMM on AMD MI210 (gfx90a)."""

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1], target="hip")
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    M, N, K = 1024, 1024, 1024
    print("Building TileLang HIP kernel ...")
    kernel = matmul(M, N, K, 128, 128, 32)

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    print("Launching kernel ...")
    c = kernel(a, b)
    ref = a @ b

    err = (c.float() - ref.float()).abs().max().item()
    print(f"max abs err = {err:.4f}")
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    print("PASS: TileLang GEMM works on MI210 (gfx90a)")


if __name__ == "__main__":
    main()
