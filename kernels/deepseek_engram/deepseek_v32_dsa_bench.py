#!/usr/bin/env python3
"""
Benchmark a single DeepSeek V3.2-Exp decoder block (MLA + DSA Lightning Indexer + MoE)
using the official inference code from deepseek-ai/DeepSeek-V3.2-Exp.

Weights are randomly initialised — no checkpoint loading required.

Usage:
    python deepseek_v32_dsa_bench.py                        # MoE layer 3, bs=1, seq=1024, prefill
    python deepseek_v32_dsa_bench.py --seq-len 4096
    python deepseek_v32_dsa_bench.py --mode decode
    python deepseek_v32_dsa_bench.py --layer-idx 5

Requirements:
    pip install torch huggingface_hub numpy
"""

import sys
import os
import types
import math
import argparse
import gc

import torch
import torch.nn.functional as F
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. Patch torch.distributed so dist.broadcast is a no-op when not initialised
#    (Indexer.forward calls dist.broadcast unconditionally)
# ─────────────────────────────────────────────────────────────────────────────
import torch.distributed as _dist

_real_broadcast = _dist.broadcast


def _safe_broadcast(tensor, src=0, group=None, async_op=False):
    if not _dist.is_initialized() or _dist.get_world_size() == 1:
        return
    return _real_broadcast(tensor, src, group, async_op)


_dist.broadcast = _safe_broadcast

# ─────────────────────────────────────────────────────────────────────────────
# 2. Stub the custom `kernel` extension (fp8_index / fp8_gemm / act_quant)
#    MI210 / ROCm: no custom CUDA kernels available — pure PyTorch equivalents.
#    We keep everything in BF16 to avoid FP8 hardware issues on MI210.
# ─────────────────────────────────────────────────────────────────────────────

def _act_quant(x: torch.Tensor, block_size: int, scale_fmt=None):
    """
    Fake-quantise x to FP8 e4m3fn with per-block scales, then immediately
    dequantise back to BF16 so downstream ops stay in BF16.
    Returns (x_bf16_as_fp8_placeholder, scale) where the 'fp8' tensor is
    actually stored as BF16 to avoid fp8 arithmetic issues on MI210.
    """
    shape = x.shape
    x_f = x.float().reshape(-1, block_size)
    scale = x_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
    x_scaled = (x_f / scale).clamp(-448.0, 448.0)
    # Store as bfloat16 instead of fp8 to stay on MI210-safe path
    x_out = x_scaled.to(torch.bfloat16).reshape(shape)
    scale_out = scale.reshape(*shape[:-1], shape[-1] // block_size)
    return x_out, scale_out.float()


def _fp8_gemm(x: torch.Tensor, x_scale, weight: torch.Tensor, weight_scale):
    """BF16 linear standing in for the FP8 GEMM kernel."""
    return F.linear(x.to(torch.bfloat16), weight.to(torch.bfloat16))


_INDEX_CHUNK = 128   # query positions per chunk inside fp8_index


def _fp8_index(
    q: torch.Tensor,              # (bsz, seqlen, n_heads, head_dim)
    weights: torch.Tensor,        # (bsz, seqlen, n_heads, 1)
    k_cache: torch.Tensor,        # (bsz, end_pos, head_dim)
    k_scale_cache: torch.Tensor,  # (bsz, end_pos, head_dim//block_size)
) -> torch.Tensor:
    """
    Lightning Indexer score chunked over query positions — avoids
    materialising the full (bsz, seqlen, n_heads, end_pos) tensor.
    Peak memory per chunk: (bsz, _INDEX_CHUNK, n_heads, end_pos).
    Returns: (bsz, seqlen, end_pos)  FP32
    """
    k_f     = k_cache.float()                 # (bsz, T, D)
    k_scale = k_scale_cache.mean(dim=-1)      # (bsz, T)
    seqlen  = q.shape[1]
    chunks  = []
    for s in range(0, seqlen, _INDEX_CHUNK):
        q_c = q[:, s:s + _INDEX_CHUNK].float()      # (bsz, C, H, D)
        w_c = weights[:, s:s + _INDEX_CHUNK]         # (bsz, C, H, 1)
        logits = torch.einsum("bchd,btd->bcht", q_c, k_f)   # (bsz, C, H, T)
        logits = logits * k_scale[:, None, None, :]
        logits = torch.relu(logits)
        chunks.append((w_c * logits).sum(dim=2))     # (bsz, C, T)
    return torch.cat(chunks, dim=1)                  # (bsz, seqlen, T)


_kernel_mod = types.ModuleType("kernel")
_kernel_mod.act_quant = _act_quant
_kernel_mod.fp8_gemm  = _fp8_gemm
_kernel_mod.fp8_index = _fp8_index
sys.modules["kernel"] = _kernel_mod

# ─────────────────────────────────────────────────────────────────────────────
# 3. Stub fast_hadamard_transform (CUDA-only package)
#    Pure PyTorch butterfly Hadamard — correct, not peak-speed.
# ─────────────────────────────────────────────────────────────────────────────

def _hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Iterative Walsh–Hadamard transform, requires last dim to be a power of 2."""
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, "head_dim must be a power of 2"
    h = x.clone()
    step = 1
    while step < n:
        h = h.reshape(*h.shape[:-1], n // (2 * step), 2, step)
        a, b = h[..., 0, :], h[..., 1, :]
        h = torch.stack([a + b, a - b], dim=-2).reshape(*h.shape[:-3], n)
        step *= 2
    return h * scale


_fht_mod = types.ModuleType("fast_hadamard_transform")
_fht_mod.hadamard_transform = _hadamard_transform
sys.modules["fast_hadamard_transform"] = _fht_mod

# ─────────────────────────────────────────────────────────────────────────────
# 4. Download and import the official inference/model.py
# ─────────────────────────────────────────────────────────────────────────────
print("Fetching inference/model.py from deepseek-ai/DeepSeek-V3.2-Exp …")
from huggingface_hub import hf_hub_download

_model_py = hf_hub_download(
    repo_id="deepseek-ai/DeepSeek-V3.2-Exp",
    filename="inference/model.py",
)
sys.path.insert(0, os.path.dirname(_model_py))

from model import ModelArgs, Block, Linear, MLA, precompute_freqs_cis, apply_rotary_emb, weight_dequant  # noqa: E402

# Force BF16 for all Linear layers (avoids FP8 hardware path on MI210)
Linear.dtype = torch.bfloat16
Linear.scale_fmt = None

# ─────────────────────────────────────────────────────────────────────────────
# 5. Patch MLA.forward with optimized sparse DSA attention
#
#    Reference code: compute full O(L²) scores → apply DSA mask.
#    This patch: run indexer first → gather only top-k KV latents →
#    attend only over those k positions (O(k) instead of O(L)).
#
#    Uses the MLA latent absorption trick for both decode and prefill:
#      q_nope_proj = q_nope @ wkv_b[:, :qk_nope_head_dim]  (absorb into query)
#      scores = q_nope_proj @ kv_latent + q_pe @ k_pe
#      output = scores @ (wkv_b[:, -v_head_dim:] @ kv_latent)  (fused, no per-head V)
#
#    Prefill: SDPA (flash attention) for causal dense attention (DSA is a
#    decode-time optimisation; prefill in production also uses dense flash attn).
#    Decode: fully sparse — gather top-k from the KV latent cache.
# ─────────────────────────────────────────────────────────────────────────────

def _mla_forward_sparse(self, x: torch.Tensor, start_pos: int,
                        freqs_cis: torch.Tensor, mask) -> torch.Tensor:
    bsz, seqlen, _ = x.size()
    end_pos = start_pos + seqlen

    # ── Q projection ──────────────────────────────────────────────────────────
    qr = self.q_norm(self.wq_a(x))
    q = self.wq_b(qr).view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    q_pe = apply_rotary_emb(q_pe, freqs_cis)

    # ── KV compression + cache update ────────────────────────────────────────
    kv_raw, k_pe = torch.split(
        self.wkv_a(x), [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    kv_raw = self.kv_norm(kv_raw)
    k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

    # Simulate FP8 KV quantisation (precision rounding)
    kv_fp8, kv_scale = _act_quant(kv_raw, 128)
    kv_raw = (kv_fp8.view(-1, 128).float() * kv_scale.reshape(-1, 1)).to(kv_raw.dtype).view_as(kv_raw)

    self.kv_cache[:bsz, start_pos:end_pos] = kv_raw
    self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

    # ── Dequant wkv_b once (absorbed into Q / V reconstruction) ─────────────
    if self.dequant_wkv_b is None and getattr(self.wkv_b, "scale", None) is not None:
        self.dequant_wkv_b = weight_dequant(self.wkv_b.weight, self.wkv_b.scale)
    wkv_b = (self.dequant_wkv_b if self.dequant_wkv_b is not None
             else self.wkv_b.weight).view(self.n_local_heads, -1, self.kv_lora_rank)

    # ── Absorb wkv_b[:, :qk_nope_head_dim] into q_nope ──────────────────────
    # q_nope_proj: (bsz, seqlen, n_heads, kv_lora_rank)
    q_nope_proj = torch.einsum("bshd,hdc->bshc", q_nope,
                               wkv_b[:, :self.qk_nope_head_dim])

    if mask is not None:
        # ── PREFILL: DSA sparse attention, chunked over query positions ───────
        # 1. Run indexer (already chunked in _fp8_index) → topk_indices
        topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
        # topk_indices: (bsz, seqlen, topk)
        topk = topk_indices.shape[-1]

        kv_full = self.kv_cache[:bsz, :end_pos]  # (bsz, end_pos, kv_lora_rank)
        pe_full = self.pe_cache[:bsz, :end_pos]  # (bsz, end_pos, qk_rope_head_dim)

        # 2. Chunk over seqlen: for each chunk, gather top-k KV and attend
        #    Peak per chunk: (bsz, C, topk, kv_lora_rank) — C * topk * 512 * 2 bytes
        #    At C=64, topk=2048: 64 * 2048 * 512 * 2 = 134 MB — safe
        _ATTN_CHUNK = 64
        out_chunks = []
        for s in range(0, seqlen, _ATTN_CHUNK):
            s_e = min(s + _ATTN_CHUNK, seqlen)

            qnp = q_nope_proj[:, s:s_e]        # (bsz, C, H, kv_lora_rank)
            qpe = q_pe[:, s:s_e]               # (bsz, C, H, qk_rope_head_dim)
            ti  = topk_indices[:, s:s_e]       # (bsz, C, topk)

            # Gather KV latents at selected positions
            idx_kv = ti.unsqueeze(-1).expand(-1, -1, -1, self.kv_lora_rank)
            sel_kv = kv_full.unsqueeze(1).expand(-1, s_e - s, -1, -1).gather(2, idx_kv).to(qnp.dtype)
            # sel_kv: (bsz, C, topk, kv_lora_rank)

            idx_pe = ti.unsqueeze(-1).expand(-1, -1, -1, self.qk_rope_head_dim)
            sel_pe = pe_full.unsqueeze(1).expand(-1, s_e - s, -1, -1).gather(2, idx_pe).to(qpe.dtype)
            # sel_pe: (bsz, C, topk, qk_rope_head_dim)

            # Sparse scores over top-k only: (bsz, C, H, topk)
            sc = (torch.einsum("bchd,bctd->bcht", qnp, sel_kv) +
                  torch.einsum("bchr,bctr->bcht", qpe, sel_pe)) * self.softmax_scale
            sc = sc.softmax(dim=-1)

            # Output via latent absorption — no per-head V materialised
            # intermediate: (bsz, C, H, kv_lora_rank)
            inter = torch.einsum("bcht,bctd->bchd", sc, sel_kv)
            out_chunks.append(torch.einsum("bchd,hud->bchu", inter,
                                           wkv_b[:, -self.v_head_dim:]))

        x = torch.cat(out_chunks, dim=1)   # (bsz, seqlen, H, v_head_dim)
    else:
        # ── DECODE: sparse attention over DSA top-k positions only ───────────
        topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
        # topk_indices: (bsz, 1, topk)
        topk = topk_indices.shape[-1]

        kv_full = self.kv_cache[:bsz, :end_pos]  # (bsz, end_pos, kv_lora_rank)
        pe_full = self.pe_cache[:bsz, :end_pos]  # (bsz, end_pos, qk_rope_head_dim)

        # Gather top-k KV latents and positional encodings
        idx_kv = topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.kv_lora_rank)
        sel_kv = kv_full.unsqueeze(1).expand(-1, seqlen, -1, -1).gather(2, idx_kv).to(q_nope_proj.dtype)
        # sel_kv: (bsz, 1, topk, kv_lora_rank)

        idx_pe = topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.qk_rope_head_dim)
        sel_pe = pe_full.unsqueeze(1).expand(-1, seqlen, -1, -1).gather(2, idx_pe).to(q_pe.dtype)
        # sel_pe: (bsz, 1, topk, qk_rope_head_dim)

        # Sparse scores: (bsz, 1, n_heads, topk)
        scores = (torch.einsum("bshc,bstc->bsht", q_nope_proj, sel_kv) +
                  torch.einsum("bshr,bstr->bsht", q_pe, sel_pe)
                  ) * self.softmax_scale
        scores = scores.softmax(dim=-1)

        # Sparse output via latent absorption — avoids materialising per-head V
        # intermediate: (bsz, 1, n_heads, kv_lora_rank)
        intermediate = torch.einsum("bsht,bstc->bshc", scores, sel_kv)
        x = torch.einsum("bshc,hdc->bshd", intermediate, wkv_b[:, -self.v_head_dim:])

    return self.wo(x.flatten(2))


MLA.forward = _mla_forward_sparse

# ─────────────────────────────────────────────────────────────────────────────
# 5. V3.2 671B config
# ─────────────────────────────────────────────────────────────────────────────
V32_671B = dict(
    vocab_size=129280,
    dim=7168,
    inter_dim=18432,
    moe_inter_dim=2048,
    n_layers=61,
    n_dense_layers=3,
    n_heads=128,
    n_routed_experts=256,
    n_shared_experts=1,
    n_activated_experts=8,
    n_expert_groups=8,
    n_limited_groups=4,
    route_scale=2.5,
    score_func="sigmoid",
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    dtype="bf16",
    index_n_heads=64,
    index_head_dim=128,
    index_topk=2048,
)

_ARGS_FIELDS = set(ModelArgs.__dataclass_fields__.keys())


def make_model_args(seq_len: int, batch_size: int) -> ModelArgs:
    kwargs = {k: v for k, v in V32_671B.items() if k in _ARGS_FIELDS}
    args = ModelArgs(**kwargs)
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len
    return args


# ─────────────────────────────────────────────────────────────────────────────
# 6. Layer creation — empty alloc + random init, no checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def create_block(layer_id: int, args: ModelArgs, device: str) -> Block:
    print(f"  Allocating Block {layer_id} on meta device …")
    with torch.device("meta"):
        block = Block(layer_id, args)

    print(f"  Materialising parameters on {device} (random, no checkpoint) …")
    block = block.to_empty(device=device)

    with torch.no_grad():
        for name, p in block.named_parameters():
            if p.numel() == 0:
                continue
            if p.ndim >= 2:
                # kaiming_uniform_ needs float; cast back after
                tmp = torch.empty_like(p, dtype=torch.float32)
                torch.nn.init.kaiming_uniform_(tmp)
                p.copy_(tmp.to(p.dtype))
            else:
                p.data.fill_(1.0) if "norm" in name else p.data.zero_()

    return block


# ─────────────────────────────────────────────────────────────────────────────
# 7. Benchmark
# ─────────────────────────────────────────────────────────────────────────────
def benchmark(
    block: Block,
    args: ModelArgs,
    device: str,
    seq_len: int,
    batch_size: int,
    num_warmup: int,
    num_iters: int,
    mode: str,
) -> np.ndarray:
    block = block.to(device).eval()

    freqs_cis = precompute_freqs_cis(args).to(device)

    if mode == "decode":
        input_len = 1
        start_pos = seq_len - 1
    else:
        input_len = seq_len
        start_pos = 0

    x = torch.randn(batch_size, input_len, args.dim,
                    dtype=torch.bfloat16, device=device)
    residual = torch.randn_like(x)
    freqs = freqs_cis[start_pos: start_pos + input_len]

    if mode == "prefill":
        mask = torch.full((input_len, input_len), float("-inf"),
                          device=device).triu_(1)
    else:
        mask = None

    print(f"  Warming up ({num_warmup} iters) …")
    with torch.no_grad():
        for i in range(num_warmup):
            out, _ = block(x, residual, start_pos, freqs, mask)
            if i == 0:
                print(f"  Output shape: {out.shape}  dtype: {out.dtype}")
    torch.cuda.synchronize()

    print(f"  Benchmarking ({num_iters} iters) …")
    lats = []
    with torch.no_grad():
        for _ in range(num_iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            block(x, residual, start_pos, freqs, mask)
            e.record()
            torch.cuda.synchronize()
            lats.append(s.elapsed_time(e))

    return np.array(lats)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Report
# ─────────────────────────────────────────────────────────────────────────────
def report(lats, ns, device_name):
    tok_count = ns.batch_size * (1 if ns.mode == "decode" else ns.seq_len)
    tput = tok_count / (lats.mean() / 1000)
    w = 68
    print()
    print("=" * w)
    print("  DeepSeek V3.2-Exp — Single Layer Benchmark (MLA + DSA + MoE)")
    print("=" * w)
    print(f"  GPU              : {device_name}")
    print(f"  Layer index      : {ns.layer_idx}  (MoE + DSA Lightning Indexer)")
    print(f"  Attention        : MLA + Lightning Indexer (PyTorch stub, BF16)")
    print(f"  Mode             : {ns.mode}")
    print(f"  Batch size       : {ns.batch_size}")
    print(f"  Sequence length  : {ns.seq_len}")
    print(f"  Precision        : BF16  (FP8 kernel stubbed for MI210/ROCm)")
    print(f"  Iterations       : {len(lats)}")
    print("-" * w)
    print(f"  Mean latency     : {lats.mean():.3f} ms")
    print(f"  Median latency   : {np.median(lats):.3f} ms")
    print(f"  Std dev          : {lats.std():.3f} ms")
    print(f"  Min latency      : {lats.min():.3f} ms")
    print(f"  Max latency      : {lats.max():.3f} ms")
    print(f"  P95 latency      : {np.percentile(lats, 95):.3f} ms")
    print(f"  P99 latency      : {np.percentile(lats, 99):.3f} ms")
    print(f"  Throughput       : {tput:,.0f} tokens/s  (single layer)")
    print("=" * w)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a single DeepSeek V3.2-Exp block (MLA+DSA+MoE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--layer-idx", type=int, default=3,
                        help="Layer index (0-2 = dense FFN, ≥3 = MoE+DSA)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    ns = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA/ROCm device found.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU     : {device_name}  ({total_mem:.1f} GB)")
    print(f"PyTorch : {torch.__version__}")

    if ns.layer_idx < V32_671B["n_dense_layers"]:
        print(f"WARNING: layer {ns.layer_idx} is a dense FFN layer "
              f"(no MoE). Use --layer-idx ≥ {V32_671B['n_dense_layers']} for MoE+DSA.")

    args = make_model_args(ns.seq_len, ns.batch_size)

    print(f"\nCreating Block {ns.layer_idx} (MLA + DSA + MoE) …")
    block = create_block(ns.layer_idx, args, ns.device)

    mem_used = torch.cuda.memory_allocated(ns.device) / (1024 ** 3)
    print(f"  GPU memory used after block alloc: {mem_used:.2f} GB")

    try:
        lats = benchmark(block, args, ns.device,
                         seq_len=ns.seq_len,
                         batch_size=ns.batch_size,
                         num_warmup=ns.num_warmup,
                         num_iters=ns.num_iterations,
                         mode=ns.mode)
    except torch.cuda.OutOfMemoryError:
        print("\nERROR: Out of GPU memory.")
        print("  Try: --seq-len smaller, --batch-size 1, or --layer-idx 0 (dense)")
        sys.exit(1)

    report(lats, ns, device_name)


if __name__ == "__main__":
    main()
