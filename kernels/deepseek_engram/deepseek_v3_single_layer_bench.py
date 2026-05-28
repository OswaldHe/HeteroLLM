#!/usr/bin/env python3
"""
Benchmark inference latency of a single DeepSeek V3 decoder layer.
Randomly initialized weights (no pretrained weights needed).
Uses flash attention / SDPA optimized kernels when available.

Usage:
    python deepseek_v3_single_layer_bench.py                         # default: layer 0, bs=1, seq=1024
    python deepseek_v3_single_layer_bench.py --layer-idx 1           # MoE layer (needs ~15GB+ VRAM)
    python deepseek_v3_single_layer_bench.py --seq-len 4096          # longer sequence
    python deepseek_v3_single_layer_bench.py --mode decode           # single-token decode step
    python deepseek_v3_single_layer_bench.py --use-compile           # torch.compile optimization
    python deepseek_v3_single_layer_bench.py --attn-impl sdpa        # use PyTorch SDPA instead of flash-attn

Requirements:
    pip install torch transformers numpy
    pip install flash-attn --no-build-isolation   # optional, for flash_attention_2
"""

import torch
import time
import argparse
import sys
import gc
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek V3 official configuration
# ─────────────────────────────────────────────────────────────────────────────
DEEPSEEK_V3_CONFIG_DICT = dict(
    architectures=["DeepseekV3ForCausalLM"],
    hidden_size=7168,
    intermediate_size=18432,
    moe_intermediate_size=2048,
    num_hidden_layers=61,
    num_attention_heads=128,
    num_key_value_heads=128,
    kv_lora_rank=512,
    q_lora_rank=1536,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    n_routed_experts=256,
    n_shared_experts=1,
    num_experts_per_tok=8,
    first_k_dense_replace=1,
    vocab_size=129280,
    max_position_embeddings=163840,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    routed_scaling_factor=2.5,
    scoring_func="softmax",
    topk_method="noaux_tc",
    norm_topk_prob=True,
    n_group=8,
    topk_group=4,
    hidden_act="silu",
    tie_word_embeddings=False,
    torch_dtype="float16",
    model_type="deepseek_v3",
)


def estimate_layer_memory(layer_idx: int) -> tuple[float, str]:
    """Estimate GPU memory (GB, FP16) for a single layer."""
    c = DEEPSEEK_V3_CONFIG_DICT
    h = c["hidden_size"]

    # ---- Attention (MLA) ----
    q_lr = c["q_lora_rank"]
    kv_lr = c["kv_lora_rank"]
    nh = c["num_attention_heads"]
    qk_nope = c["qk_nope_head_dim"]
    qk_rope = c["qk_rope_head_dim"]
    v_hd = c["v_head_dim"]

    attn_params = (
        h * q_lr                                      # q_a_proj
        + q_lr * nh * (qk_nope + qk_rope)             # q_b_proj
        + h * (kv_lr + qk_rope)                        # kv_a_proj_with_mqa
        + kv_lr * nh * (qk_nope + v_hd)               # kv_b_proj
        + nh * v_hd * h                                # o_proj
    )

    # ---- FFN ----
    if layer_idx < c["first_k_dense_replace"]:
        ffn_params = 3 * h * c["intermediate_size"]    # gate + up + down
        layer_type = "Dense FFN"
    else:
        ne = c["n_routed_experts"]
        expert_params = 3 * h * c["moe_intermediate_size"]
        shared_params = 3 * h * (c["moe_intermediate_size"] * c["n_shared_experts"])
        gate_params = h * ne
        ffn_params = ne * expert_params + shared_params + gate_params
        layer_type = f"MoE ({ne} routed + {c['n_shared_experts']} shared experts)"

    # ---- LayerNorms ----
    norm_params = 2 * h

    total = attn_params + ffn_params + norm_params
    gb = total * 2 / (1024 ** 3)  # FP16
    return gb, layer_type


# ─────────────────────────────────────────────────────────────────────────────
# Layer creation
# ─────────────────────────────────────────────────────────────────────────────
def _try_create_layer(layer_idx: int, attn_impl: str):
    """
    Attempt to create a single DeepSeek V3 decoder layer via transformers.
    Returns (layer, config) or raises on failure.
    """
    # Try native transformers class first, then trust_remote_code
    config = None
    for trust_rc in (False, True):
        try:
            from transformers import AutoConfig, AutoModelForCausalLM

            try:
                config = AutoConfig.from_pretrained(
                    "deepseek-ai/DeepSeek-V3",
                    trust_remote_code=trust_rc,
                )
            except Exception:
                # Offline fallback – build config from dict
                from transformers import AutoConfig
                import json, tempfile, os

                cfg_dict = dict(DEEPSEEK_V3_CONFIG_DICT)
                cfg_dict["auto_map"] = {
                    "AutoConfig": "configuration_deepseek.DeepseekV3Config",
                    "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM",
                }
                tmpdir = tempfile.mkdtemp()
                with open(os.path.join(tmpdir, "config.json"), "w") as f:
                    json.dump(cfg_dict, f)
                config = AutoConfig.from_pretrained(tmpdir, trust_remote_code=trust_rc)

            # Shrink model to only the layers we need
            config.num_hidden_layers = layer_idx + 1

            # Set attention implementation
            config._attn_implementation = attn_impl

            print(f"  Creating model skeleton (trust_remote_code={trust_rc}) ...")
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(
                    config,
                    trust_remote_code=trust_rc,
                    torch_dtype=torch.float16,
                    attn_implementation=attn_impl,
                )

            layer = model.model.layers[layer_idx]

            # Materialize parameters with random data
            layer = layer.to_empty(device="cpu")
            for p in layer.parameters():
                if p.ndim >= 2:
                    torch.nn.init.kaiming_uniform_(p)
                else:
                    p.data.normal_(0, 0.02)

            # Restore full config for reference
            config.num_hidden_layers = DEEPSEEK_V3_CONFIG_DICT["num_hidden_layers"]
            del model
            gc.collect()
            return layer, config

        except Exception as e:
            last_err = e
            continue

    raise last_err


def create_layer(layer_idx: int, attn_impl: str):
    """Create a layer, trying progressively simpler attention backends."""
    attn_order = []
    if attn_impl == "flash_attention_2":
        attn_order = ["flash_attention_2", "sdpa", "eager"]
    elif attn_impl == "sdpa":
        attn_order = ["sdpa", "eager"]
    else:
        attn_order = ["eager"]

    for impl in attn_order:
        try:
            print(f"  Trying attention implementation: {impl}")
            layer, config = _try_create_layer(layer_idx, impl)
            print(f"  ✓ Success with {impl}")
            return layer, config, impl
        except Exception as e:
            print(f"  ✗ {impl} failed: {e}")

    print("\nERROR: Could not create layer with any attention backend.")
    print("Install requirements:  pip install transformers>=4.48 torch")
    print("For flash attention:   pip install flash-attn --no-build-isolation")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────
def benchmark(
    layer: torch.nn.Module,
    config,
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    num_warmup: int,
    num_iters: int,
    device: str,
    use_compile: bool,
    mode: str,  # "prefill" or "decode"
):
    """Run the benchmark and return latency array (ms)."""
    layer = layer.to(device).half().eval()

    # Create rotary embedding to produce (cos, sin) position embeddings
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding
    rotary_emb = DeepseekV3RotaryEmbedding(config=config).to(device)

    if use_compile:
        print("Compiling with torch.compile (mode=reduce-overhead) ...")
        layer = torch.compile(layer, mode="reduce-overhead")

    # ---- Build inputs ----
    if mode == "decode":
        # Decode: single new token, but we keep seq_len as context length
        input_len = 1
    else:
        input_len = seq_len

    x = torch.randn(batch_size, input_len, hidden_size, dtype=torch.float16, device=device)
    pos = torch.arange(seq_len - input_len, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # Compute rotary position embeddings (cos, sin)
    with torch.no_grad():
        position_embeddings = rotary_emb(x, pos)

    # ---- Warmup ----
    print(f"Warming up ({num_warmup} iters) ...")
    with torch.no_grad():
        for i in range(num_warmup):
            out = layer(x, position_embeddings=position_embeddings)
            if i == 0:
                t = out[0] if isinstance(out, tuple) else out
                print(f"  Output shape: {t.shape}  dtype: {t.dtype}")
    torch.cuda.synchronize()

    # ---- Timed iterations ----
    print(f"Benchmarking ({num_iters} iters) ...")
    lats = []
    with torch.no_grad():
        for _ in range(num_iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = layer(x, position_embeddings=position_embeddings)
            e.record()
            torch.cuda.synchronize()
            lats.append(s.elapsed_time(e))

    return np.array(lats)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def report(lats, args, layer_type, mem_gb, attn_used, device_name):
    tok_count = args.batch_size * (1 if args.mode == "decode" else args.seq_len)
    tput = tok_count / (lats.mean() / 1000)

    w = 65
    print()
    print("=" * w)
    print("  DeepSeek V3 — Single Layer Inference Benchmark")
    print("=" * w)
    print(f"  GPU              : {device_name}")
    print(f"  Layer index      : {args.layer_idx}  ({layer_type})")
    print(f"  Attention        : {attn_used}")
    print(f"  Mode             : {args.mode}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Sequence length  : {args.seq_len}")
    print(f"  Precision        : FP16")
    print(f"  torch.compile    : {'yes' if args.use_compile else 'no'}")
    print(f"  Est. layer size  : {mem_gb:.2f} GB")
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
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a single DeepSeek V3 decoder layer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--layer-idx", type=int, default=0,
                        help="Layer index (0 = dense FFN, ≥1 = MoE)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--mode", choices=["prefill", "decode"], default="prefill",
                        help="prefill = full sequence; decode = single token step")
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--use-compile", action="store_true",
                        help="Apply torch.compile (reduce-overhead)")
    parser.add_argument("--attn-impl", default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"],
                        help="Attention kernel backend")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # ---- Device check ----
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm is not available.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU : {device_name}  ({total_mem:.1f} GB)")
    print(f"PyTorch : {torch.__version__}")

    # ---- Memory estimate ----
    mem_gb, layer_type = estimate_layer_memory(args.layer_idx)
    print(f"Layer {args.layer_idx} : {layer_type}")
    print(f"Estimated weight size : {mem_gb:.2f} GB (FP16)")

    if mem_gb > total_mem * 0.80:
        print(f"⚠  Layer may not fit in GPU memory ({mem_gb:.1f} GB vs {total_mem:.1f} GB).")
        if args.layer_idx >= DEEPSEEK_V3_CONFIG_DICT["first_k_dense_replace"]:
            print("   Tip: use --layer-idx 0 for the smaller dense layer (~0.8 GB).")

    # ---- Create layer ----
    print()
    layer, config, attn_used = create_layer(args.layer_idx, args.attn_impl)
    hidden_size = config.hidden_size

    # ---- Benchmark ----
    try:
        lats = benchmark(
            layer, config, hidden_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_warmup=args.num_warmup,
            num_iters=args.num_iterations,
            device=args.device,
            use_compile=args.use_compile,
            mode=args.mode,
        )
    except torch.cuda.OutOfMemoryError:
        print("\nERROR: Out of GPU memory!")
        print(f"  Current seq_len={args.seq_len}, batch_size={args.batch_size}")
        print("  Try reducing --seq-len or --batch-size, or use --layer-idx 0")
        sys.exit(1)

    # ---- Report ----
    report(lats, args, layer_type, mem_gb, attn_used, device_name)


if __name__ == "__main__":
    main()
