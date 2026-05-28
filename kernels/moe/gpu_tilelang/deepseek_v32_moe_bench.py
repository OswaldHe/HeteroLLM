#!/usr/bin/env python3
"""
Latency benchmark for a single MoE layer from DeepSeek V3.2 (671B config).
Weights are randomly initialized — measures compute/memory latency only.

Runs on GPU via PyTorch (AMD MI210 w/ ROCm or NVIDIA w/ CUDA).

Usage:
  python deepseek_v32_moe_bench.py --L 1
  python deepseek_v32_moe_bench.py --L 128 --dtype bf16 --iters 200
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal, Tuple


# ---------------------------------------------------------------------------
# Config — DeepSeek V3.2 671B
# ---------------------------------------------------------------------------

@dataclass
class ModelArgs:
    dim: int              = 7168
    moe_inter_dim: int    = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int  = 8
    n_limited_groups: int = 4
    route_scale: float    = 2.5
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"


# ---------------------------------------------------------------------------
# Modules (faithful to model.py)
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
        # bias is present in V3.2 (dim == 7168 branch in the original)
        self.bias = nn.Parameter(torch.zeros(args.n_routed_experts, dtype=torch.float32))
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

        # Group-based routing: select topk_groups out of n_groups
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            # sum of top-2 within each group as group score
            group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            group_indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=torch.bool)
            mask.scatter_(1, group_indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)

        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)

        # sigmoid mode: renormalize selected weights
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
    """Shared expert MLP."""
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(
            (F.silu(self.w1(x).float()) * self.w3(x).float()).to(x.dtype)
        )


class MoE(nn.Module):
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
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()

        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            y[idx] += self.experts[i](x[idx]).float() * weights[idx, top, None]

        # NOTE: shared expert disabled for routed-only benchmark
        # y = y + self.shared_experts(x).float()
        return y.to(x.dtype).view(shape)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def sync():
    torch.cuda.synchronize()


def run_benchmark(model: nn.Module, x: torch.Tensor, warmup: int, iters: int):
    for _ in range(warmup):
        with torch.no_grad():
            model(x)
    sync()

    latencies_ms = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        sync()
        latencies_ms.append((time.perf_counter() - t0) * 1e3)

    return sorted(latencies_ms)


def pct(vals, p):
    return vals[int(len(vals) * p / 100)]


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek V3.2 single-layer MoE latency benchmark (random weights)"
    )
    parser.add_argument("--L",      type=int, required=True,
                        help="Number of tokens in the batch")
    parser.add_argument("--dtype",  choices=["bf16", "fp16", "fp32"], default="bf16",
                        help="Compute dtype (default: bf16)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup iterations (default: 20)")
    parser.add_argument("--iters",  type=int, default=100,
                        help="Timed iterations (default: 100)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="PyTorch device string (default: cuda)")
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype  = dtype_map[args.dtype]
    device = torch.device(args.device)

    print("=" * 60)
    print("DeepSeek V3.2 671B — single MoE layer latency benchmark")
    print("=" * 60)
    print(f"  Device : {torch.cuda.get_device_name(device)}")
    print(f"  Tokens : L = {args.L}")
    print(f"  Dtype  : {args.dtype}")
    print(f"  Warmup : {args.warmup} iters")
    print(f"  Timed  : {args.iters} iters")
    print()

    cfg = ModelArgs()
    print("MoE config (V3.2 671B):")
    print(f"  dim={cfg.dim}, moe_inter_dim={cfg.moe_inter_dim}")
    print(f"  n_routed_experts={cfg.n_routed_experts}, n_activated={cfg.n_activated_experts}")
    print(f"  n_expert_groups={cfg.n_expert_groups}, n_limited_groups={cfg.n_limited_groups}")
    print(f"  n_shared_experts={cfg.n_shared_experts}, route_scale={cfg.route_scale}")
    print()

    # Rough weight memory estimate
    expert_params = cfg.n_routed_experts * 3 * cfg.dim * cfg.moe_inter_dim
    shared_params = 3 * cfg.dim * (cfg.n_shared_experts * cfg.moe_inter_dim)
    gate_params   = cfg.n_routed_experts * cfg.dim
    total_params  = expert_params + shared_params + gate_params
    bytes_per_el  = {"bf16": 2, "fp16": 2, "fp32": 4}[args.dtype]
    mem_gb        = total_params * bytes_per_el / 1e9
    print(f"Estimated weight memory: {total_params/1e9:.2f}B params  ({mem_gb:.1f} GB in {args.dtype})")
    print()

    print("Initializing model on GPU (random weights)...")
    model = MoE(cfg).to(device=device, dtype=dtype)
    model.eval()
    print("Done.\n")

    x = torch.randn(args.L, cfg.dim, device=device, dtype=dtype)

    print("Warming up...")
    latencies = run_benchmark(model, x, args.warmup, args.iters)

    mean_ms = sum(latencies) / len(latencies)
    print(f"\nResults ({args.iters} iterations):")
    print(f"  Min    : {latencies[0]:.3f} ms")
    print(f"  P50    : {pct(latencies, 50):.3f} ms")
    print(f"  P90    : {pct(latencies, 90):.3f} ms")
    print(f"  P99    : {pct(latencies, 99):.3f} ms")
    print(f"  Max    : {latencies[-1]:.3f} ms")
    print(f"  Mean   : {mean_ms:.3f} ms")
    print(f"  Thruput: {args.L / (mean_ms * 1e-3):.0f} tokens/sec")


if __name__ == "__main__":
    main()
