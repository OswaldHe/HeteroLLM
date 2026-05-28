"""Test shared-expert decode kernels alone vs PyTorch reference."""
import torch
import torch.nn.functional as F
from moe_tilelang_decode import (
    moe_shared_decode_stage1_kernel,
    moe_shared_decode_stage2_kernel,
)

torch.manual_seed(0)
d_h = 7168
d_i = 2048
device = "cuda"

x      = torch.randn(d_h,      device=device, dtype=torch.float16)
W_gate = (torch.randn(d_i, d_h, device=device, dtype=torch.float16) * 0.01).contiguous()
W_up   = (torch.randn(d_i, d_h, device=device, dtype=torch.float16) * 0.01).contiguous()
W_down = (torch.randn(d_h, d_i, device=device, dtype=torch.float16) * 0.01).contiguous()

# Reference
gate = F.linear(x.float(), W_gate.float())   # [d_i]
up   = F.linear(x.float(), W_up.float())     # [d_i]
upl  = F.silu(gate) * up                      # [d_i]
out  = F.linear(upl, W_down.float())          # [d_h]
ref  = out.to(torch.float16)

# TileLang
k1 = moe_shared_decode_stage1_kernel(d_h, d_i, BLOCK_N=16, reduce_threads=64)
k2 = moe_shared_decode_stage2_kernel(d_h, d_i, BLOCK_N=16, reduce_threads=64)

up_logits = torch.empty(d_i, device=device, dtype=torch.float16)
out_til   = torch.empty(d_h, device=device, dtype=torch.float16)
k1(x, W_gate, W_up, up_logits)

# Compare against gate*up (no silu) to see if the issue is in silu
no_silu = (gate * up).to(torch.float16)
err_up = (up_logits.float() - upl).abs().max().item()
err_gu = (up_logits.float() - no_silu.float()).abs().max().item()
print(f"Stage 1 vs SiLU(g)*u  err = {err_up:.5f}  ref_max={upl.abs().max().item():.5f}")
print(f"Stage 1 vs g*u (raw)  err = {err_gu:.5f}  ref_max={no_silu.abs().max().item():.5f}")

k2(W_down, up_logits, out_til)
err_o = (out_til.float() - ref.float()).abs().max().item()
print(f"Stage 2 max_abs_err (out)      = {err_o:.5f}, ref_max={ref.abs().max().item():.5f}")
