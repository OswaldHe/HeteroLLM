#!/usr/bin/env python3
"""
GPU → FPGA P2P Transfer — Python Example

Demonstrates the full round-trip:
  1. FPGA generates indices (FPGA kernel → P2P buffer)
  2. GPU reads indices from FPGA (existing FPGA→GPU path)
  3. GPU computes results
  4. GPU writes results back to FPGA (new GPU→FPGA path)
  5. FPGA reads the results (via host-mapped pointer)
"""

import numpy as np
import heteromem_p2p as hm

def main():
    # ------------------------------------------------------------------
    # Device setup
    # ------------------------------------------------------------------
    print("=== GPU → FPGA P2P Python Demo ===\n")

    fpga = hm.FPGADevice.auto_detect()
    gpu  = hm.GPUDevice.auto_detect()
    mgr  = hm.P2PManager(fpga, gpu)

    print(f"FPGA: {fpga.name} (BDF: {fpga.bdf})")
    print(f"GPU:  {gpu.name} (device {gpu.device_id})")

    # ------------------------------------------------------------------
    # 1. Allocate a P2P buffer on the FPGA
    # ------------------------------------------------------------------
    COUNT = 1024
    buf = fpga.create_buffer(COUNT, np.dtype(np.uint32))
    buf.register_with_gpu()

    print(f"\nP2P buffer: {buf.count} x uint32 = {buf.size} bytes")
    print(f"  is_p2p:          {buf.is_p2p}")
    print(f"  is_gpu_registered: {buf.is_gpu_registered}")

    # ------------------------------------------------------------------
    # 2. GPU writes data to FPGA via kernel P2P
    # ------------------------------------------------------------------
    data = np.arange(COUNT, dtype=np.uint32) * 7 + 42
    print(f"\nWriting {COUNT} elements GPU → FPGA...")

    mgr.memcpy_transfer_gpu_to_fpga(buf, data)

    # ------------------------------------------------------------------
    # 3. Verify from FPGA side (read through host-mapped pointer)
    # ------------------------------------------------------------------
    buf.sync_to_host()
    result = buf.read()
    assert np.array_equal(result, data), "Verification FAILED!"
    print(f"  Verified: first 8 = {result[:8]}")

    # ------------------------------------------------------------------
    # 4. Benchmark
    # ------------------------------------------------------------------
    print("\nBenchmarking GPU→FPGA write (100 iterations)...")
    bench = mgr.benchmark_gpu_write(buf, data, iterations=100)
    print(f"  Bandwidth: {bench.bandwidth_gbps:.3f} GB/s")
    print(f"  Latency:   {bench.latency_ms:.3f} ms")

    # ------------------------------------------------------------------
    # 5. Compare with existing FPGA→GPU direction
    # ------------------------------------------------------------------
    print("\nBenchmarking FPGA→GPU read (100 iterations)...")
    bench_read = mgr.benchmark_transfer(buf, fpga_to_gpu=True, iterations=100)
    print(f"  Bandwidth: {bench_read.bandwidth_gbps:.3f} GB/s")
    print(f"  Latency:   {bench_read.latency_ms:.3f} ms")

    # ------------------------------------------------------------------
    # 6. Float data round-trip
    # ------------------------------------------------------------------
    print("\n--- Float data round-trip ---")
    fbuf = fpga.create_buffer(COUNT, np.dtype(np.float32))
    fbuf.register_with_gpu()

    fdata = np.random.randn(COUNT).astype(np.float32)
    gpu.memcpy_write_floats_to_fpga(fbuf, fdata)

    fbuf.sync_to_host()
    fresult = fbuf.read()
    max_err = np.max(np.abs(fresult - fdata))
    print(f"  Max error after round-trip: {max_err}")
    assert max_err < 1e-6, "Float verification FAILED!"
    print("  Float round-trip PASSED")

    # Cleanup
    buf.unregister_from_gpu()
    fbuf.unregister_from_gpu()

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()