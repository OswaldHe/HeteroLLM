#!/usr/bin/env python3
"""
FPGA-GPU P2P Transfer Demo

Demonstrates basic P2P communication between FPGA and GPU.
This is a simplified demo showing the P2P infrastructure.

For full BM25 indexer functionality with complex kernel invocation,
see the C++ demo at u55c_rocm_p2p/spmv_bm25_demo.cpp
"""

import heteromem_p2p as p2p
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='FPGA-GPU P2P demo')
    parser.add_argument('--fpga-bdf', type=str,
                       default=os.environ.get('FPGA_BDF'),
                       help='FPGA PCIe BDF (e.g., 81:00.1) or set FPGA_BDF env var')
    parser.add_argument('--xclbin', type=str,
                       required=True,
                       help='Path to XCLBIN file')
    parser.add_argument('--num-indices', type=int, default=64,
                       help='Number of indices for P2P transfer')
    args = parser.parse_args()
    
    print("=" * 60)
    print("HeteroMem P2P - FPGA-GPU Transfer Demo")
    print("=" * 60)
    
    # Initialize devices
    print("\n[1/4] Initializing devices...")
    try:
        if args.fpga_bdf:
            print(f"  Using specified BDF: {args.fpga_bdf}")
            fpga = p2p.FPGADevice(args.fpga_bdf)
        else:
            fpga = p2p.FPGADevice.auto_detect()
        print(f"  FPGA: {fpga.name} ({fpga.bdf})")
    except p2p.FPGAException as e:
        print(f"  Error: {e}")
        print("\n  Available FPGA devices:")
        devices = p2p.list_fpga_devices()
        if devices:
            for dev in devices:
                print(f"    - {dev}")
            print(f"\n  Try: python {__file__} --fpga-bdf <BDF>")
        else:
            print("    No FPGA devices found!")
        return 1
    
    gpu = p2p.GPUDevice(0)
    print(f"  GPU: {gpu.name} (ID: {gpu.device_id})")
    
    # Load FPGA kernel
    print("\n[2/4] Loading FPGA kernel...")
    try:
        fpga.load_xclbin(args.xclbin)
        print(f"  ✓ Kernel loaded from {args.xclbin}")
    except p2p.FPGAException as e:
        print(f"  Error loading kernel: {e}")
        return 1
    
    # Create P2P buffer
    print(f"\n[3/4] Creating P2P buffer for {args.num_indices} indices...")
    indices_buffer = fpga.create_buffer(args.num_indices, np.dtype('uint32'))
    indices_buffer.register_with_gpu()
    print(f"  Buffer: {indices_buffer.size} bytes ({args.num_indices} indices)")
    
    # For this demo, write test data from CPU
    print("\n[4/4] Testing P2P transfer...")
    test_data = np.arange(args.num_indices, dtype=np.uint32)
    indices_buffer.write(test_data)
    indices_buffer.sync_to_device()
    print("  ✓ Data written to FPGA buffer")
    
    # GPU reads via P2P
    indices = gpu.read_buffer(indices_buffer)
    print("  ✓ GPU read via P2P")
    
    # Verify
    if np.array_equal(indices, test_data):
        print("  ✓ Data verified!")
    else:
        print("  ✗ Data mismatch!")
        return 1
    
    # Display results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Transferred {len(indices)} indices via P2P:")
    print(f"  First 16: {indices[:16].tolist()}")
    if len(indices) > 16:
        print(f"  Last 16:  {indices[-16:].tolist()}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"FPGA Device:     {fpga.name}")
    print(f"GPU Device:      {gpu.name}")
    print(f"Transfer Size:   {args.num_indices} indices")
    print(f"P2P Transfer:    ✓ Success")
    print(f"Verification:    ✓ Passed")
    
    print("\nNote: This demo shows basic P2P infrastructure.")
    print("      For BM25 kernel with complex invocation,")
    print("      see C++ demo: u55c_rocm_p2p/spmv_bm25_demo.cpp")
    
    print("\nDone!")
    return 0

if __name__ == "__main__":
    exit(main())
