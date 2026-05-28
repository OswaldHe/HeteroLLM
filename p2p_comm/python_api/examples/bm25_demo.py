#!/usr/bin/env python3
"""
BM25 Indexer P2P Demo

Demonstrates FPGA-GPU P2P communication using the BM25 indexer kernel.
This demo works with indexer_bm25_fixed.xclbin.
"""

import heteromem_p2p as p2p
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='BM25 P2P demo')
    parser.add_argument('--fpga-bdf', type=str,
                       default=os.environ.get('FPGA_BDF'),
                       help='FPGA PCIe BDF (e.g., 81:00.1) or set FPGA_BDF env var')
    parser.add_argument('--xclbin', type=str,
                       required=True,
                       help='Path to BM25 XCLBIN file (e.g., indexer_bm25_fixed.xclbin)')
    parser.add_argument('--kernel-name', type=str,
                       default='indexer_top',
                       help='Kernel name in XCLBIN (default: indexer_top)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("HeteroMem P2P - BM25 Indexer Demo")
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
            print(f"\n  Try: python {__file__} --fpga-bdf <BDF> --xclbin <path>")
        else:
            print("    No FPGA devices found!")
        return 1
    
    gpu = p2p.GPUDevice(0)
    print(f"  GPU: {gpu.name} (ID: {gpu.device_id})")
    
    # Load FPGA kernel
    print(f"\n[2/4] Loading FPGA kernel from {args.xclbin}...")
    try:
        fpga.load_xclbin(args.xclbin)
        print(f"  ✓ Kernel loaded (expecting '{args.kernel_name}' kernel)")
    except p2p.FPGAException as e:
        print(f"  Error loading kernel: {e}")
        return 1
    
    # Create P2P buffer for output indices
    print(f"\n[3/4] Creating P2P buffer for top-K indices...")
    num_indices = 64  # BM25 kernel outputs top-64 document IDs
    indices_buffer = fpga.create_buffer(num_indices, np.dtype('uint32'))
    indices_buffer.register_with_gpu()
    print(f"  Buffer: {indices_buffer.size} bytes ({num_indices} indices)")
    
    # Note: Full BM25 kernel invocation requires many input buffers
    # (df_buffer, query_bitmap, inst_mem, doc_mem[0-3])
    # For this demo, we'll just demonstrate the P2P infrastructure
    # For full kernel invocation, see C++ demo: u55c_rocm_p2p/spmv_bm25_demo.cpp
    
    print(f"\n[4/4] Testing P2P transfer...")
    print(f"  Note: This demo shows P2P infrastructure with BM25 kernel")
    print(f"        For full BM25 invocation with input data preparation,")
    print(f"        see C++ demo: u55c_rocm_p2p/spmv_bm25_demo.cpp")
    
    # Write test data to buffer to verify P2P works
    test_data = np.arange(num_indices, dtype=np.uint32)
    indices_buffer.write(test_data)
    indices_buffer.sync_to_device()
    print("  ✓ Test data written to FPGA buffer")
    
    # GPU reads via P2P
    indices = gpu.read_buffer(indices_buffer)
    print("  ✓ GPU read via P2P")
    
    # Verify P2P transfer works
    if np.array_equal(indices, test_data):
        print("  ✓ P2P transfer verified!")
    else:
        print("  ✗ P2P transfer failed!")
        return 1
    
    # Display results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"P2P transfer test with BM25 kernel:")
    print(f"  Kernel:      {args.kernel_name}")
    print(f"  XCLBIN:      {os.path.basename(args.xclbin)}")
    print(f"  Buffer size: {num_indices} indices")
    print(f"  First 16:    {indices[:16].tolist()}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"FPGA Device:     {fpga.name}")
    print(f"GPU Device:      {gpu.name}")
    print(f"Kernel Name:     {args.kernel_name}")
    print(f"P2P Transfer:    ✓ Success")
    print(f"Verification:    ✓ Passed")
    
    print("\n✓ P2P infrastructure works with BM25 kernel!")
    print("  For full BM25 kernel invocation with query processing,")
    print("  see C++ demo: u55c_rocm_p2p/spmv_bm25_demo.cpp")
    
    print("\nDone!")
    return 0

if __name__ == "__main__":
    exit(main())
