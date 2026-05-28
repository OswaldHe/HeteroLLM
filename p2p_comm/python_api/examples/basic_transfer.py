#!/usr/bin/env python3
"""
Basic P2P Transfer Example

Demonstrates simple data transfer between FPGA and GPU using P2P buffers.

Usage:
    python basic_transfer.py [--fpga-bdf BDF]
    
Example:
    python basic_transfer.py --fpga-bdf 81:00.1
"""

import heteromem_p2p as p2p
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Basic P2P transfer example')
    parser.add_argument('--fpga-bdf', type=str, 
                       default=os.environ.get('FPGA_BDF'),
                       help='FPGA PCIe BDF (e.g., 81:00.1) or set FPGA_BDF env var')
    args = parser.parse_args()
    
    print("=" * 60)
    print("HeteroMem P2P - Basic Transfer Example")
    print("=" * 60)
    
    # Initialize devices
    print("\n[1/5] Initializing devices...")
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
    
    # Create P2P buffer
    print("\n[2/5] Creating P2P buffer...")
    buffer_size = 64
    buffer = fpga.create_buffer(buffer_size, np.dtype('uint32'))
    print(f"  Buffer: {buffer.count} elements ({buffer.size} bytes)")
    
    # Register with GPU
    print("\n[3/5] Registering buffer with GPU...")
    buffer.register_with_gpu()
    print(f"  Registered: {buffer.is_gpu_registered}")
    
    # Write data from host
    print("\n[4/5] Writing test data...")
    test_data = np.arange(buffer_size, dtype=np.uint32)
    buffer.write(test_data)
    buffer.sync_to_device()
    print(f"  Wrote: {test_data[:5]}... (first 5 elements)")
    
    # Read data on GPU
    print("\n[5/5] Reading data on GPU...")
    gpu_data = gpu.read_buffer(buffer)
    print(f"  Read: {gpu_data[:5]}... (first 5 elements)")
    
    # Verify
    if np.array_equal(test_data, gpu_data):
        print("\n✓ Transfer successful! Data matches.")
    else:
        print("\n✗ Transfer failed! Data mismatch.")
        errors = np.sum(test_data != gpu_data)
        print(f"  Errors: {errors}/{buffer_size}")
    
    # Benchmark
    print("\n" + "=" * 60)
    print("Benchmarking P2P Transfer")
    print("=" * 60)
    
    manager = p2p.P2PManager(fpga, gpu)
    
    # FPGA -> GPU
    result = manager.benchmark_transfer(buffer, fpga_to_gpu=True, iterations=100)
    print(f"\nFPGA → GPU:")
    print(f"  Bandwidth: {result.bandwidth_gbps:.2f} GB/s")
    print(f"  Latency: {result.latency_ms:.3f} ms")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
