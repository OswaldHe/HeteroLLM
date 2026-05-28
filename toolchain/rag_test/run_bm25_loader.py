#!/usr/bin/env python3
"""
Python wrapper for running bm25_loader and extracting top-64 indices.

Usage:
    from run_bm25_loader import run_bm25_loader
    
    indices, latency_ms = run_bm25_loader(
        export_dir="./export",
        bitstream="path/to/bitstream.xclbin",
        limit_docs=1000
    )
"""

import subprocess
import re
import sys
import time
from typing import List, Tuple, Optional


def run_bm25_loader(
    export_dir: str = "./export",
    bitstream: str = "",
    use_mmap: bool = True,
    num_threads: int = 8,
    query_tokens: str = "",
    limit_docs: int = 0,
    executable: str = "./bm25_loader"
) -> Tuple[List[int], float, float]:
    """
    Run the bm25_loader executable and extract the top-64 indices.
    
    Args:
        export_dir: Directory containing exported BM25 data
        bitstream: Path to bitstream file (empty for csim)
        use_mmap: Use memory mapping for faster loading
        num_threads: Number of threads for packing
        query_tokens: Comma-separated list of query token IDs
        limit_docs: Limit to first N documents (0 = use all)
        executable: Path to the bm25_loader executable
        
    Returns:
        Tuple of (top_64_indices, extraction_latency_ms, kernel_latency_ms)
        - top_64_indices: List of 64 document indices
        - extraction_latency_ms: Time in milliseconds from subprocess exit to indices extraction
        - kernel_latency_ms: Kernel execution time reported by bm25_loader
    """
    # Build command line arguments
    cmd = [executable]
    cmd.extend(["--export_dir", export_dir])
    
    if bitstream:
        cmd.extend(["--bitstream", bitstream])
    
    if use_mmap:
        cmd.append("--use_mmap")
    else:
        cmd.append("--nouse_mmap")
    
    cmd.extend(["--num_threads", str(num_threads)])
    
    if query_tokens:
        cmd.extend(["--query_tokens", query_tokens])
    
    if limit_docs > 0:
        cmd.extend(["--limit_docs", str(limit_docs)])
    
    # Run the subprocess, capturing output but also printing it
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    # Start timing immediately after subprocess exits
    start_time = time.perf_counter()
    
    # Print stdout and stderr for debugging
    if result.stdout:
        print(result.stdout)
    
    # Extract top-64 indices from stdout
    stdout = result.stdout
    top_64_indices = []
    
    # Look for the line starting with "First 64 HW indices:"
    pattern = r"First 64 HW indices:\s*([\d\s]+)"
    match = re.search(pattern, stdout)
    
    if match:
        indices_str = match.group(1).strip()
        # Parse space-separated integers
        top_64_indices = [int(x) for x in indices_str.split()]
    
    # Extract kernel latency from stdout
    kernel_latency_ms = 0.0
    kernel_pattern = r"Kernel time:\s*([\d.]+)\s*ms"
    kernel_match = re.search(kernel_pattern, stdout)
    if kernel_match:
        kernel_latency_ms = float(kernel_match.group(1))
    
    # End timing after extraction
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000.0
    
    # Check for errors
    if result.returncode != 0:
        print(f"Warning: bm25_loader exited with code {result.returncode}")
    
    return top_64_indices, latency_ms, kernel_latency_ms


def main():
    """Example usage and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run bm25_loader and extract top-64 indices")
    parser.add_argument("--export_dir", default="./export", help="Export directory")
    parser.add_argument("--bitstream", default="", help="Path to bitstream file")
    parser.add_argument("--use_mmap", action="store_true", default=True, help="Use mmap")
    parser.add_argument("--nouse_mmap", action="store_true", help="Disable mmap")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of threads")
    parser.add_argument("--query_tokens", default="", help="Comma-separated query tokens")
    parser.add_argument("--limit_docs", type=int, default=0, help="Limit documents")
    parser.add_argument("--executable", default="./bm25_loader", help="Path to executable")
    
    args = parser.parse_args()
    
    use_mmap = args.use_mmap and not args.nouse_mmap
    
    print("Running bm25_loader...")
    indices, latency_ms, kernel_latency_ms = run_bm25_loader(
        export_dir=args.export_dir,
        bitstream=args.bitstream,
        use_mmap=use_mmap,
        num_threads=args.num_threads,
        query_tokens=args.query_tokens,
        limit_docs=args.limit_docs,
        executable=args.executable
    )
    
    print(f"\nExtracted {len(indices)} indices:")
    print(f"Top-64 indices: {indices}")
    print(f"\nKernel latency: {kernel_latency_ms:.4f} ms")
    print(f"Extraction latency: {latency_ms:.4f} ms")


if __name__ == "__main__":
    main()
