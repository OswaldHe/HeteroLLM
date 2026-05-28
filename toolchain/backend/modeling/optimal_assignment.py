#!/usr/bin/env python3
"""
Optimal Kernel and Data Placement Assignment for FPGA/GPU Heterogeneous System

This script traverses all possibilities of kernel assignment (each kernel can run 
on either GPU or FPGA).

Data placement is determined by where kernels execute:
- Data is generated on the device where the producing kernel runs
- When data is transferred to another device, a copy is registered on that device
- Future accesses can use the local copy if available

PCIe transfer constraints:
- Memory transfers only happen FPGA -> GPU (not GPU -> FPGA)
- Other data can transfer in both directions
"""

import json
import os
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Set

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_PROFILE_DIR = os.path.join(BASE_DIR, "kernel_profile_results")
PCIE_CONFIG_DIR = os.path.join(BASE_DIR, "pcie_config")

# Device types
GPU = "GPU"
FPGA = "FPGA"

# Kernel names and their corresponding profile files
KERNELS = [
    "paged_kv_index_builder",
    "inner_product",
    "topk_retrieval",
    "block_sparse_attention"
]

# Define kernel data dependencies (input -> kernel -> output)
# TODO: defined by the chained kernels in deployment
# Note: data_sizes_byte is loaded dynamically from kernel profile JSON files
KERNEL_DEPENDENCIES = {
    "paged_kv_index_builder": {
        "inputs": ["retrieved_data"],
        "outputs": ["memory"],
    },
    "inner_product": {
        "inputs": ["memory", "query"],
        "outputs": ["score"],
    },
    "topk_retrieval": {
        "inputs": ["score"],
        "outputs": ["retrieved_index"],
    },
    "block_sparse_attention": {
        "inputs": ["retrieved_data", "retrieved_index", "target_data"],
        "outputs": ["target_data"],
    }
}


def load_gpu_profile() -> Dict[str, float]:
    """Load GPU kernel profiling results."""
    gpu_profile_path = os.path.join(KERNEL_PROFILE_DIR, "gpu_profile.json")
    with open(gpu_profile_path, 'r') as f:
        data = json.load(f)
    
    # Map to kernel names
    return {
        "paged_kv_index_builder": data["paged_kv_index_builder_us"],
        "inner_product": data["inner_product_us"],
        "topk_retrieval": data["topk_retrieval_us"],
        "block_sparse_attention": data["block_sparse_attention_us"]
    }


def load_fpga_profiles() -> Dict[str, float]:
    """Load FPGA kernel profiling results."""
    fpga_profiles = {}
    
    # Mapping of kernel names to their JSON files
    kernel_files = {
        "paged_kv_index_builder": "paged_kv_index_builder.json",
        "inner_product": "innerproduct.json",
        "topk_retrieval": "topk_retrival.json",
        "block_sparse_attention": "block_sparse_attention.json"
    }
    
    for kernel_name, filename in kernel_files.items():
        filepath = os.path.join(KERNEL_PROFILE_DIR, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        fpga_profiles[kernel_name] = data["roofline"]["estimated_latency_us"]
    
    return fpga_profiles


def load_memory_transactions() -> Dict[str, Dict[str, int]]:
    """
    Load memory transaction data sizes from kernel profile JSON files.
    
    Returns:
        Dict mapping kernel names to their memory transactions (data name -> size in bytes)
    """
    memory_transactions = {}
    
    # Mapping of kernel names to their JSON files
    kernel_files = {
        "paged_kv_index_builder": "paged_kv_index_builder.json",
        "inner_product": "innerproduct.json",
        "topk_retrieval": "topk_retrival.json",
        "block_sparse_attention": "block_sparse_attention.json"
    }
    
    for kernel_name, filename in kernel_files.items():
        filepath = os.path.join(KERNEL_PROFILE_DIR, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        memory_transactions[kernel_name] = data.get("memory_transactions", {})
    
    return memory_transactions


def load_pcie_config() -> Dict[str, List[Dict]]:
    """Load PCIe transfer configuration (piecewise linear regression models)."""
    pcie_config_path = os.path.join(PCIE_CONFIG_DIR, "p2p_host_jump.json")
    with open(pcie_config_path, 'r') as f:
        return json.load(f)


def load_kernel_roofline_data() -> Dict[str, Dict[str, Any]]:
    """
    Load full roofline data from kernel profile JSON files.
    
    Returns:
        Dict mapping kernel names to their roofline data including:
        - total_flops: total arithmetic operations
        - memory_bytes: total memory bytes
        - arithmetic_intensity: FLOPs per byte
        - ridge_point: device ridge point
        - estimated_latency_us: estimated latency
        - target_freq_mhz: target frequency
        - peak_memory_bw_bytes_per_cycle: derived from ridge_point
        - peak_flops_per_cycle: derived from ridge_point
    """
    roofline_data = {}
    
    kernel_files = {
        "paged_kv_index_builder": "paged_kv_index_builder.json",
        "inner_product": "innerproduct.json",
        "topk_retrieval": "topk_retrival.json",
        "block_sparse_attention": "block_sparse_attention.json"
    }
    
    for kernel_name, filename in kernel_files.items():
        filepath = os.path.join(KERNEL_PROFILE_DIR, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        roofline = data.get("roofline", {})
        op_counts = data.get("operation_counts", {})
        
        roofline_data[kernel_name] = {
            "total_flops": op_counts.get("fadd", 0),  # Using fadd as total ops
            "memory_bytes": data.get("memory_bytes", 0),
            "arithmetic_intensity": roofline.get("arithmetic_intensity", 0),
            "ridge_point": roofline.get("ridge_point", 0),
            "estimated_latency_us": roofline.get("estimated_latency_us", 0),
            "memory_bound_cycles": roofline.get("memory_bound_cycles", 0),
            "compute_bound_cycles": roofline.get("compute_bound_cycles", 0),
            "target_freq_mhz": data.get("target_freq_mhz", 250.0),
            "bottleneck": roofline.get("bottleneck", "UNKNOWN"),
            "memory_transactions": data.get("memory_transactions", {})
        }
    
    return roofline_data


def find_consecutive_kernel_groups(kernel_assignment: Dict[str, str]) -> List[List[str]]:
    """
    Find groups of consecutive kernels running on the same device.
    
    Args:
        kernel_assignment: Mapping of kernel name to device (GPU/FPGA)
    
    Returns:
        List of kernel groups, where each group contains consecutive kernels on the same device
    """
    groups = []
    current_group = []
    current_device = None
    
    for kernel in KERNELS:
        device = kernel_assignment[kernel]
        
        if device == current_device:
            current_group.append(kernel)
        else:
            if current_group:
                groups.append((current_device, current_group))
            current_group = [kernel]
            current_device = device
    
    if current_group:
        groups.append((current_device, current_group))
    
    return groups


def calculate_fused_kernel_latency(
    kernels: List[str],
    device: str,
    roofline_data: Dict[str, Dict[str, Any]],
    gpu_profiles: Dict[str, float]
) -> Tuple[float, Dict]:
    """
    Calculate the latency of fused kernels using combined roofline model.
    
    When kernels are fused, intermediate data stays in on-chip memory,
    reducing total memory traffic. This function estimates the potential
    speedup from fusion.
    
    Args:
        kernels: List of kernel names to fuse
        device: Device where kernels run (GPU/FPGA)
        roofline_data: Roofline data for each kernel
        gpu_profiles: GPU kernel latencies (for GPU device)
    
    Returns:
        Tuple of (fused_latency_us, analysis_details)
    """
    if device == GPU:
        # For GPU, we don't model fusion - just sum up individual latencies
        total_latency = sum(gpu_profiles[k] for k in kernels)
        return total_latency, {
            "device": GPU,
            "kernels": kernels,
            "fusion_applicable": False,
            "reason": "GPU fusion not modeled"
        }
    
    if len(kernels) == 1:
        # Single kernel, no fusion benefit
        kernel = kernels[0]
        return roofline_data[kernel]["estimated_latency_us"], {
            "device": FPGA,
            "kernels": kernels,
            "fusion_applicable": False,
            "reason": "Single kernel"
        }
    
    # For FPGA with multiple consecutive kernels, model fusion
    # Combine arithmetic ops and reduce memory traffic
    
    total_flops = 0
    total_memory_bytes = 0
    individual_latencies = []
    
    # Collect data for all kernels
    all_memory_transactions = {}
    for kernel in kernels:
        data = roofline_data[kernel]
        total_flops += data["total_flops"]
        individual_latencies.append(data["estimated_latency_us"])
        
        # Collect memory transactions
        for data_name, size in data["memory_transactions"].items():
            if data_name not in all_memory_transactions:
                all_memory_transactions[data_name] = {"producers": [], "consumers": [], "size": size}
            # Track which kernels produce/consume this data
            kernel_deps = KERNEL_DEPENDENCIES[kernel]
            if data_name in kernel_deps["inputs"]:
                all_memory_transactions[data_name]["consumers"].append(kernel)
            if data_name in kernel_deps["outputs"]:
                all_memory_transactions[data_name]["producers"].append(kernel)
    
    # Calculate reduced memory traffic:
    # - External inputs (not produced by any kernel in the group) must be loaded
    # - External outputs (not consumed by any kernel in the group, or final outputs) must be stored
    # - Intermediate data (produced and consumed within the group) can stay on-chip
    
    external_memory_bytes = 0
    intermediate_memory_bytes = 0
    
    for data_name, info in all_memory_transactions.items():
        producers = info["producers"]
        consumers = info["consumers"]
        size = info["size"]
        
        is_produced_internally = any(p in kernels for p in producers)
        is_consumed_internally = any(c in kernels for c in consumers)
        
        if not is_produced_internally:
            # External input - must load from memory
            external_memory_bytes += size
        elif is_produced_internally and not is_consumed_internally:
            # Output that goes outside the fused group - must store
            external_memory_bytes += size
        elif is_produced_internally and is_consumed_internally:
            # Check if it's also a final output (consumed by kernel outside group)
            # For now, mark as intermediate
            intermediate_memory_bytes += size
    
    # Use the first kernel's ridge point (assume same device config)
    ridge_point = roofline_data[kernels[0]]["ridge_point"]
    target_freq_mhz = roofline_data[kernels[0]]["target_freq_mhz"]
    
    # Calculate fused arithmetic intensity
    if external_memory_bytes > 0:
        fused_arithmetic_intensity = total_flops / external_memory_bytes
    else:
        fused_arithmetic_intensity = float('inf')
    
    # Derive peak metrics from ridge point
    # ridge_point = peak_flops / peak_memory_bw
    # For U55C at 250MHz: peak_memory_bw ≈ 390 GB/s, peak_flops depends on DSPs
    # From the JSON, we can derive: if ridge_point = 3.847, and we know the relationship
    peak_memory_bw_bytes_per_cycle = 1564.0  # Approximate for U55C
    peak_flops_per_cycle = ridge_point * peak_memory_bw_bytes_per_cycle
    
    # Calculate fused latency using roofline model
    memory_bound_cycles = external_memory_bytes / peak_memory_bw_bytes_per_cycle
    compute_bound_cycles = total_flops / peak_flops_per_cycle
    fused_cycles = max(memory_bound_cycles, compute_bound_cycles)
    fused_latency_us = fused_cycles / target_freq_mhz
    
    # Determine bottleneck
    if fused_arithmetic_intensity < ridge_point * 0.9:
        bottleneck = "MEMORY-BOUND"
    elif fused_arithmetic_intensity > ridge_point * 1.1:
        bottleneck = "COMPUTE-BOUND"
    else:
        bottleneck = "BALANCED"
    
    original_total_latency = sum(individual_latencies)
    
    return fused_latency_us, {
        "device": FPGA,
        "kernels": kernels,
        "fusion_applicable": True,
        "total_flops": total_flops,
        "external_memory_bytes": external_memory_bytes,
        "intermediate_memory_bytes": intermediate_memory_bytes,
        "original_memory_bytes": sum(roofline_data[k]["memory_bytes"] for k in kernels),
        "memory_reduction_ratio": 1.0 - (external_memory_bytes / max(1, sum(roofline_data[k]["memory_bytes"] for k in kernels))),
        "fused_arithmetic_intensity": fused_arithmetic_intensity,
        "ridge_point": ridge_point,
        "bottleneck": bottleneck,
        "memory_bound_cycles": memory_bound_cycles,
        "compute_bound_cycles": compute_bound_cycles,
        "fused_latency_us": fused_latency_us,
        "original_total_latency_us": original_total_latency,
        "speedup": original_total_latency / fused_latency_us if fused_latency_us > 0 else 0
    }


def analyze_fusion_opportunities(
    kernel_assignment: Dict[str, str],
    gpu_profiles: Dict[str, float],
    fpga_profiles: Dict[str, float],
    pcie_config: Dict,
    memory_transactions: Dict[str, Dict[str, int]]
) -> Dict[str, Any]:
    """
    Analyze potential speedup from kernel fusion for consecutive kernels on the same device.
    
    Args:
        kernel_assignment: Mapping of kernel name to device
        gpu_profiles: GPU kernel latencies
        fpga_profiles: FPGA kernel latencies
        pcie_config: PCIe configuration
        memory_transactions: Memory transaction data
    
    Returns:
        Analysis report with fusion opportunities and potential speedup
    """
    # Load full roofline data
    roofline_data = load_kernel_roofline_data()
    
    # Find consecutive kernel groups
    groups = find_consecutive_kernel_groups(kernel_assignment)
    
    # Calculate original latency (without fusion)
    original_latency, original_breakdown = evaluate_configuration(
        kernel_assignment, gpu_profiles, fpga_profiles, pcie_config, memory_transactions
    )
    
    # Analyze each group for fusion opportunities
    fusion_analysis = []
    total_fused_kernel_latency = 0.0
    total_original_kernel_latency = 0.0
    
    for device, kernels in groups:
        fused_latency, analysis = calculate_fused_kernel_latency(
            kernels, device, roofline_data, gpu_profiles
        )
        fusion_analysis.append(analysis)
        total_fused_kernel_latency += fused_latency
        
        if device == GPU:
            total_original_kernel_latency += sum(gpu_profiles[k] for k in kernels)
        else:
            total_original_kernel_latency += sum(fpga_profiles[k] for k in kernels)
    
    # Calculate total transfer latency (unchanged by fusion)
    transfer_latency = original_latency - total_original_kernel_latency
    
    # New total latency with fusion
    fused_total_latency = total_fused_kernel_latency + transfer_latency
    
    # Prepare report
    report = {
        "kernel_assignment": kernel_assignment,
        "consecutive_groups": [(device, kernels) for device, kernels in groups],
        "fusion_analysis": fusion_analysis,
        "original_latency_us": original_latency,
        "original_kernel_latency_us": total_original_kernel_latency,
        "fused_kernel_latency_us": total_fused_kernel_latency,
        "transfer_latency_us": transfer_latency,
        "fused_total_latency_us": fused_total_latency,
        "potential_speedup": original_latency / fused_total_latency if fused_total_latency > 0 else 0,
        "kernel_speedup": total_original_kernel_latency / total_fused_kernel_latency if total_fused_kernel_latency > 0 else 0
    }
    
    return report


def print_fusion_analysis(kernel_assignment: Dict[str, str]):
    """
    Print detailed fusion analysis for a given kernel assignment.
    """
    # Load profiling data
    gpu_profiles = load_gpu_profile()
    fpga_profiles = load_fpga_profiles()
    pcie_config = load_pcie_config()
    memory_transactions = load_memory_transactions()
    
    report = analyze_fusion_opportunities(
        kernel_assignment, gpu_profiles, fpga_profiles, pcie_config, memory_transactions
    )
    
    print("\n" + "=" * 80)
    print("KERNEL FUSION ANALYSIS")
    print("=" * 80)
    
    print("\nKernel Assignment:")
    for k, v in kernel_assignment.items():
        print(f"  {k}: {v}")
    
    print("\nConsecutive Kernel Groups:")
    for device, kernels in report["consecutive_groups"]:
        print(f"  {device}: {kernels}")
    
    print("\n" + "-" * 80)
    print("Fusion Opportunities:")
    print("-" * 80)
    
    for analysis in report["fusion_analysis"]:
        kernels = analysis["kernels"]
        device = analysis["device"]
        
        print(f"\n  Group: {kernels} on {device}")
        
        if not analysis["fusion_applicable"]:
            print(f"    Fusion not applicable: {analysis['reason']}")
        else:
            print(f"    Total FLOPs: {analysis['total_flops']:,}")
            print(f"    Original memory bytes: {analysis['original_memory_bytes']:,}")
            print(f"    External memory bytes (after fusion): {analysis['external_memory_bytes']:,}")
            print(f"    Intermediate memory bytes (saved): {analysis['intermediate_memory_bytes']:,}")
            print(f"    Memory reduction: {analysis['memory_reduction_ratio']:.1%}")
            print(f"    Original arithmetic intensity: ~{analysis['original_memory_bytes']/max(1,analysis['total_flops']):.4f} bytes/FLOP")
            print(f"    Fused arithmetic intensity: {analysis['fused_arithmetic_intensity']:.4f} FLOPs/byte")
            print(f"    Ridge point: {analysis['ridge_point']:.4f} FLOPs/byte")
            print(f"    Bottleneck: {analysis['bottleneck']}")
            print(f"    Original latency: {analysis['original_total_latency_us']:.2f} us")
            print(f"    Fused latency: {analysis['fused_latency_us']:.2f} us")
            print(f"    Kernel speedup: {analysis['speedup']:.2f}x")
    
    print("\n" + "-" * 80)
    print("Summary:")
    print("-" * 80)
    print(f"  Original total latency: {report['original_latency_us']:.2f} us")
    print(f"    - Kernel execution: {report['original_kernel_latency_us']:.2f} us")
    print(f"    - Data transfer: {report['transfer_latency_us']:.2f} us")
    print(f"  Fused total latency: {report['fused_total_latency_us']:.2f} us")
    print(f"    - Kernel execution: {report['fused_kernel_latency_us']:.2f} us")
    print(f"    - Data transfer: {report['transfer_latency_us']:.2f} us")
    print(f"\n  >>> Potential E2E Speedup: {report['potential_speedup']:.2f}x <<<")
    print(f"  >>> Kernel-only Speedup: {report['kernel_speedup']:.2f}x <<<")
    
    if report['potential_speedup'] > 1.0:
        print("\n  ✓ Fusion CAN improve performance!")
        print("\n  Recommended kernel fusions:")
        for analysis in report["fusion_analysis"]:
            if analysis["fusion_applicable"] and analysis.get("speedup", 0) > 1.0:
                kernels = analysis["kernels"]
                device = analysis["device"]
                speedup = analysis["speedup"]
                print(f"    - Fuse {kernels} on {device} (speedup: {speedup:.2f}x)")
    else:
        print("\n  ✗ Fusion may NOT improve performance (already optimal)")
    
    print("\n" + "=" * 80)
    
    return report


def get_pcie_bandwidth(size_kb: float, direction: str, pcie_config: Dict) -> float:
    """
    Get PCIe bandwidth using piecewise linear regression.
    
    Args:
        size_kb: Data size in KB
        direction: "gpu_to_fpga" or "fpga_to_gpu"
        pcie_config: PCIe configuration with piecewise linear models
    
    Returns:
        Bandwidth in GB/s
    """
    segments = pcie_config[direction]
    
    for segment in segments:
        upperbound = segment["upperbound"]
        if upperbound == -1 or size_kb <= upperbound:
            bandwidth = segment["slope"] * size_kb + segment["intercept"]
            return max(bandwidth, 0.001)  # Avoid division by zero
    
    # Default to last segment if not found
    last_segment = segments[-1]
    return last_segment["slope"] * size_kb + last_segment["intercept"]


def calculate_pcie_latency(size_kb: float, direction: str, pcie_config: Dict) -> float:
    """
    Calculate PCIe transfer latency.
    
    Args:
        size_kb: Data size in KB
        direction: "gpu_to_fpga" or "fpga_to_gpu"
        pcie_config: PCIe configuration
    
    Returns:
        Transfer latency in microseconds
    """
    bandwidth_gbps = get_pcie_bandwidth(size_kb, direction, pcie_config)
    
    # Convert size from KB to GB
    size_gb = size_kb / (1024 * 1024)
    
    # Latency = size / bandwidth (in seconds), convert to microseconds
    latency_seconds = size_gb / bandwidth_gbps
    latency_us = latency_seconds * 1e6
    
    return latency_us


def evaluate_configuration(
    kernel_assignment: Dict[str, str],
    gpu_profiles: Dict[str, float],
    fpga_profiles: Dict[str, float],
    pcie_config: Dict,
    memory_transactions: Dict[str, Dict[str, int]]
) -> Tuple[float, Dict]:
    """
    Evaluate the total latency for a given configuration.
    
    Args:
        kernel_assignment: Mapping of kernel name to device (GPU/FPGA)
        gpu_profiles: GPU kernel latencies
        fpga_profiles: FPGA kernel latencies
        pcie_config: PCIe transfer configuration
        memory_transactions: Mapping of kernel name to data sizes in bytes
    
    Returns:
        Total latency in microseconds and breakdown details
    """
    total_latency = 0.0
    breakdown = {
        "kernel_latencies": {},
        "transfer_latencies": {}
    }
    
    # Track which devices have a copy of each data
    # Each data can exist on multiple devices after being transferred
    data_locations: Dict[str, Set[str]] = {}
    
    # Process kernels in order
    for kernel in KERNELS:
        kernel_device = kernel_assignment[kernel]
        deps = KERNEL_DEPENDENCIES[kernel]
        
        # Get kernel execution latency
        if kernel_device == GPU:
            kernel_latency = gpu_profiles[kernel]
        else:
            kernel_latency = fpga_profiles[kernel]
        
        breakdown["kernel_latencies"][kernel] = {
            "device": kernel_device,
            "latency_us": kernel_latency
        }
        total_latency += kernel_latency
        
        # Calculate transfer latencies for inputs
        transfer_latency = 0.0
        transfer_details = []
        
        for input_data in deps["inputs"]:
            # Check if data exists on any device
            if input_data not in data_locations:
                # Data is instantiated on the kernel's device for the first time
                data_locations[input_data] = {kernel_device}
                continue
            
            current_locations = data_locations[input_data]
            
            if kernel_device not in current_locations:
                # Need to transfer data to kernel's device
                # Get data size in bytes from memory_transactions, convert to KB
                data_sizes = memory_transactions.get(kernel, {})
                size_bytes = data_sizes.get(input_data, 1024)  # Default to 1KB if not found
                size_kb = size_bytes / 1024.0
                
                # Determine transfer direction
                if kernel_device == GPU:
                    # Transfer from FPGA to GPU
                    source_device = FPGA
                    direction = "fpga_to_gpu"
                else:
                    # Transfer from GPU to FPGA
                    source_device = GPU
                    direction = "gpu_to_fpga"
                    
                    # Skip GPU->FPGA transfer for "memory" data
                    if input_data == "memory":
                        continue
                
                # Check if source has the data
                if source_device not in current_locations:
                    continue
                
                latency = calculate_pcie_latency(size_kb, direction, pcie_config)
                transfer_latency += latency
                transfer_details.append({
                    "data": input_data,
                    "from": source_device,
                    "to": kernel_device,
                    "size_kb": size_kb,
                    "latency_us": latency
                })
                
                # Register the copy on the target device
                data_locations[input_data].add(kernel_device)
        
        if transfer_details:
            breakdown["transfer_latencies"][f"{kernel}_inputs"] = transfer_details
        
        total_latency += transfer_latency
        
        # Outputs are produced on the kernel's device
        for output_data in deps["outputs"]:
            data_locations[output_data] = {kernel_device}
    
    breakdown["total_latency_us"] = total_latency
    
    return total_latency, breakdown


def find_optimal_assignment():
    """Find the optimal kernel and data placement assignment."""
    
    # Load profiling data
    print("Loading profiling data...")
    gpu_profiles = load_gpu_profile()
    fpga_profiles = load_fpga_profiles()
    pcie_config = load_pcie_config()
    memory_transactions = load_memory_transactions()
    
    print("\nGPU Kernel Latencies (us):")
    for k, v in gpu_profiles.items():
        print(f"  {k}: {v:.2f}")
    
    print("\nFPGA Kernel Latencies (us):")
    for k, v in fpga_profiles.items():
        print(f"  {k}: {v:.2f}")
    
    print("\nMemory Transactions (bytes):")
    for kernel, transactions in memory_transactions.items():
        print(f"  {kernel}: {transactions}")
    
    # Generate all possible kernel assignments
    kernel_devices = [GPU, FPGA]
    kernel_assignments = list(itertools.product(kernel_devices, repeat=len(KERNELS)))
    
    print(f"\nTotal kernel assignment combinations: {len(kernel_assignments)}")
    
    best_latency = float('inf')
    best_config = None
    best_breakdown = None
    
    all_results = []
    
    # Traverse all possibilities
    for ka in kernel_assignments:
        kernel_assignment = dict(zip(KERNELS, ka))
        
        latency, breakdown = evaluate_configuration(
            kernel_assignment,
            gpu_profiles,
            fpga_profiles,
            pcie_config,
            memory_transactions
        )
        
        all_results.append({
            "kernel_assignment": kernel_assignment.copy(),
            "total_latency_us": latency,
            "breakdown": breakdown
        })
        
        if latency < best_latency:
            best_latency = latency
            best_config = {
                "kernel_assignment": kernel_assignment.copy()
            }
            best_breakdown = breakdown
    
    # Sort all results by latency
    all_results.sort(key=lambda x: x["total_latency_us"])
    
    # Print results
    print("\n" + "=" * 80)
    print("OPTIMAL CONFIGURATION FOUND")
    print("=" * 80)
    
    print("\nBest Kernel Assignment:")
    for kernel, device in best_config["kernel_assignment"].items():
        print(f"  {kernel}: {device}")
    
    print(f"\nTotal Latency: {best_latency:.2f} us")
    
    print("\nLatency Breakdown:")
    print("\n  Kernel Execution Latencies:")
    for kernel, info in best_breakdown["kernel_latencies"].items():
        print(f"    {kernel} ({info['device']}): {info['latency_us']:.2f} us")
    
    print("\n  Transfer Latencies:")
    if best_breakdown["transfer_latencies"]:
        for stage, transfers in best_breakdown["transfer_latencies"].items():
            print(f"    {stage}:")
            for t in transfers:
                print(f"      {t['data']}: {t['from']} -> {t['to']}, "
                      f"{t['size_kb']:.1f} KB, {t['latency_us']:.4f} us")
    else:
        print("    (no transfers needed)")
    
    # Print top 5 configurations
    print("\n" + "=" * 80)
    print("TOP 5 CONFIGURATIONS")
    print("=" * 80)
    
    for i, result in enumerate(all_results[:5]):
        print(f"\n#{i+1} - Latency: {result['total_latency_us']:.2f} us")
        print("  Kernels:", result["kernel_assignment"])
    
    # Compare with baseline (all GPU)
    all_gpu_latency, _ = evaluate_configuration(
        {k: GPU for k in KERNELS}, gpu_profiles, fpga_profiles, pcie_config, memory_transactions
    )
    all_fpga_latency, _ = evaluate_configuration(
        {k: FPGA for k in KERNELS}, gpu_profiles, fpga_profiles, pcie_config, memory_transactions
    )
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINES")
    print("=" * 80)
    print(f"\nAll-GPU baseline: {all_gpu_latency:.2f} us")
    print(f"All-FPGA baseline: {all_fpga_latency:.2f} us")
    print(f"Optimal configuration: {best_latency:.2f} us")
    if best_latency > 0:
        print(f"Speedup over All-GPU: {all_gpu_latency / best_latency:.2f}x")
    
    # Save results to file
    output_path = os.path.join(BASE_DIR, "optimal_assignment_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            "best_configuration": best_config,
            "best_latency_us": best_latency,
            "best_breakdown": best_breakdown,
            "all_gpu_latency_us": all_gpu_latency,
            "all_fpga_latency_us": all_fpga_latency,
            "top_10_configurations": all_results[:10]
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return best_config, best_latency, best_breakdown


def print_specific_configuration(
    kernel_assignment: Dict[str, str]
):
    """Print detailed latency breakdown for a specific configuration."""
    
    # Load profiling data
    gpu_profiles = load_gpu_profile()
    fpga_profiles = load_fpga_profiles()
    pcie_config = load_pcie_config()
    memory_transactions = load_memory_transactions()
    
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print("\nKernel Assignment:")
    for k, v in kernel_assignment.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 80)
    print("LATENCY BREAKDOWN")
    print("=" * 80)
    
    total_latency = 0.0
    
    # Track which devices have a copy of each data
    data_locations: Dict[str, Set[str]] = {}
    
    for kernel in KERNELS:
        kernel_device = kernel_assignment[kernel]
        deps = KERNEL_DEPENDENCIES[kernel]
        
        # Kernel latency
        if kernel_device == GPU:
            kernel_latency = gpu_profiles[kernel]
        else:
            kernel_latency = fpga_profiles[kernel]
        
        print(f"\n{kernel} ({kernel_device}):")
        print(f"  Kernel execution: {kernel_latency:.4f} us")
        total_latency += kernel_latency
        
        # Input transfers
        transfer_total = 0.0
        for inp in deps["inputs"]:
            if inp not in data_locations:
                print(f"  [INFO] {inp} is instantiated on {kernel_device} for the first time.")
                data_locations[inp] = {kernel_device}
            
            elif kernel_device not in data_locations[inp]:
                # Get data size in bytes from memory_transactions, convert to KB
                data_sizes = memory_transactions.get(kernel, {})
                size_bytes = data_sizes.get(inp, 1024)  # Default to 1KB if not found
                size_kb = size_bytes / 1024.0
                
                if kernel_device == GPU:
                    source_device = FPGA
                    direction = "fpga_to_gpu"
                else:
                    source_device = GPU
                    direction = "gpu_to_fpga"
                    
                    # Skip GPU->FPGA for memory
                    if inp == "memory":
                        print(f"  Skip {inp}: GPU -> FPGA not allowed for memory")
                        continue
                
                if source_device not in data_locations[inp]:
                    print(f"  WARNING: {inp} not available on {source_device}")
                    continue
                
                lat = calculate_pcie_latency(size_kb, direction, pcie_config)
                print(f"  Transfer {inp}: {source_device} -> {kernel_device}, {size_kb} KB, {lat:.4f} us")
                transfer_total += lat
                
                # Register copy on target device
                data_locations[inp].add(kernel_device)
        
        if transfer_total > 0:
            print(f"  Transfer subtotal: {transfer_total:.4f} us")
        total_latency += transfer_total
        
        # Update output locations (produced on kernel's device)
        for out in deps["outputs"]:
            data_locations[out] = {kernel_device}
        
        # Show current data state
        print(f"  Data state after kernel: {dict((k, list(v)) for k, v in data_locations.items())}")
    
    print("\n" + "=" * 80)
    print(f"TOTAL LATENCY: {total_latency:.4f} us")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--specific":
        # User-specified configuration
        kernel_assignment = {
            "paged_kv_index_builder": GPU,
            "inner_product": FPGA,
            "topk_retrieval": FPGA,
            "block_sparse_attention": GPU
        }
        print_specific_configuration(kernel_assignment)
    else:
        # Find optimal assignment and analyze fusion opportunities
        best_config, best_latency, best_breakdown = find_optimal_assignment()
        print_fusion_analysis(best_config["kernel_assignment"])
