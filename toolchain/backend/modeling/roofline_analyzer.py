#!/usr/bin/env python3
"""
roofline_analyzer.py - Python interface for FPGA roofline model analysis

This script provides a Python interface to analyze C++ kernel files
and estimate FPGA execution cycles using a roofline model.

Usage:
    python roofline_analyzer.py --profile-target profile_innerproduct

Or as a library:
    from roofline_analyzer import RooflineAnalyzer
    analyzer = RooflineAnalyzer()
    result = analyzer.analyze(profile_target="profile_innerproduct")
    print(f"Estimated cycles: {result.estimated_cycles}")
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any


@dataclass
class OperationCounts:
    """Operation counts extracted from LLVM IR"""
    loads: int = 0
    stores: int = 0
    fadd: int = 0
    fsub: int = 0
    fmul: int = 0
    fdiv: int = 0
    fmadd: int = 0
    iadd: int = 0
    isub: int = 0
    imul: int = 0
    idiv: int = 0
    comparisons: int = 0
    branches: int = 0
    calls: int = 0
    phi_nodes: int = 0
    getelementptr: int = 0
    
    @property
    def total_memory_ops(self) -> int:
        return self.loads + self.stores
    
    @property
    def total_flops(self) -> int:
        return self.fadd + self.fsub + self.fmul + self.fdiv + self.fmadd * 2
    
    @property
    def total_int_ops(self) -> int:
        return self.iadd + self.isub + self.imul + self.idiv
    
    def to_dict(self) -> Dict[str, int]:
        return {
            'loads': self.loads,
            'stores': self.stores,
            'fadd': self.fadd,
            'fsub': self.fsub,
            'fmul': self.fmul,
            'fdiv': self.fdiv,
            'fmadd': self.fmadd,
            'iadd': self.iadd,
            'isub': self.isub,
            'imul': self.imul,
            'idiv': self.idiv,
            'total_memory_ops': self.total_memory_ops,
            'total_flops': self.total_flops,
            'total_int_ops': self.total_int_ops,
        }


@dataclass
class FpgaConfig:
    """FPGA device configuration"""
    lut: int = 0
    slr: int = 0
    bram: int = 0
    uram: int = 0
    dsp: int = 0
    off_chip_size_gb: float = 0.0
    off_chip_bw_gb: float = 0.0
    off_chip_port: int = 0
    dsp_version: str = ""
    dsp_ops: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Computed metrics
    peak_memory_bw_bytes_per_cycle: float = 0.0
    peak_flops_per_cycle: float = 0.0
    
    def compute_peak_metrics(self, target_freq_mhz: float = 300.0):
        """Compute peak performance metrics"""
        # Peak memory bandwidth: bytes per cycle
        self.peak_memory_bw_bytes_per_cycle = (self.off_chip_bw_gb * 1e9) / (target_freq_mhz * 1e6)
        
        # Peak FLOPs per cycle based on DSP count
        if 'fmadd' in self.dsp_ops:
            dsp_per_fmadd = self.dsp_ops['fmadd']['count']
            if dsp_per_fmadd > 0:
                parallel_fmadds = self.dsp / dsp_per_fmadd
                self.peak_flops_per_cycle = parallel_fmadds * 2.0
        elif 'fmul' in self.dsp_ops:
            dsp_per_fmul = self.dsp_ops['fmul']['count']
            if dsp_per_fmul > 0:
                self.peak_flops_per_cycle = self.dsp / dsp_per_fmul
        else:
            self.peak_flops_per_cycle = self.dsp


@dataclass
class RooflineResult:
    """Roofline model analysis result"""
    function_name: str = ""
    fpga_device: str = ""
    target_freq_mhz: float = 300.0
    
    op_counts: OperationCounts = field(default_factory=OperationCounts)
    
    arithmetic_intensity: float = 0.0
    ridge_point: float = 0.0
    memory_bound_cycles: float = 0.0
    compute_bound_cycles: float = 0.0
    estimated_cycles: float = 0.0
    estimated_latency_us: float = 0.0
    bottleneck: str = "UNKNOWN"
    
    estimated_iterations: int = 1
    memory_bytes: int = 0
    memory_transactions: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'function_name': self.function_name,
            'fpga_device': self.fpga_device,
            'target_freq_mhz': self.target_freq_mhz,
            'operation_counts': self.op_counts.to_dict(),
            'roofline': {
                'arithmetic_intensity': self.arithmetic_intensity,
                'ridge_point': self.ridge_point,
                'memory_bound_cycles': self.memory_bound_cycles,
                'compute_bound_cycles': self.compute_bound_cycles,
                'estimated_cycles': self.estimated_cycles,
                'estimated_latency_us': self.estimated_latency_us,
                'bottleneck': self.bottleneck,
            },
            'estimated_iterations': self.estimated_iterations,
            'memory_bytes': self.memory_bytes,
            'memory_transactions': self.memory_transactions,
        }
    
    def print_report(self):
        """Print a formatted analysis report"""
        print("\n" + "=" * 50)
        print("     FPGA Roofline Model Analysis")
        print("=" * 50)
        
        print(f"\nFunction: {self.function_name}")
        print(f"FPGA Device: {self.fpga_device}")
        print(f"Target Frequency: {self.target_freq_mhz} MHz")
        
        print("\n--- Operation Counts ---")
        print(f"  Loads:          {self.op_counts.loads}")
        print(f"  Stores:         {self.op_counts.stores}")
        print(f"  FAdd:           {self.op_counts.fadd}")
        print(f"  FMul:           {self.op_counts.fmul}")
        print(f"  FMAdd:          {self.op_counts.fmadd}")
        print(f"  Total FLOPs:    {self.op_counts.total_flops}")
        
        print("\n--- Memory Transactions ---")
        if self.memory_transactions:
            for name, bytes_count in self.memory_transactions.items():
                print(f"  {name}: {bytes_count} bytes")
        else:
            print("  No memory transaction details available")
        
        print("\n--- Roofline Analysis ---")
        print(f"  Arithmetic Intensity: {self.arithmetic_intensity:.4f} FLOPs/byte")
        print(f"  Ridge Point:          {self.ridge_point:.4f} FLOPs/byte")
        
        print("\n--- Performance Estimation ---")
        print(f"  Memory-Bound Cycles:  {self.memory_bound_cycles:.0f}")
        print(f"  Compute-Bound Cycles: {self.compute_bound_cycles:.0f}")
        print(f"\n  >>> Estimated Cycles:  {self.estimated_cycles:.0f} <<<")
        print(f"  >>> Estimated Latency: {self.estimated_latency_us:.2f} µs <<<")
        print(f"\n  Bottleneck: {self.bottleneck}")
        
        print("\n" + "=" * 50)


class RooflineAnalyzer:
    """FPGA Roofline Model Analyzer"""
    
    def __init__(self, 
                 device_config: Optional[str] = None,
                 resource_config: Optional[str] = None):
        """
        Initialize the analyzer.
        
        Args:
            device_config: Path to device config JSON (e.g., u55c.json)
            resource_config: Path to resource config JSON (e.g., fpga_resource.json)
        """
        self.fpga_config = FpgaConfig()
        self.target_freq_mhz = 300.0
        self.config_loaded = False
        self.bw_scale = 1.0
        
        # Find default config paths
        self.modeling_dir = Path(__file__).parent
        
        if device_config is None:
            device_config = self.modeling_dir / "fpga_config" / "u55c.json"
        if resource_config is None:
            resource_config = self.modeling_dir / "fpga_config" / "fpga_resource.json"
        
        if Path(device_config).exists() and Path(resource_config).exists():
            self.load_fpga_config(str(device_config), str(resource_config))
    
    def load_fpga_config(self, device_config: str, resource_config: str) -> bool:
        """Load FPGA configuration from JSON files"""
        try:
            # Load device config
            with open(device_config, 'r') as f:
                device = json.load(f)
            
            self.fpga_config.lut = device.get('lut', 0)
            self.fpga_config.slr = device.get('slr', 0)
            self.fpga_config.bram = device.get('bram', 0)
            self.fpga_config.uram = device.get('uram', 0)
            self.fpga_config.dsp = device.get('dsp', 0)
            self.fpga_config.off_chip_size_gb = device.get('off_chip_size_gb', 0)
            self.fpga_config.off_chip_bw_gb = device.get('off_chip_bw_gb', 0)
            self.fpga_config.off_chip_port = device.get('off_chip_port', 0)
            self.fpga_config.dsp_version = device.get('dsp_version', '')
            
            # Load resource config
            with open(resource_config, 'r') as f:
                resource = json.load(f)
            
            # Extract DSP operations for the device's DSP version
            for dsp_config in resource.get('dsp', []):
                if dsp_config.get('version') == self.fpga_config.dsp_version:
                    for op in dsp_config.get('operations', []):
                        self.fpga_config.dsp_ops[op['name']] = {
                            'count': op['count'],
                            'latency': op['latency']
                        }
                    break
            
            self.fpga_config.compute_peak_metrics(self.target_freq_mhz)
            self.config_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            return False
    
    def set_target_frequency(self, freq_mhz: float):
        """Set target FPGA frequency"""
        self.target_freq_mhz = freq_mhz
        if self.config_loaded:
            self.fpga_config.compute_peak_metrics(freq_mhz)
    
    def scale_memory_bandwidth(self, scale_factor: float):
        """Scale the peak memory bandwidth by a factor (e.g., to account for overhead)"""
        self.bw_scale = scale_factor
        if self.config_loaded:
            self.fpga_config.peak_memory_bw_bytes_per_cycle *= scale_factor
    
    def run_profile_test(self, target_name: str = "profile_innerproduct") -> tuple:
        """
        Run make profile test and extract operation counts from output.
        
        Runs 'make <target_name>' in the frontend/dev/profile_test directory
        and parses the output to extract:
        - TOTAL: for arithmetic operations
        - TOTAL_MEM: for memory operations (bytes)
        - Memory transaction details for: score, memory, query, retrieved_data, 
          target_data, retrieved_index, output
        
        Args:
            target_name: The make target to run (e.g., 'profile_innerproduct')
            
        Returns:
            Tuple of (arithmetic_ops, memory_bytes, memory_transactions) extracted from the output.
            Returns (0, 0, {}) if parsing fails.
        """
        # Find the profile_test directory
        profile_test_dir = self.modeling_dir.parent.parent / 'frontend' / 'dev' / 'profile_test'
        
        if not profile_test_dir.exists():
            print(f"Error: profile_test directory not found: {profile_test_dir}", file=sys.stderr)
            return (0, 0, {})
        
        # Memory transaction categories to look for
        memory_categories = ('score', 'memory', 'query', 'retrieved_data', 'target_data', 'retrieved_index', 'output')
        
        try:
            # Run make clean first to ensure fresh build
            subprocess.run(
                ['make', 'clean'],
                cwd=str(profile_test_dir),
                capture_output=True,
                text=True
            )
            
            # Run make with the target
            result = subprocess.run(
                ['make', target_name],
                cwd=str(profile_test_dir),
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"Make failed with return code {result.returncode}", file=sys.stderr)
                print(f"stderr: {result.stderr}", file=sys.stderr)
                return (0, 0, {})
            
            # Combine stdout and stderr for parsing (profile output goes to stderr via std::clog)
            output = result.stdout + result.stderr
            
            # Parse the output to extract TOTAL, TOTAL_MEM, and memory transactions
            arithmetic_ops = 0
            memory_bytes = 0
            memory_transactions = {}
            
            for line in output.split('\n'):
                line = line.strip()
                
                # Match "TOTAL: <number>" for arithmetic operations
                if line.startswith('TOTAL:'):
                    match = re.search(r'TOTAL:\s*(\d+)', line)
                    if match:
                        arithmetic_ops = int(match.group(1))
                
                # Match "TOTAL_MEM: <number>" for memory operations
                elif line.startswith('TOTAL_MEM:'):
                    match = re.search(r'TOTAL_MEM:\s*(\d+)', line)
                    if match:
                        memory_bytes = int(match.group(1))
                
                # Match memory transaction categories: "<category>: <number> Byte"
                else:
                    for category in memory_categories:
                        if line.startswith(f'{category}:'):
                            match = re.search(rf'{category}:\s*(\d+)', line)
                            if match:
                                memory_transactions[category] = int(match.group(1))
                            break
            
            # clean up
            result = subprocess.run(
                ['make', 'clean'],
                cwd=str(profile_test_dir)
            )
            
            return (arithmetic_ops, memory_bytes, memory_transactions)
            
        except Exception as e:
            print(f"Error running profile test: {e}", file=sys.stderr)
            return (0, 0, {})
    
    def analyze(self,
                function_name: str = "run_cpu_kernel",
                profile_target: Optional[str] = None) -> RooflineResult:
        """
        Analyze a kernel and estimate FPGA execution cycles.
        
        This method runs a profile test via 'make <profile_target>' in the
        frontend/dev/profile_test directory to obtain actual operation counts.
        
        Args:
            function_name: Function to analyze (used to derive profile target if not specified)
            profile_target: The make target to run (e.g., 'profile_innerproduct').
                           If not specified, derived from function_name.
            
        Returns:
            RooflineResult with analysis data
        """
        result = RooflineResult()
        result.function_name = function_name
        result.fpga_device = self.fpga_config.dsp_version
        result.target_freq_mhz = self.target_freq_mhz
        
        if not self.config_loaded:
            print("Error: FPGA config not loaded", file=sys.stderr)
            return result
        
        # Run profile test to get operation counts
        # Use the specified profile target or derive from function name
        if profile_target:
            target_name = profile_target
        else:
            # Derive target name from function name (e.g., run_cpu_kernel -> profile_cpu)
            target_name = f"profile_{function_name.replace('run_', '').replace('_kernel', '').replace('_compute', '')}"
            # Default to profile_innerproduct if we can't derive a target
            if not target_name or target_name == "profile_":
                target_name = "profile_innerproduct"
        
        arithmetic_ops, memory_bytes, memory_transactions = self.run_profile_test(target_name)
        
        # Update operation counts from profile results
        # We store total arithmetic ops in total_flops for simplicity
        result.op_counts.fadd = arithmetic_ops  # Store total in fadd field for now
        result.memory_bytes = memory_bytes
        result.memory_transactions = memory_transactions
        
        # Compute roofline metrics using profiled values directly
        # No need to scale by iterations - profile test already gives actual values
        total_flops = arithmetic_ops
        total_bytes = memory_bytes
        result.estimated_iterations = 1  # Profile already includes all iterations
        
        # Arithmetic intensity
        if total_bytes > 0:
            result.arithmetic_intensity = total_flops / total_bytes
        
        # Ridge point
        if self.fpga_config.peak_memory_bw_bytes_per_cycle > 0:
            result.ridge_point = (self.fpga_config.peak_flops_per_cycle / 
                                  self.fpga_config.peak_memory_bw_bytes_per_cycle)
        
        # Memory-bound cycles
        if self.fpga_config.peak_memory_bw_bytes_per_cycle > 0:
            result.memory_bound_cycles = total_bytes / self.fpga_config.peak_memory_bw_bytes_per_cycle
        
        # Compute-bound cycles
        if self.fpga_config.peak_flops_per_cycle > 0:
            result.compute_bound_cycles = total_flops / self.fpga_config.peak_flops_per_cycle
        
        # Estimated cycles is the max of bounds plus pipeline latency
        max_latency = max((op['latency'] for op in self.fpga_config.dsp_ops.values()), default=0)
        result.estimated_cycles = max(result.memory_bound_cycles, result.compute_bound_cycles) + max_latency
        
        # Convert to microseconds
        result.estimated_latency_us = result.estimated_cycles / self.target_freq_mhz
        
        # Determine bottleneck
        if result.arithmetic_intensity < result.ridge_point * 0.9:
            result.bottleneck = "MEMORY-BOUND"
        elif result.arithmetic_intensity > result.ridge_point * 1.1:
            result.bottleneck = "COMPUTE-BOUND"
        else:
            result.bottleneck = "BALANCED"
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description='FPGA Roofline Model Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example:
  %(prog)s --profile-target profile_innerproduct
        '''
    )
    
    parser.add_argument('--function', default='run_cpu_kernel',
                        help='Function to analyze (default: run_cpu_kernel)')
    parser.add_argument('--profile-target', dest='profile_target',
                        help='Make target for profile test (e.g., profile_innerproduct). '
                             'If not specified, derived from function name.')
    parser.add_argument('--device', help='Device config JSON path')
    parser.add_argument('--resource', help='Resource config JSON path')
    parser.add_argument('--freq', type=float, default=250.0,
                        help='Target frequency in MHz (default: 250)')
    parser.add_argument('--bw_scale', type=float, default=0.85,
                        help='Scale factor for memory bandwidth (default: 1.0)')
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON only')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = RooflineAnalyzer(args.device, args.resource)
    
    if not analyzer.config_loaded:
        print("Error: Failed to load FPGA configuration", file=sys.stderr)
        sys.exit(1)
    
    analyzer.set_target_frequency(args.freq)
    analyzer.scale_memory_bandwidth(args.bw_scale)
    
    # Run analysis
    result = analyzer.analyze(args.function, profile_target=args.profile_target)
    
    # Output results
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        result.print_report()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()