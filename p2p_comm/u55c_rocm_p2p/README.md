# FPGA-GPU P2P Data Transfer Demo

This demo shows how to transfer data directly between the HBM of a Xilinx U55C FPGA
and the HBM of an AMD MI210 GPU using PCIe Peer-to-Peer (P2P) transfer, bypassing
system DRAM and CPU control.

## Quick Start

```bash
# 1. Setup environment
cd u55c_rocm_p2p
source env.sh

# 2. Build and run the simple P2P demo (no FPGA bitstream needed)
make simple
./p2p_simple --fpga 81:00.1 --gpu 0 --size 64

# 3. Build and run SpMV with BM25 indexer (requires indexer_bm25_fixed.xclbin)
make spmv_bm25
./spmv_bm25_demo

# 4. Build and run GPU→FPGA write demo (no FPGA bitstream needed)
make gpu_write
./gpu_to_fpga_demo
```

## System Requirements

- **FPGA**: Xilinx Alveo U55C with shell `xilinx_u55c_gen3x16_xdma_base_3`
- **GPU**: AMD Instinct MI210
- **XRT**: 2.14.418
- **ROCm**: 6.2
- **P2P**: Must be enabled on the FPGA

## PCIe Topology

For this system:

- MI210 GPU: `0000:63:00.0` (root complex 0x60, NUMA node 0)
- U55C FPGA: `0000:81:00.1` (root complex 0x80, NUMA node 1)

Since they're on different root complexes, the P2P transfer goes through the CPU's
AMD Infinity Fabric interconnect, but still bypasses system DRAM allocation.

## System Setup

### 1. Verify P2P is Enabled on FPGA

```bash
# Check P2P status
sudo xbutil examine -d 81:00.1 --report platform

# If P2P needs to be enabled (requires reboot):
sudo xbutil configure -d 81:00.1 --p2p enable
sudo reboot
```

### 2. Check IOMMU Settings (if P2P fails)

For P2P to work across root complexes, IOMMU may need special configuration:

```bash
# Check current IOMMU status
dmesg | grep -i iommu

# If IOMMU is blocking P2P, add to kernel cmdline in /etc/default/grub:
# GRUB_CMDLINE_LINUX="... amd_iommu=on iommu=pt"
# Then: sudo update-grub && sudo reboot
```

### 3. Set Device Permissions

```bash
sudo chmod 666 /dev/dri/renderD*
sudo chmod 666 /dev/xclmgmt*
sudo chmod 666 /dev/xocl*
```

## Build Instructions

### Simple P2P Demo (no bitstream needed)

```bash
source env.sh
make simple
./p2p_simple --fpga 81:00.1 --gpu 0 --size 64 --iterations 10
```

### SpMV with BM25 Indexer

End-to-end demo: FPGA runs BM25 top-K indexer, results transfer via P2P to GPU
for sparse matrix-vector multiplication.

```bash
source env.sh
make spmv_bm25
./spmv_bm25_demo    # requires indexer_bm25_fixed.xclbin
```

### GPU → FPGA P2P Write Demo

Demonstrates the reverse direction: GPU writes data into FPGA HBM via P2P.
Includes sequential write, pattern generation, scatter write, and bandwidth tests.

```bash
source env.sh
make gpu_write
./gpu_to_fpga_demo
```

### Full Demo with Custom FPGA Kernel

```bash
source env.sh
make host
./p2p_transfer --xclbin p2p_demo.xclbin --fpga 81:00.1 --gpu 0
```

To build the FPGA kernel (takes 2-4 hours):

```bash
source /opt/xilinx/Vitis/2023.2/settings64.sh
make fpga_kernel
```

## How It Works

### P2P Transfer Mechanism

1. **XRT P2P Buffer**: XRT creates a buffer with `xrt::bo::flags::p2p` flag
   - This maps a region of FPGA HBM to PCIe BAR4
   - The buffer becomes accessible via PCIe from other devices

2. **Memory Mapping**: `xrt::bo::map()` returns a CPU pointer to the P2P region
   - This pointer references FPGA HBM through PCIe MMIO

3. **ROCm Registration**: `hipHostRegister()` with `hipHostRegisterIoMemory` flag
   - Registers the FPGA BAR memory with ROCm
   - Allows GPU DMA engine to access this address space

4. **GPU DMA**: GPU kernels read/write to the registered address
   - DMA goes directly over PCIe to FPGA HBM
   - Bypasses CPU and system DRAM

### Transfer Flows

**FPGA → GPU (Read from FPGA HBM):**

```
FPGA HBM → PCIe BAR4 → PCIe Switch → GPU DMA → GPU HBM
```

**GPU → FPGA (Write to FPGA HBM):**

```
GPU HBM → GPU DMA → PCIe Switch → PCIe BAR4 → FPGA HBM
```

## Files

| File                            | Description                                         |
| ------------------------------- | --------------------------------------------------- |
| `p2p_simple.cpp`                | Simple P2P demo (no custom xclbin needed)           |
| `spmv_bm25_demo.cpp`            | SpMV with BM25 indexer — FPGA→GPU P2P demo          |
| `gpu_to_fpga_demo.cpp`          | GPU→FPGA P2P write demo (4 tests + bandwidth)       |
| `gpu_spmv.hip`                  | HIP kernels for FPGA→GPU read and SpMV              |
| `gpu_write_kernels.hip`         | HIP kernels for GPU→FPGA write                      |
| `fpga_verify_kernel.h`          | TAPA kernel: FPGA reads & verifies GPU-written data |
| `fpga_verify_tb.cpp`            | TAPA C-sim testbench for verification kernel        |
| `fpga_verify_standalone_tb.cpp` | Standalone testbench (no TAPA needed)               |
| `gpu_to_fpga_verify.cpp`        | E2E test: GPU writes via P2P, FPGA verifies         |
| `host.cpp`                      | Full host application with FPGA kernel control      |
| `fpga_kernel.cpp`               | HLS kernels for FPGA-side data movement             |
| `Makefile`                      | Build system                                        |
| `env.sh`                        | Environment setup script                            |

### Python API

A Python API with pybind11 bindings is available in `../python_api/`. It supports
both FPGA→GPU and GPU→FPGA P2P transfers. See `../python_api/README.md` for details.

## TAPA FPGA Verification Kernel

The FPGA verification kernel proves that the FPGA can correctly observe data
written by the GPU through the P2P path. It uses TAPA for C-simulation,
RTL cosimulation, and hardware synthesis.

See [TAPA Full Compilation docs](https://tapa.readthedocs.io/en/main/start/full-compilation.html)
for the general workflow.

### Prerequisites

```bash
# Ensure TAPA and Vitis are available
tapa compile --help
which v++

# Source XRT environment
source /opt/xilinx/xrt/setup.sh
```

### Stage 0: Software C-simulation

Build and run a pure software simulation of the FPGA kernel. Requires the
TAPA C++ runtime (libtapa, libfrt, libglog, libgflags).

```bash
make fpga_verify_csim
./fpga_verify_tb --count 1024
```

### Stage 1: Synthesize Kernel to XO

Translate the TAPA C++ kernel into an RTL object (.xo):

```bash
make fpga_verify_xo
# Equivalent to:
# tapa compile --top fpga_verify_top \
#   --platform xilinx_u55c_gen3x16_xdma_3_202210_1 \
#   -f fpga_verify_kernel.h -o fpga_verify_top.xo
```

### Stage 2: Fast RTL Cosimulation

Validate the synthesized RTL using TAPA's fast cosimulation. Pass the `.xo`
as the `--bitstream` argument:

```bash
make fpga_verify_cosim
# Equivalent to:
# ./fpga_verify_tb --bitstream fpga_verify_top.xo --count 1024
```

### Stage 3: Link to Hardware xclbin

Use Vitis `v++` to link the `.xo` into a hardware bitstream (takes hours):

```bash
make verify_xclbin
# Equivalent to:
# v++ --link --target hw \
#   --platform xilinx_u55c_gen3x16_xdma_3_202210_1 \
#   --output fpga_verify.xclbin fpga_verify_top.xo
```

### Stage 4: On-board GPU→FPGA Verification

Run the end-to-end test with real hardware:

```bash
make fpga_verify_hw
./gpu_to_fpga_verify --xclbin fpga_verify.xclbin --fpga 81:00.1
```

### Standalone Testbench (no TAPA needed)

For quick kernel logic verification without any TAPA dependency:

```bash
make fpga_verify_standalone
./fpga_verify_standalone_tb 1024
```

### Expected C-simulation Output

```
=== FPGA Verify GPU→FPGA P2P ===
Mode: TAPA C-simulation
Count: 1024 uint32 elements (4096 bytes)

Invoking fpga_verify_top...
I... task.h:55] running software simulation with TAPA library
  Kernel completed in ... ns

Verification:
  FPGA checksum: 3709440  expected: 3709440
  FPGA count:    1024  expected: 1024
  All 1024 elements match

=== PASS ===
```

## Performance Notes

- **First transfer** may be slower due to PCIe link training and page table setup
- **PCIe Gen3 x16** theoretical max: ~16 GB/s per direction
- **Cross-socket penalty**: Since FPGA and GPU are on different root complexes,
  bandwidth may be limited by Infinity Fabric (~25 GB/s bidirectional)
- **P2P vs fallback**: If `hipHostRegisterIoMemory` fails, transfers fall back to
  host-staged `hipHostRegisterMapped` (still functional, lower bandwidth)
- **Actual bandwidth** depends on:
  - PCIe topology and switch configuration
  - Buffer size (larger = more efficient)
  - Memory access pattern (sequential > random)

## Troubleshooting

### P2P Registration Fails

If `hipHostRegister` with `hipHostRegisterIoMemory` fails:

1. **Check IOMMU**: Try `iommu=pt` kernel parameter
2. **Check permissions**: Ensure user can access `/dev/dri/renderD*`
3. **Check P2P BAR**: Verify FPGA P2P is enabled with `xbutil examine`

The demo will fall back to standard `hipHostRegister` which uses CPU-mediated
transfers (still works, but slower).

### Low Bandwidth

1. Increase buffer size (`--size 128` or larger)
2. Check for PCIe errors: `dmesg | grep -i pcie`
3. Verify NUMA locality: use `numactl --cpunodebind=1 ./p2p_simple` to bind to FPGA's NUMA node

### Device Not Found

1. Check FPGA BDF: `lspci | grep Xilinx`
2. Check GPU: `rocm-smi --showbus`
3. Ensure XRT and ROCm are properly sourced

## Expected Output

```
========================================
Simple FPGA-GPU P2P Demo (XRT P2P BO)
========================================
FPGA BDF:      81:00.1
GPU ID:        0
Buffer Size:   64 MB
Iterations:    10
========================================

[1/5] Initializing GPU...
  GPU: AMD Instinct MI210
  PCIe: 63:00.0

[2/5] Initializing FPGA and creating P2P buffer...
  FPGA: xilinx_u55c_gen3x16_xdma_base_3
  P2P buffer: 64 MB at 0x7f1234560000

[3/5] Registering P2P buffer with ROCm...
  ✓ True P2P enabled (hipHostRegisterIoMemory)
  GPU-accessible pointer: 0x7f1234560000

[4/5] Benchmarking GPU -> FPGA transfer...
  Average: 8.45 ms, 7.58 GB/s
  ✓ Verification passed

[5/5] Benchmarking FPGA -> GPU transfer...
  Average: 9.12 ms, 7.02 GB/s
  ✓ Verification passed

========================================
Summary
========================================
Buffer size:        64 MB
P2P mode:           True P2P
GPU -> FPGA:        7.58 GB/s (8.45 ms)
FPGA -> GPU:        7.02 GB/s (9.12 ms)
========================================
```
