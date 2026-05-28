# HeteroLLM

HeteroLLM is an AI infrastructure research framework for serving LLM memory
workloads across CPUs, GPUs, and FPGAs. The repository combines a typed C++
pipeline frontend, schedule-driven deployment interfaces, FPGA/GPU kernel
implementations, roofline and placement modeling, RAG experiments, and
FPGA-GPU peer-to-peer communication paths.

The core abstraction is a heterogeneous memory-management pipeline:

```text
RetrievedData -> BuildMemory -> Memory
Memory + Query -> ComputeScore -> Score
Score -> MemoryRetrieval -> RetrievedIndex
RetrievedData + RetrievedIndex + TargetData -> ApplyMemory -> TargetData
```

This maps naturally to LLM systems that need retrieval, sparse attention,
KV-cache indexing, BM25/RAG lookup, or expert routing, while letting each stage
run on the best available execution target.

## Project Structure

```text
.
├── toolchain/
│   ├── frontend/              # HeteroMM C++ APIs, deploy managers, passes, tests
│   ├── backend/               # Profiling, roofline modeling, assignment analysis
│   └── rag_test/              # BM25/RAG pipeline and FPGA-backed BM25 loader
├── kernels/
│   ├── bm25/                  # BM25 top-k indexer kernels and bitstreams
│   ├── seerattention/         # SeerAttention indexer variants
│   ├── lserve/                # LServe indexer variants
│   ├── moe/                   # DeepSeek-style MoE FPGA/GPU kernels
│   ├── deepseek_engram/       # Engram GPU-FPGA experiments
│   └── dsa_indexer_lut/       # LUT-based DSA indexer
├── p2p_comm/
│   ├── u55c_rocm_p2p/         # Xilinx U55C + AMD MI210 P2P demos
│   └── python_api/            # pybind11 P2P API for Python workflows
├── aws_ec2_ena/               # AWS FPGA preview experiment artifacts
└── README.md
```

## Architecture

HeteroLLM is organized as a framework stack rather than a single benchmark.

| Layer | Role |
| --- | --- |
| Frontend types | C++ data abstractions for memory, query, score, retrieved indices, retrieved data, and target data. |
| Pipeline steps | `BuildMemory`, `ComputeScore`, `MemoryRetrieval`, and `ApplyMemory` interfaces with CPU/GPU/FPGA dispatch hooks. |
| Deploy managers | `MemoryManager` orchestrates full pipelines and selects devices from JSON schedules. |
| Passes | Source-level Python dispatch pass detects `PY_FUNC` annotations and emits Python pipeline wrappers. |
| Backend modeling | Roofline analysis, design-space profiling, kernel/data placement, and PCIe transfer modeling. |
| Kernels | TAPA/Vitis HLS FPGA kernels, HIP/Torch GPU kernels, and prebuilt `.xclbin` artifacts for selected flows. |
| Communication | XRT + ROCm P2P buffer management for FPGA HBM to GPU memory paths. |

## Main Capabilities

- Typed C++ memory pipeline for heterogeneous LLM serving.
- Static or schedule-driven CPU/GPU/FPGA dispatch per pipeline stage.
- Built-in examples for paged KV indexing, inner product scoring, top-k and
  threshold retrieval, block-sparse attention, BM25 retrieval, and RAG prompt
  application.
- FPGA kernels for BM25, SeerAttention, LServe, MoE, and Engram-style
  GPU-FPGA execution.
- Python generation pass that bridges C++ step definitions to Python runtime
  pipelines.
- Roofline and placement models for choosing kernel/device assignments.
- ROCm/XRT peer-to-peer demos and pybind11 APIs for FPGA-GPU transfers.

## Hardware And Software Requirements

The software-only frontend tests need only a C++17 compiler and `make`.

The full heterogeneous stack is built around:

- Xilinx Alveo U55C with platform `xilinx_u55c_gen3x16_xdma_3_202210_1`
- XRT, Vitis/Vitis HLS, and TAPA
- AMD ROCm/HIP, tested in the repo against MI210-style systems
- Python 3 with packages such as `numpy`, `pybind11`, `bm25s`, `datasets`,
  `transformers`, `torch`, `vllm`, and `pyxrt` depending on the workflow

Some Makefiles assume local install paths such as `/opt/xilinx/xrt`,
`/opt/xilinx/Vitis/2024.2`, `/opt/rocm`, or a RapidStream/TAPA environment.
Adjust the corresponding environment variables before building on a different
machine.

## Quick Starts

### 1. Frontend Software Tests

Use this path when developing the HeteroMM C++ pipeline API without FPGA/GPU
hardware:

```bash
cd toolchain/frontend/dev/unittest
make test
```

This builds and runs doctest-based checks for inner product scoring, top-k
retrieval, threshold retrieval, paged KV index building, and block-sparse
attention.

### 2. Build A C++ Pipeline Step

```cpp
#include "dev/dev.h"

using namespace heteromm;

int main() {
    std::vector<std::vector<float>> mem_data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    data_type::FlatIndexMemory<float> memory(mem_data);

    std::vector<float> query_data = {1.0f, 1.0f};
    data_type::VectorQuery<float> query(query_data);

    data_type::VectorScore<float> score({});
    step::InnerProductCompute compute;
    compute.set_current_kernel(step::KernelType::CPU);
    compute.execute(memory, query, score);

    data_type::TopKIndex indices({});
    step::TopKRetrieval retrieval(1);
    retrieval.execute(score, indices);
}
```

### 3. Run The RAG Prototype

```bash
cd toolchain/rag_test
pip install -r requirements.txt
python rag_pipeline.py --mode simple --question "What is machine learning?"
```

The RAG path uses BM25S for retrieval and HuggingFace/vLLM for generation. With
XRT/PyXRT and a compatible bitstream, the BM25 stage can be backed by FPGA
execution.

### 4. Compile And Run BM25 Loader Tests

```bash
cd toolchain/rag_test
make
make csim
```

For FPGA execution:

```bash
make run_xrt
```

The hardware path expects a BM25 `.xclbin` and exported BM25 data under
`toolchain/rag_test/export`.

### 5. Try FPGA-GPU P2P Transfer

```bash
cd p2p_comm/u55c_rocm_p2p
source env.sh
make simple
./p2p_simple --fpga 81:00.1 --gpu 0 --size 64
```

This path creates XRT P2P buffers on the FPGA and registers them with ROCm so
GPU kernels can read from or write to FPGA HBM over PCIe.

### 6. Install The Python P2P API

```bash
cd p2p_comm/python_api
pip install -e .
python examples/basic_transfer.py --fpga-bdf 81:00.1
```

The package exposes `FPGADevice`, `GPUDevice`, and P2P buffer operations through
the `heteromem_p2p` module.

## Schedule-Driven Deployment

`toolchain/frontend/deploy/memory_manager.h` provides a reusable
`MemoryManager` template with three public entry points:

- `build_memory(...)`: build an index or memory structure from raw retrieved data.
- `manage_memory_and_apply(...)`: run query, retrieval, and apply stages against existing memory.
- `build_and_apply_memory(...)`: run the full pipeline.

Schedules are JSON rules mapping problem sizes to device choices. Example:

```json
{
  "manage_memory_and_apply": [
    {
      "retrieved_data": 4096,
      "memory": 4096,
      "query": 1,
      "output": 1,
      "config": ["fpga", "fpga", "gpu"]
    }
  ]
}
```

The checked-in example lives at `toolchain/frontend/deploy/schedule.json`.

## Python Dispatch Pass

The frontend includes a source-level pass that detects annotations such as:

```cpp
PY_FUNC("launch_bm25.fpga_retriver_launch")
void FusedBM25Retrieval::run_fpga_kernel(...);
```

It can generate Python pipeline wrappers that call native Python functions for
annotated steps and pybind11-exported C++ modules for unannotated steps.

```bash
cd toolchain/frontend/dev/passes
make
```

Generated output is written under `toolchain/frontend/dev/passes/generated/`.

## Modeling And Profiling

Backend tools estimate and explore heterogeneous execution plans:

```bash
cd toolchain/backend/modeling
python roofline_analyzer.py --profile-target profile_innerproduct --json
python optimal_assignment.py
```

Relevant inputs are stored in:

- `toolchain/backend/modeling/fpga_config/`
- `toolchain/backend/modeling/kernel_profile_results/`
- `toolchain/backend/modeling/pcie_config/`
- `toolchain/backend/profiling/design_space.json`

## Kernel Families

| Directory | Description |
| --- | --- |
| `kernels/bm25` | FPGA BM25 top-k indexer and testbench artifacts. |
| `kernels/seerattention` | SeerAttention threshold and token-budget indexers. |
| `kernels/lserve` | LServe indexer variants, including long-context versions. |
| `kernels/moe` | DeepSeek-style MoE FPGA kernels, int8 decode variants, and TileLang GPU experiments. |
| `kernels/deepseek_engram` | Engram-style GPU-FPGA heterogeneous execution and DeepSeek V3 benchmarks. |
| `kernels/dsa_indexer_lut` | LUT-based DSA indexer implementation. |

Most FPGA kernel directories use TAPA/Vitis build targets such as `csim`,
`hls`, or `xclbin`. Hardware synthesis targets can take hours.

## P2P Communication

`p2p_comm/u55c_rocm_p2p` demonstrates:

- FPGA-to-GPU reads from FPGA HBM.
- GPU-to-FPGA writes into FPGA P2P buffers.
- SpMV demos that consume FPGA-produced BM25 indices on the GPU.
- FPGA-side verification kernels for GPU-written data.

Typical system checks:

```bash
xbutil examine -d 81:00.1 --report platform
rocm-smi --showbus
lspci | grep -i xilinx
```

If true P2P registration fails, the demo can fall back to host-staged mapped
memory, which remains functional but has lower bandwidth.

## Development Notes

- Keep software-only changes covered by `toolchain/frontend/dev/unittest`.
- Keep hardware changes paired with C-simulation before running long synthesis jobs.
- Do not assume default tool paths are portable; most Makefiles expose
  `XILINX_XRT`, `XILINX_VITIS`, `ROCM_PATH`, or related variables.
- Several `.xclbin` files are checked in as experiment artifacts; rebuilding
  them requires the matching platform and toolchain versions.
- The top-level license for this repository is not specified yet.

## More Documentation

- `toolchain/frontend/README.md`: HeteroMM frontend API details.
- `toolchain/frontend/dev/passes/README.md`: Python dispatch pass design.
- `toolchain/rag_test/README.md`: BM25/RAG experiment usage.
- `p2p_comm/u55c_rocm_p2p/README.md`: FPGA-GPU P2P setup and troubleshooting.
- `p2p_comm/python_api/README.md`: pybind11 P2P API usage.
