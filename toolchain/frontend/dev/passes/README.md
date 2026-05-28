# PythonDispatchPass — Source-Level Pass for Python Kernel Detection

## Overview

The `PythonDispatchPass` is a **source-level, regex-based** pass that scans `.cpp`
implementation files (typically `example_kernel/*.cpp`) for `PY_FUNC` annotations
on kernel methods. It then generates a **Python pipeline script** that orchestrates
the discovered steps—calling Python functions directly for `PY_FUNC`-annotated
kernels or importing Pybind11-exported C++ modules for the rest.

No LLVM, MLIR, or libclang dependency is required.

## How It Works

### 1. The PY_FUNC Attribute

The `PY_FUNC` macro is defined **only in `.cpp` files** (not in headers),
because different users may or may not implement a given kernel in Python.
The header declares the kernel method as a normal virtual function; the
`.cpp` file optionally annotates the implementation:

```cpp
// In example_kernel/bm25_dataset_builder.cpp
#include <Python.h>
#define PY_FUNC(func_name) __attribute__((annotate("py_func=" func_name)))

PY_FUNC("bm25_loader_xrt.fpga_retriever_setup")
void BM25DatasetBuilder::run_cpu_kernel(
    const TextDBData& raw, BM25IndexMemory& mem) { ... }
```

```cpp
// In example_kernel/bm25_fpga_retrieval.cpp
PY_FUNC("launch_bm25.fpga_retriver_launch")
void FusedBM25Retrieval::run_fpga_kernel(
    const BM25IndexMemory& mem, const BM25Query& q, TopKIndex& idx) { ... }
```

```cpp
// In example_kernel/rag_apply_memory.cpp
PY_FUNC("rag_pipeline.rag_apply_memory")
void RAGApplyMemory::run_gpu_kernel(
    const TextDBData& data, const TopKIndex& idx,
    const TextInputOutputData<int>& in, TextInputOutputData<int>& out) { ... }
```

Headers use `void*` for Python objects to avoid `#include <Python.h>`;
`.cpp` files `static_cast<PyObject*>(...)` as needed.

### 2. What the Pass Scans

The pass reads each `.cpp` file and matches two patterns with regex:

1. **Annotated kernels** — `PY_FUNC("module.func")` on the line(s) directly
   before `void ClassName::run_{cpu,gpu,fpga}_kernel(`.
2. **Plain kernels** — `void ClassName::run_{cpu,gpu,fpga}_kernel(` with no
   preceding annotation.

It also infers the step type (`BuildMemory`, `FusedComputeScoreAndRetrieval`,
`ApplyMemory`, …) from `#include` directives or class-name heuristics.

### 3. Compilation Strategies

**Mixed Mode (PY_FUNC annotations found)**

Output: a single Python file (`heteromm_pipeline.py`) that:
- Imports Python functions named in PY_FUNC annotations directly.
- Imports a Pybind11 module (`heteromm_cpp_kernels`) for any pure-C++ steps.
- Provides a `HeteroMMPipeline` class with per-step methods, convenience
  wrappers (`build_memory`, `retrieve`, `apply_memory`), and a `run()` method
  mirroring the flow of `rag_test/rag_pipeline.py`.

**Pure C++ Mode (no PY_FUNC annotations)**

Output: a Python file that imports a single Pybind11 module
(`heteromm_pipeline`) wrapping the whole pipeline.

### 4. Python Dependency Tracking

The pass extracts `module.function` references from annotations:

| Module              | Function                 |
|---------------------|--------------------------|
| `bm25_loader_xrt`  | `fpga_retriever_setup`   |
| `launch_bm25`      | `fpga_retriver_launch`   |
| `rag_pipeline`      | `rag_apply_memory`       |

These appear as `from module import function` in the generated script.

## Usage

### Building the Driver

```bash
g++ -std=c++17 -o python_dispatch_pass \
    python_dispatch_pass_driver.cpp \
    -I../../ \
    -lstdc++fs
```

### Running the Pass

```bash
./python_dispatch_pass \
    --base-dir ../steps \
    --source example_kernel/bm25_dataset_builder.cpp \
    --source example_kernel/bm25_fpga_retrieval.cpp \
    --source example_kernel/rag_apply_memory.cpp \
    --output-dir ./generated \
    --verbose
```

### Output

```
generated/
└── heteromm_pipeline.py    # Python pipeline script (mixed or pure-C++ mode)
```

### Generated Pipeline Script (mixed mode, excerpt)

```python
#!/usr/bin/env python3
"""Auto-generated HeteroMM pipeline."""

from bm25_loader_xrt import fpga_retriever_setup
from launch_bm25 import fpga_retriver_launch
from rag_pipeline import rag_apply_memory

class HeteroMMPipeline:
    def build_memory(self, *args, **kwargs):
        return fpga_retriever_setup(*args, **kwargs)

    def retrieve(self, *args, **kwargs):
        return fpga_retriver_launch(*args, **kwargs)

    def apply_memory(self, *args, **kwargs):
        return rag_apply_memory(*args, **kwargs)

    def run(self, query, **kwargs):
        self._fpga_setup = self.build_memory(...)
        docs, latency = self.retrieve(self._fpga_setup, query, ...)
        prompt = self.apply_memory(query, docs, ...)
        return prompt
```

## Relationship to the RAG Pipeline

The RAG pipeline from `rag_test/rag_pipeline.py` is the **reference implementation**
that this pass aims to regenerate from annotated `.cpp` kernel files. The mapping:

| RAG Pipeline Stage     | Step Class             | Kernel | PY_FUNC Target                         |
|------------------------|------------------------|--------|-----------------------------------------|
| Dataset loading + FPGA | `BM25DatasetBuilder`   | CPU    | `bm25_loader_xrt.fpga_retriever_setup` |
| BM25 scoring + top-K   | `FusedBM25Retrieval`   | FPGA   | `launch_bm25.fpga_retriver_launch`     |
| RAG prompt building    | `RAGApplyMemory`       | GPU    | `rag_pipeline.rag_apply_memory`        |

The deploy header `frontend/deploy/simple_rag.h` wires these three steps together
through the `SimpleRAG` class (following the `PagedAttention` pattern), which
inherits from `MemoryManager`.

## File Layout

```
frontend/dev/passes/
├── README.md                        # This file
├── python_dispatch_pass.h           # Pass implementation (header-only, regex-based)
└── python_dispatch_pass_driver.cpp  # CLI driver

frontend/dev/steps/
├── build_memory.h                   # BM25DatasetBuilder declared here (no PY_FUNC)
├── apply_memory.h                   # RAGApplyMemory declared here (no PY_FUNC)
├── fused_steps/
│   └── template.h                   # FusedBM25Retrieval declared here (no PY_FUNC)
└── example_kernel/
    ├── bm25_dataset_builder.cpp     # PY_FUNC + implementations for BM25DatasetBuilder
    ├── bm25_fpga_retrieval.cpp      # PY_FUNC + implementations for FusedBM25Retrieval
    └── rag_apply_memory.cpp         # PY_FUNC + implementations for RAGApplyMemory

frontend/deploy/
├── memory_manager.h                 # Base MemoryManager template
├── paged_attention.h                # Reference deploy header (PagedAttention pattern)
└── simple_rag.h                     # SimpleRAG — RAG deploy header (same pattern)
```
