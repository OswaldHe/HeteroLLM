# HeteroMM - Heterogeneous Memory Management Framework

A C++ framework for GPU-FPGA heterogeneous systems designed for LLM serving with memory retrieval capabilities.

## Project Structure

```
├── frontend/
│   ├── hetero_llm.h              # Main include header
│   ├── dev/                      # Development interfaces
│   │   ├── dev.h                 # Dev unified header
│   │   ├── types/                # Data type definitions
│   │   │   ├── base_types.h      # Base type with DataLocation enum
│   │   │   ├── memory.h          # Memory types (FlatIndexMemory, MinMaxIndexMemory, LaCTMLPMemory)
│   │   │   ├── query.h           # Query types (VectorQuery, MultiHeadQuery, BM25Query)
│   │   │   ├── score.h           # Score types (VectorScore, MatrixScore, LossScore)
│   │   │   ├── retrieved_index.h # Index types (TopKIndex, ThresholdBitmapIndex)
│   │   │   ├── retrieved_data.h  # Retrieved data (KVCacheData, TextDBData, ParametrizedData)
│   │   │   └── target_data.h     # Input/Output data (VectorInputOutputData, MatrixInputOutputData)
│   │   ├── steps/                # Pipeline step definitions
│   │   │   ├── util.h            # KernelType enum (CPU, GPU, FPGA)
│   │   │   ├── build_memory.h    # BuildMemory step + PagedKVIndexBuilder
│   │   │   ├── compute_score.h   # ComputeScore step + InnerProductCompute
│   │   │   ├── memory_retrieval.h # MemoryRetrieval step + TopKRetrieval, ThresholdRetrieval
│   │   │   ├── apply_memory.h    # ApplyMemory step + BlockSparseAttention
│   │   │   ├── example_kernel/   # Reference kernel implementations
│   │   │   └── fused_steps/      # Fused step templates (e.g., FusedComputeScoreAndRetrieval)
│   │   └── unittest/             # Unit tests with doctest
│   │       ├── doctest.h
│   │       ├── Makefile
│   │       ├── test_innerproduct.cpp
│   │       ├── test_topk_retrieval.cpp
│   │       ├── test_threshold_retrieval.cpp
│   │       ├── test_paged_kv_index_builder.cpp
│   │       └── test_block_sparse_attention.cpp
│   └── deploy/                   # Deployment interfaces
│       ├── deploy.h
│       ├── memory_manager.h
│       └── default_manager.h
├── backend/                      # Backend (placeholders)
│   ├── compilation/
│   ├── exploration/
│   ├── profiling/
│   ├── modeling/
│   └── scheduler/
└── examples/
    ├── example_vector_retrieval.cpp
    └── example_test.cpp
```

## Pipeline Architecture

The framework implements a 4-step memory retrieval pipeline:

```
RetrievedData ──► BuildMemory ──► Memory
                                    │
                                    ▼
Query ─────────────────────────► ComputeScore ──► Score
                                                    │
                                                    ▼
                                            MemoryRetrieval ──► RetrievedIndex
                                                                    │
RetrievedData ─────────────────────────────────────────────────────►│
TargetData (Input) ────────────────────────────────────────────────►│
                                                                    ▼
                                                            ApplyMemory ──► TargetData (Output)
```

### Data Types

All types inherit from `BaseType<T>` which provides device location tracking (`SYS_DRAM`, `GPU_MEM`, `FPGA_MEM`).

| Base Type | Implementations | Description |
|-----------|-----------------|-------------|
| `Memory<T>` | `FlatIndexMemory<S>`, `MinMaxIndexMemory<S>`, `LaCTMLPMemory` | Indexed memory for retrieval |
| `Query<T>` | `VectorQuery<S>`, `MultiHeadQuery<S>`, `BM25Query` | Search queries |
| `Score<T>` | `VectorScore<S>`, `MatrixScore<S>`, `LossScore` | Similarity/relevance scores |
| `RetrievedIndex<T>` | `TopKIndex`, `ThresholdBitmapIndex` | Selected entry indices |
| `RetrievedData<T>` | `KVCacheData<S>`, `TextDBData`, `ParametrizedData` | Data to be retrieved from |
| `TargetData<T>` | `VectorInputOutputData<S>`, `MatrixInputOutputData<S>`, `TextInputOutputData<S>` | Pipeline input/output |

### Pipeline Steps

Each step is templated on its input/output types and provides four kernel implementations:

| Kernel | Method | Purpose |
|--------|--------|---------|
| **CPU** | `run_cpu_kernel()` | CPU implementation |
| **GPU** | `run_gpu_kernel()` | GPU (CUDA/HIP) implementation |
| **FPGA** | `run_fpga_kernel()` | FPGA accelerated kernel |
| **Test** | `run_test_kernel()` | Functional verification kernel |

### Built-in Step Implementations

| Step Class | Description |
|------------|-------------|
| `PagedKVIndexBuilder` | Builds paged KV cache index from `KVCacheData` |
| `InnerProductCompute` | Dot product between `VectorQuery` and `FlatIndexMemory` |
| `TopKRetrieval` | Selects top-k highest scoring entries |
| `ThresholdRetrieval` | Selects entries above a threshold (bitmap output) |
| `BlockSparseAttention` | Sparse attention over selected KV cache blocks |

### Fused Steps

For performance optimization, fused steps combine multiple pipeline stages:

```cpp
template<typename MemoryType, typename QueryType, typename IndexType>
class FusedComputeScoreAndRetrieval { ... };
```

## Usage

### Quick Start

```cpp
#include "dev/dev.h"

using namespace heteromm;

int main() {
    // 1. Create memory and query
    std::vector<std::vector<float>> mem_data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    data_type::FlatIndexMemory<float> memory(mem_data);
    
    std::vector<float> query_data = {1.0f, 1.0f};
    data_type::VectorQuery<float> query(query_data);
    
    // 2. Compute scores
    step::InnerProductCompute compute;
    compute.set_current_kernel(step::KernelType::CPU);
    data_type::VectorScore<float> score({});
    compute.execute(memory, query, score);
    
    // 3. Retrieve top-k
    step::TopKRetrieval retrieval(1);  // k=1
    data_type::TopKIndex indices({});
    retrieval.execute(score, indices);
    
    return 0;
}
```

### Custom Step Implementation

```cpp
class MyScoreCompute : public step::ComputeScore<
    data_type::FlatIndexMemory<float>,
    data_type::VectorQuery<float>,
    data_type::VectorScore<float>
> {
protected:
    void run_cpu_kernel(
        const data_type::FlatIndexMemory<float>& memory,
        const data_type::VectorQuery<float>& query,
        data_type::VectorScore<float>& score
    ) override {
        // Your CPU implementation
    }
    
    void run_gpu_kernel(...) override { /* GPU impl */ }
    void run_fpga_kernel(...) override { /* FPGA impl */ }
    void run_test_kernel(...) override { /* Test impl */ }
};
```

### Functional Testing

The framework supports built-in functional testing by comparing kernel output against ground truth:

```cpp
// Pass ground truth as the output parameter with run_functional_test=true
std::vector<float> ground_truth = {3.0f, 7.0f};
data_type::VectorScore<float> expected(ground_truth);

step::InnerProductCompute compute;
int result = compute.execute(memory, query, expected, 
                             /*run_functional_test=*/true, 
                             /*verbose=*/true);
// result == 0 means test passed
```

### Unit Tests with doctest

```bash
cd frontend/dev/unittest
make all
make test        # Run all tests
make test-verbose  # Run with verbose output
```

Example test:

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../steps/compute_score.h"

TEST_CASE("Basic dot product computation") {
    std::vector<std::vector<float>> mem_data = {
        {1.0f, 0.0f}, {0.0f, 1.0f}
    };
    data_type::FlatIndexMemory<float> memory(mem_data);
    
    std::vector<float> query_data = {1.0f, 2.0f};
    data_type::VectorQuery<float> query(query_data);
    
    std::vector<float> ground_truth = {1.0f, 2.0f};
    data_type::VectorScore<float> score(ground_truth);
    
    step::InnerProductCompute compute;
    CHECK(compute.execute(memory, query, score, true) == 0);
}
```

## Building

```bash
# Build unit tests
cd frontend/dev/unittest
make all

# Run tests
make test
```

## Backend (Future Work)

The backend will provide:

- **Compilation**: HLS compilation for FPGA, CUDA/HIP for GPU
- **Exploration**: Design space exploration for optimal configurations
- **Profiling**: Performance measurement and analysis
- **Modeling**: Performance prediction models
- **Scheduler**: Dynamic runtime device selection

## License

[Add your license here]
