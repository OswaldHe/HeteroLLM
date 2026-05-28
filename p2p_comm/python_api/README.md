# HeteroMem P2P Python API

Python bindings for FPGA-GPU P2P communication using XRT and ROCm.

## Installation

```bash
cd python_api
pip install -e .
```

## Quick Start

```python
import heteromem_p2p as p2p
import numpy as np

# Initialize devices
fpga = p2p.FPGADevice(bdf="81:00.1")
gpu = p2p.GPUDevice(device_id=0)

# Create P2P buffer on FPGA (64 uint32 indices)
buffer = fpga.create_buffer(count=64, dtype=np.dtype('uint32'))

# Generate indices on FPGA
fpga.generate_indices(buffer=buffer, count=64, mode='sequential')

# GPU reads indices via P2P
indices = gpu.read_buffer(buffer)
print(f"Indices: {indices}")
```

## API Design

### Device Management

```python
# FPGA Device
fpga = p2p.FPGADevice(bdf="81:00.1")
fpga = p2p.FPGADevice(device_index=0)  # Auto-detect

# GPU Device
gpu = p2p.GPUDevice(device_id=0)
gpu = p2p.GPUDevice()  # Use default GPU

# Device info
print(fpga.name)
print(fpga.bdf)
print(gpu.name)
print(gpu.pcie_id)
```

### P2P Buffers

```python
# Create buffers
buffer = fpga.create_buffer(count=1024, dtype=np.float32)
buffer = fpga.create_buffer(count=64, dtype=np.uint32)

# Properties
print(buffer.size)
print(buffer.dtype)

# Read/write from host
buffer.write(data)  # NumPy array
data = buffer.read()

# Sync
buffer.sync_to_device()
buffer.sync_to_host()
```

### FPGA Operations

```python
# Load bitstream
fpga.load_xclbin("kernel.xclbin")

# Generate indices
fpga.generate_indices(
    buffer=buffer,
    count=64,
    mode='sequential',  # or 'strided', 'pattern'
    stride=2,
    start_offset=0
)
```

### GPU Operations

```python
# Read from P2P buffer
data = gpu.read_buffer(buffer)

# Write to P2P buffer
gpu.write_buffer(buffer, data)

# Note: SpMV and custom kernel APIs are not yet implemented
# See examples/spmv_demo.py for a working SpMV example using HIP directly
```

### High-Level Workflows

Note: High-level pipeline APIs (SpMVPipeline, etc.) are planned for future releases. For now, use the low-level APIs shown above. See `examples/spmv_demo.py` for a complete working example.

## Architecture

```
Python Layer (heteromem_p2p)
    ↓
pybind11 Bindings (heteromem_p2p_bindings.cpp)
    ↓
C++ Wrapper Classes (P2PManager, FPGADevice, GPUDevice)
    ↓
XRT API (FPGA)  +  HIP/ROCm API (GPU)
```

## Features

- ✅ Automatic device detection
- ✅ P2P buffer management
- ✅ FPGA index generation kernel
- ✅ NumPy integration
- ✅ Error handling
- ✅ Buffer synchronization

## Examples

See `examples/` directory:

- `basic_transfer.py` - Simple P2P data transfer
- `p2p_demo.py` - FPGA-GPU P2P transfer demo
- `bm25_demo.py` - BM25 kernel P2P demo (works with indexer_bm25_fixed.xclbin)
