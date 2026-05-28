/**
 * GPU Kernels for GPU→FPGA P2P Transfer
 *
 * These kernels write data from GPU memory into FPGA-mapped P2P buffers.
 * The FPGA buffer must be registered with hipHostRegister() and the
 * device pointer obtained via hipHostGetDevicePointer() before use.
 *
 * Build: hipcc -c gpu_write_kernels.hip -o gpu_write_kernels.o  // add --offload-arch as needed
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

/**
 * Write uint32 data from GPU memory to FPGA buffer via P2P.
 * Mirror of read_indices_from_fpga — same structure, reversed direction.
 */
extern "C" __global__ void write_indices_to_fpga(
    const uint32_t* __restrict__ gpu_indices,
    uint32_t* __restrict__ fpga_buffer,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        fpga_buffer[idx] = gpu_indices[idx];
    }
}

/**
 * Write float data from GPU memory to FPGA buffer via P2P.
 */
extern "C" __global__ void write_floats_to_fpga(
    const float* __restrict__ gpu_data,
    float* __restrict__ fpga_buffer,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        fpga_buffer[idx] = gpu_data[idx];
    }
}

/**
 * Compute SpMV and write results directly to FPGA buffer (fused compute+write).
 */
extern "C" __global__ void spmv_write_to_fpga(
    const uint32_t* __restrict__ row_indices,
    uint32_t num_rows,
    uint32_t matrix_size,
    const float* __restrict__ values,
    const uint32_t* __restrict__ col_idx,
    const uint32_t* __restrict__ row_ptr,
    const float* __restrict__ x,
    float* __restrict__ fpga_output
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        uint32_t row = row_indices[idx];
        if (row >= matrix_size) {
            row = row % matrix_size;
        }

        uint32_t row_start = row_ptr[row];
        uint32_t row_end   = row_ptr[row + 1];

        float sum = 0.0f;
        for (uint32_t i = row_start; i < row_end; i++) {
            sum += values[i] * x[col_idx[i]];
        }

        fpga_output[idx] = sum;
    }
}

/**
 * Generate a sequential pattern directly into FPGA buffer.
 * Useful for verifying P2P write correctness.
 */
extern "C" __global__ void generate_pattern_to_fpga(
    uint32_t* __restrict__ fpga_buffer,
    uint32_t count,
    uint32_t start,
    uint32_t stride
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        fpga_buffer[idx] = start + idx * stride;
    }
}

/**
 * Scatter-write: GPU writes to non-contiguous positions in FPGA buffer.
 */
extern "C" __global__ void scatter_write_to_fpga(
    const uint32_t* __restrict__ gpu_data,
    const uint32_t* __restrict__ scatter_indices,
    uint32_t* __restrict__ fpga_buffer,
    uint32_t count,
    uint32_t buffer_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint32_t dest = scatter_indices[idx];
        if (dest < buffer_size) {
            fpga_buffer[dest] = gpu_data[idx];
        }
    }
}
