/**
 * GPU Kernels for SpMV P2P Demo
 * 
 * Demonstrates reading indices from FPGA memory via P2P
 * and performing sparse matrix-vector multiplication.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

/**
 * Read Indices from FPGA Memory
 * 
 * Reads row indices from FPGA memory (via P2P) into GPU memory.
 * This demonstrates FPGA→GPU P2P transfer.
 * 
 * @param fpga_indices  Pointer to FPGA memory (P2P registered)
 * @param gpu_indices   Pointer to GPU memory (output)
 * @param count         Number of indices to read
 */
extern "C" __global__ void read_indices_from_fpga(
    const uint32_t* fpga_indices,
    uint32_t* gpu_indices,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        // Read from FPGA memory via P2P
        gpu_indices[idx] = fpga_indices[idx];
    }
}

/**
 * SpMV Kernel - CSR Format
 * 
 * Performs sparse matrix-vector multiplication for selected rows.
 * y[row_indices[i]] = A[row_indices[i], :] * x
 * 
 * @param row_indices   Array of row indices to process
 * @param num_rows      Number of rows to process
 * @param matrix_size   Total number of rows in the matrix (for bounds checking)
 * @param values        CSR values array
 * @param col_idx       CSR column indices array
 * @param row_ptr       CSR row pointers array
 * @param x             Input dense vector
 * @param y             Output dense vector
 */
extern "C" __global__ void spmv_csr_kernel(
    const uint32_t* row_indices,
    uint32_t num_rows,
    uint32_t matrix_size,
    const float* values,
    const uint32_t* col_idx,
    const uint32_t* row_ptr,
    const float* x,
    float* y
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_rows) {
        uint32_t row = row_indices[idx];
        
        // Clamp row to valid range (FPGA may generate indices > matrix size)
        if (row >= matrix_size) {
            row = row % matrix_size;
        }
        
        // Get start and end of this row in CSR format
        uint32_t row_start = row_ptr[row];
        uint32_t row_end = row_ptr[row + 1];
        
        // Compute dot product for this row
        float sum = 0.0f;
        for (uint32_t i = row_start; i < row_end; i++) {
            uint32_t col = col_idx[i];
            sum += values[i] * x[col];
        }
        
        y[row] = sum;
    }
}

/**
 * Verify Results Kernel
 * 
 * Checks if computed results match expected values.
 * 
 * @param y             Computed output vector
 * @param y_expected    Expected output vector
 * @param indices       Row indices that were computed
 * @param count         Number of rows to verify
 * @param error_count   Output: number of errors found
 */
extern "C" __global__ void verify_results_kernel(
    const float* y,
    const float* y_expected,
    const uint32_t* indices,
    uint32_t count,
    uint32_t* error_count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        uint32_t row = indices[idx];
        float computed = y[row];
        float expected = y_expected[row];
        
        // Allow small floating-point error
        float diff = fabsf(computed - expected);
        if (diff > 1e-5f) {
            atomicAdd(error_count, 1);
        }
    }
}

/**
 * Initialize Test Vector
 * 
 * Fills vector with test pattern: x[i] = i + 1
 */
extern "C" __global__ void init_vector_kernel(
    float* x,
    uint32_t size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        x[idx] = (float)(idx + 1);
    }
}
