/**
 * GPU Kernel for P2P Demo
 * 
 * This file contains HIP kernels for reading from and writing to 
 * memory buffers. When the host maps the FPGA's P2P buffer and 
 * registers it with ROCm, these kernels can directly access FPGA HBM.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

/**
 * Write Pattern Kernel
 * 
 * Writes a pattern to memory. When 'dst' points to a registered
 * FPGA P2P buffer, this writes directly to FPGA HBM.
 * 
 * @param dst       Destination buffer (can be FPGA P2P memory)
 * @param num_words Number of 32-bit words to write
 * @param pattern   Base pattern value
 */
extern "C" __global__ void gpu_write_pattern(
    uint32_t* dst,
    uint32_t num_words,
    uint32_t pattern
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    for (uint32_t i = idx; i < num_words; i += stride) {
        // Pattern: base + word_index / 16 (so each 64-byte block has same value)
        // This matches the FPGA's expectation where each 512-bit beat has same pattern
        dst[i] = pattern + (i / 16);
    }
}

/**
 * Read Verify Kernel
 * 
 * Reads from memory and counts verification errors.
 * When 'src' points to a registered FPGA P2P buffer, 
 * this reads directly from FPGA HBM.
 * 
 * @param src           Source buffer (can be FPGA P2P memory)
 * @param num_words     Number of 32-bit words to read
 * @param expected      Expected base pattern value
 * @param error_count   Output: number of errors (atomic increment)
 */
extern "C" __global__ void gpu_read_verify(
    const uint32_t* src,
    uint32_t num_words,
    uint32_t expected,
    uint32_t* error_count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    uint32_t local_errors = 0;
    
    for (uint32_t i = idx; i < num_words; i += stride) {
        uint32_t val = src[i];
        // Expected: base + word_index / 16 (matching FPGA's pattern)
        uint32_t expected_val = expected + (i / 16);
        if (val != expected_val) {
            local_errors++;
        }
    }
    
    // Aggregate errors atomically
    if (local_errors > 0) {
        atomicAdd(error_count, local_errors);
    }
}

/**
 * Memory Copy Kernel
 * 
 * Copies data from src to dst.
 * Can be used for:
 * - GPU HBM to FPGA HBM (when dst is registered P2P buffer)
 * - FPGA HBM to GPU HBM (when src is registered P2P buffer)
 * 
 * @param src       Source buffer
 * @param dst       Destination buffer
 * @param num_words Number of 32-bit words to copy
 */
extern "C" __global__ void gpu_memcpy(
    const uint32_t* src,
    uint32_t* dst,
    uint32_t num_words
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    for (uint32_t i = idx; i < num_words; i += stride) {
        dst[i] = src[i];
    }
}

/**
 * Vectorized Memory Copy Kernel
 * 
 * Uses 128-bit (float4) loads/stores for better bandwidth.
 * 
 * @param src       Source buffer (must be 16-byte aligned)
 * @param dst       Destination buffer (must be 16-byte aligned)
 * @param num_vec   Number of float4 vectors to copy
 */
extern "C" __global__ void gpu_memcpy_vec4(
    const float4* src,
    float4* dst,
    uint32_t num_vec
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    for (uint32_t i = idx; i < num_vec; i += stride) {
        dst[i] = src[i];
    }
}

/**
 * Bandwidth Test Kernel
 * 
 * Performs streaming reads to measure FPGA->GPU P2P bandwidth.
 * Uses vectorized loads and doesn't store results (except checksum).
 * 
 * @param src       Source buffer (FPGA P2P memory)
 * @param num_vec   Number of float4 vectors to read
 * @param checksum  Output: simple checksum to prevent optimization
 */
extern "C" __global__ void gpu_read_bandwidth_test(
    const float4* src,
    uint32_t num_vec,
    float* checksum
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    float local_sum = 0.0f;
    
    for (uint32_t i = idx; i < num_vec; i += stride) {
        float4 val = src[i];
        local_sum += val.x + val.y + val.z + val.w;
    }
    
    // Simple reduction to first thread
    __shared__ float sdata[256];
    uint32_t tid = threadIdx.x;
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Reduce within block
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // First thread of each block contributes to global checksum
    if (tid == 0) {
        atomicAdd(checksum, sdata[0]);
    }
}

/**
 * Bandwidth Test Kernel - Write
 * 
 * Performs streaming writes to measure GPU->FPGA P2P bandwidth.
 * 
 * @param dst       Destination buffer (FPGA P2P memory)
 * @param num_vec   Number of float4 vectors to write
 * @param pattern   Pattern value to write
 */
extern "C" __global__ void gpu_write_bandwidth_test(
    float4* dst,
    uint32_t num_vec,
    float pattern
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    float4 val = make_float4(pattern, pattern + 1.0f, pattern + 2.0f, pattern + 3.0f);
    
    for (uint32_t i = idx; i < num_vec; i += stride) {
        dst[i] = val;
    }
}
