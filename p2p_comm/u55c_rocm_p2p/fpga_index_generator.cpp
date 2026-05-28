/**
 * FPGA Index Generator Kernel
 * 
 * Generates sparse matrix row indices for GPU SpMV computation.
 * This kernel demonstrates FPGA→GPU P2P data transfer.
 * 
 * The kernel can operate in different modes:
 * - Sequential: indices = [0, 1, 2, ..., count-1]
 * - Strided: indices = [0, stride, 2*stride, ...]
 * - Pattern-based: indices from a predefined pattern
 */

#include <stdint.h>
#include <hls_stream.h>
#include <ap_int.h>

// AXI interface for HBM access
#include "ap_axi_sdata.h"

/**
 * Index Generator Kernel
 * 
 * Generates row indices and writes them to HBM memory.
 * The output buffer is P2P accessible by the GPU.
 * 
 * @param indices       Output buffer for indices (HBM, P2P accessible)
 * @param count         Number of indices to generate
 * @param mode          Generation mode (0=sequential, 1=strided, 2=pattern)
 * @param stride        Stride value (for mode=1)
 * @param start_offset  Starting offset for index generation
 */
extern "C" {
void fpga_index_generator(
    uint32_t* indices,      // Output: generated indices
    uint32_t count,         // Number of indices to generate
    uint32_t mode,          // Generation mode
    uint32_t stride,        // Stride for strided mode
    uint32_t start_offset   // Starting offset
) {
    #pragma HLS INTERFACE m_axi port=indices offset=slave bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=indices bundle=control
    #pragma HLS INTERFACE s_axilite port=count bundle=control
    #pragma HLS INTERFACE s_axilite port=mode bundle=control
    #pragma HLS INTERFACE s_axilite port=stride bundle=control
    #pragma HLS INTERFACE s_axilite port=start_offset bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    // Generate indices based on mode
    switch (mode) {
        case 0: // Sequential mode
            gen_sequential:
            for (uint32_t i = 0; i < count; i++) {
                #pragma HLS PIPELINE II=1
                indices[i] = start_offset + i;
            }
            break;
            
        case 1: // Strided mode
            gen_strided:
            for (uint32_t i = 0; i < count; i++) {
                #pragma HLS PIPELINE II=1
                indices[i] = start_offset + (i * stride);
            }
            break;
            
        case 2: // Pattern mode (example: powers of 2)
            gen_pattern:
            for (uint32_t i = 0; i < count; i++) {
                #pragma HLS PIPELINE II=1
                // Example pattern: 0, 1, 2, 4, 8, 16, ...
                if (i < 2) {
                    indices[i] = start_offset + i;
                } else {
                    // Clamp shift to prevent overflow (max shift is 31 for 32-bit)
                    uint32_t shift = (i - 1 < 31) ? (i - 1) : 31;
                    indices[i] = start_offset + (1u << shift);
                }
            }
            break;
            
        default: // Default to sequential
            gen_default:
            for (uint32_t i = 0; i < count; i++) {
                #pragma HLS PIPELINE II=1
                indices[i] = start_offset + i;
            }
            break;
    }
}
}

/**
 * Optimized Burst Index Generator
 * 
 * Uses burst writes for better HBM bandwidth utilization.
 * Generates indices in local buffer first, then writes in bursts.
 */
extern "C" {
void fpga_index_generator_burst(
    uint32_t* indices,
    uint32_t count,
    uint32_t mode,
    uint32_t stride,
    uint32_t start_offset
) {
    #pragma HLS INTERFACE m_axi port=indices offset=slave bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=indices bundle=control
    #pragma HLS INTERFACE s_axilite port=count bundle=control
    #pragma HLS INTERFACE s_axilite port=mode bundle=control
    #pragma HLS INTERFACE s_axilite port=stride bundle=control
    #pragma HLS INTERFACE s_axilite port=start_offset bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    const uint32_t BURST_SIZE = 64;
    uint32_t local_buffer[BURST_SIZE];
    #pragma HLS ARRAY_PARTITION variable=local_buffer cyclic factor=16
    
    uint32_t num_bursts = (count + BURST_SIZE - 1) / BURST_SIZE;
    
    burst_loop:
    for (uint32_t burst = 0; burst < num_bursts; burst++) {
        uint32_t burst_start = burst * BURST_SIZE;
        uint32_t burst_count = (burst_start + BURST_SIZE > count) ? 
                               (count - burst_start) : BURST_SIZE;
        
        // Generate indices in local buffer
        generate_burst:
        for (uint32_t i = 0; i < burst_count; i++) {
            #pragma HLS PIPELINE II=1
            uint32_t global_idx = burst_start + i;
            
            switch (mode) {
                case 0: // Sequential
                    local_buffer[i] = start_offset + global_idx;
                    break;
                case 1: // Strided
                    local_buffer[i] = start_offset + (global_idx * stride);
                    break;
                default:
                    local_buffer[i] = start_offset + global_idx;
                    break;
            }
        }
        
        // Write burst to HBM
        write_burst:
        for (uint32_t i = 0; i < burst_count; i++) {
            #pragma HLS PIPELINE II=1
            indices[burst_start + i] = local_buffer[i];
        }
    }
}
}

/**
 * Sparse Pattern Generator
 * 
 * Generates indices for sparse matrix access patterns.
 * Useful for demonstrating real-world SpMV workloads.
 */
extern "C" {
void fpga_sparse_index_generator(
    uint32_t* indices,          // Output indices
    const uint32_t* pattern,    // Input pattern (lookup table)
    uint32_t count,             // Number of indices
    uint32_t pattern_size,      // Size of pattern
    uint32_t repeat             // Repeat pattern
) {
    #pragma HLS INTERFACE m_axi port=indices offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=pattern offset=slave bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=indices bundle=control
    #pragma HLS INTERFACE s_axilite port=pattern bundle=control
    #pragma HLS INTERFACE s_axilite port=count bundle=control
    #pragma HLS INTERFACE s_axilite port=pattern_size bundle=control
    #pragma HLS INTERFACE s_axilite port=repeat bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    // Validate pattern_size to prevent divide-by-zero and out-of-bounds access
    if (pattern_size == 0 || pattern_size > 256) {
        return;  // Invalid pattern size, do nothing
    }
    
    // Cache pattern in local memory for faster access
    uint32_t pattern_cache[256];
    #pragma HLS ARRAY_PARTITION variable=pattern_cache cyclic factor=16
    
    // Load pattern
    load_pattern:
    for (uint32_t i = 0; i < pattern_size && i < 256; i++) {
        #pragma HLS PIPELINE II=1
        pattern_cache[i] = pattern[i];
    }
    
    // Generate indices using pattern
    generate_indices:
    for (uint32_t i = 0; i < count; i++) {
        #pragma HLS PIPELINE II=1
        uint32_t pattern_idx = i % pattern_size;
        uint32_t repeat_offset = (i / pattern_size) * repeat;
        indices[i] = pattern_cache[pattern_idx] + repeat_offset;
    }
}
}
