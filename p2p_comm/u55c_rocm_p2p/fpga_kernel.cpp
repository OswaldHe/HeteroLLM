/**
 * FPGA Kernel for P2P Demo
 * 
 * This kernel demonstrates simple HBM read/write operations.
 * The host will use XRT P2P buffers to expose this HBM region
 * to the GPU for direct PCIe access.
 * 
 * Build with Vitis HLS for Xilinx U55C
 */

#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

// Data width for AXI interface (512 bits = 64 bytes)
#define DATA_WIDTH 512
#define BYTES_PER_BEAT (DATA_WIDTH / 8)

typedef ap_uint<DATA_WIDTH> wide_t;

/**
 * Write Pattern Kernel
 * 
 * Writes a simple pattern to HBM for GPU to read.
 * Pattern: Each 64-byte block contains its block index repeated.
 * 
 * @param out       Output buffer in HBM (P2P accessible)
 * @param num_beats Number of 64-byte beats to write
 * @param pattern   Base pattern value to use
 */
extern "C" void fpga_write_pattern(
    wide_t* out,
    unsigned int num_beats,
    unsigned int pattern
) {
    #pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem0 \
        max_widen_bitwidth=512 num_write_outstanding=32 max_write_burst_length=64
    #pragma HLS INTERFACE s_axilite port=out bundle=control
    #pragma HLS INTERFACE s_axilite port=num_beats bundle=control
    #pragma HLS INTERFACE s_axilite port=pattern bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    write_loop:
    for (unsigned int i = 0; i < num_beats; i++) {
        #pragma HLS PIPELINE II=1
        
        wide_t data = 0;
        // Fill each 32-bit word with (pattern + beat_index)
        for (int w = 0; w < DATA_WIDTH / 32; w++) {
            #pragma HLS UNROLL
            unsigned int word_val = pattern + i;
            data.range(w * 32 + 31, w * 32) = word_val;
        }
        out[i] = data;
    }
}

/**
 * Read Verify Kernel
 * 
 * Reads data from HBM (written by GPU) and verifies pattern.
 * Returns number of mismatches.
 * 
 * @param in           Input buffer in HBM (P2P accessible)
 * @param num_beats    Number of 64-byte beats to read
 * @param expected     Expected pattern base value
 * @param error_count  Output: number of verification errors
 */
extern "C" void fpga_read_verify(
    const wide_t* in,
    unsigned int num_beats,
    unsigned int expected,
    unsigned int* error_count
) {
    #pragma HLS INTERFACE m_axi port=in offset=slave bundle=gmem0 \
        max_widen_bitwidth=512 num_read_outstanding=32 max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=error_count offset=slave bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=in bundle=control
    #pragma HLS INTERFACE s_axilite port=num_beats bundle=control
    #pragma HLS INTERFACE s_axilite port=expected bundle=control
    #pragma HLS INTERFACE s_axilite port=error_count bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    unsigned int errors = 0;

    read_loop:
    for (unsigned int i = 0; i < num_beats; i++) {
        #pragma HLS PIPELINE II=1
        
        wide_t data = in[i];
        
        // Check first 32-bit word of each beat
        unsigned int word_val = data.range(31, 0);
        unsigned int expected_val = expected + i;
        
        if (word_val != expected_val) {
            errors++;
        }
    }
    
    *error_count = errors;
}

/**
 * Memory Copy Kernel
 * 
 * Simple memory copy within FPGA HBM.
 * Useful for testing HBM bandwidth independently.
 * 
 * @param in        Input buffer
 * @param out       Output buffer
 * @param num_beats Number of 64-byte beats to copy
 */
extern "C" void fpga_memcpy(
    const wide_t* in,
    wide_t* out,
    unsigned int num_beats
) {
    #pragma HLS INTERFACE m_axi port=in offset=slave bundle=gmem0 \
        max_widen_bitwidth=512 num_read_outstanding=32 max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1 \
        max_widen_bitwidth=512 num_write_outstanding=32 max_write_burst_length=64
    #pragma HLS INTERFACE s_axilite port=in bundle=control
    #pragma HLS INTERFACE s_axilite port=out bundle=control
    #pragma HLS INTERFACE s_axilite port=num_beats bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    copy_loop:
    for (unsigned int i = 0; i < num_beats; i++) {
        #pragma HLS PIPELINE II=1
        out[i] = in[i];
    }
}
