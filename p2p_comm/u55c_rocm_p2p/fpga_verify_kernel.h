/**
 * FPGA Verification Kernel for GPU→FPGA P2P Transfer
 *
 * A minimal TAPA kernel that reads data written by the GPU into a P2P buffer,
 * computes a checksum (sum of all elements), and copies the data to an output
 * buffer. This proves that the FPGA can correctly observe GPU-written data
 * through the P2P path.
 *
 * Memory layout:
 *   input_mem:  N uint32 values written by GPU via P2P
 *   output_mem: N uint32 values (FPGA readback copy)
 *   result_mem: 2 uint32 values [checksum, count]
 */

#ifndef __FPGA_VERIFY_KERNEL_H__
#define __FPGA_VERIFY_KERNEL_H__

#include <tapa.h>
#include <cstdint>

// ============================================================================
// Read N uint32 values from P2P input buffer via async_mmap
// ============================================================================
void read_p2p_input(
    const int N,
    tapa::async_mmap<int>& input_mem,
    tapa::ostream<int>& data_fifo
) {
    for (int i_req = 0, i_resp = 0; i_resp < N;) {
        #pragma HLS pipeline II=1
        if ((i_req < N) & !input_mem.read_addr.full()) {
            input_mem.read_addr.try_write(i_req);
            ++i_req;
        }
        if (!input_mem.read_data.empty()) {
            int tmp;
            input_mem.read_data.try_read(tmp);
            data_fifo.write(tmp);
            ++i_resp;
        }
    }
}

// ============================================================================
// Compute checksum and forward data to output writer
// ============================================================================
void compute_checksum(
    const int N,
    tapa::istream<int>& data_in,
    tapa::ostream<int>& data_out,
    tapa::ostream<int>& checksum_fifo
) {
    unsigned int sum = 0;
    for (int i = 0; i < N; i++) {
        #pragma HLS pipeline II=1
        int val = data_in.read();
        sum += static_cast<unsigned int>(val);
        data_out.write(val);
    }
    // Send checksum and count
    checksum_fifo.write(static_cast<int>(sum));
    checksum_fifo.write(N);
}

// ============================================================================
// Write readback data to output buffer via async_mmap
// ============================================================================
void write_output(
    const int N,
    tapa::istream<int>& data_fifo,
    tapa::async_mmap<int>& output_mem
) {
    for (int i_req = 0, i_resp = 0; i_resp < N;) {
        #pragma HLS pipeline II=1
        if ((i_req < N) & !data_fifo.empty() &
            !output_mem.write_addr.full() & !output_mem.write_data.full()) {
            int tmp;
            data_fifo.try_read(tmp);
            output_mem.write_addr.try_write(i_req);
            output_mem.write_data.try_write(tmp);
            ++i_req;
        }
        bool success = false;
        auto resp = output_mem.write_resp.read(success);
        if (success) {
            i_resp += (unsigned)(resp) + 1;
        }
    }
}

// ============================================================================
// Write checksum result (2 ints: [sum, count]) to result buffer
// ============================================================================
void write_result(
    tapa::istream<int>& checksum_fifo,
    tapa::async_mmap<int>& result_mem
) {
    for (int i_req = 0, i_resp = 0; i_resp < 2;) {
        #pragma HLS pipeline II=1
        if ((i_req < 2) & !checksum_fifo.empty() &
            !result_mem.write_addr.full() & !result_mem.write_data.full()) {
            int tmp;
            checksum_fifo.try_read(tmp);
            result_mem.write_addr.try_write(i_req);
            result_mem.write_data.try_write(tmp);
            ++i_req;
        }
        bool success = false;
        auto resp = result_mem.write_resp.read(success);
        if (success) {
            i_resp += (unsigned)(resp) + 1;
        }
    }
}

// ============================================================================
// Top-level: fpga_verify_top
//
// Pipeline:  input_mem → read → checksum + forward → write output_mem
//                                   └→ write result_mem
// ============================================================================
void fpga_verify_top(
    const int N,
    tapa::mmap<int> input_mem,
    tapa::mmap<int> output_mem,
    tapa::mmap<int> result_mem
) {
    tapa::stream<int> data_fifo("data_fifo");
    tapa::stream<int> forward_fifo("forward_fifo");
    tapa::stream<int> checksum_fifo("checksum_fifo");

    tapa::task()
        .invoke<tapa::join>(read_p2p_input, N, input_mem, data_fifo)
        .invoke<tapa::join>(compute_checksum, N, data_fifo, forward_fifo, checksum_fifo)
        .invoke<tapa::join>(write_output, N, forward_fifo, output_mem)
        .invoke<tapa::join>(write_result, checksum_fifo, result_mem);
}

#endif // __FPGA_VERIFY_KERNEL_H__
