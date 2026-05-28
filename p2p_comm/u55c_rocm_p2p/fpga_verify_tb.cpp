/**
 * FPGA Verification Testbench for GPU→FPGA P2P Transfer
 *
 * Two modes:
 *   1. TAPA C-simulation (no --bitstream): CPU fills the input buffer,
 *      invokes the TAPA kernel in software, and verifies the output.
 *   2. Hardware (--bitstream=<xclbin>): Uses XRT P2P buffers + HIP to
 *      have the GPU write data, then the FPGA kernel reads and verifies.
 *
 * Build (csim):   tapa g++ -- fpga_verify_tb.cpp -o fpga_verify_tb
 * Build (hw):     hipcc fpga_verify_tb.cpp gpu_write_kernels.hip \
 *                       -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
 *                       -lxrt_coreutil -o fpga_verify_hw_tb
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <cassert>

#include <gflags/gflags.h>

#include "fpga_verify_kernel.h"

DEFINE_string(bitstream, "", "Path to xclbin bitstream (empty = TAPA csim)");
DEFINE_int32(count, 1024, "Number of uint32 elements to transfer");

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const int N = FLAGS_count;
    std::cout << "=== FPGA Verify GPU→FPGA P2P ===" << std::endl;
    std::cout << "Mode: " << (FLAGS_bitstream.empty() ? "TAPA C-simulation" : "Hardware")
              << std::endl;
    std::cout << "Count: " << N << " uint32 elements (" << N * 4 << " bytes)" << std::endl;

    // --- Prepare input data ---
    aligned_vector<int> input(N);
    for (int i = 0; i < N; i++) {
        input[i] = static_cast<int>(i * 7 + 42);
    }

    // Compute expected checksum on host
    unsigned int expected_sum = 0;
    for (int i = 0; i < N; i++) {
        expected_sum += static_cast<unsigned int>(input[i]);
    }

    // --- Prepare output buffers ---
    aligned_vector<int> output(N, -1);
    aligned_vector<int> result(2, 0);  // [checksum, count]

    // --- Invoke FPGA kernel ---
    std::cout << "\nInvoking fpga_verify_top..." << std::endl;

    int64_t elapsed_ns = tapa::invoke(
        fpga_verify_top,
        FLAGS_bitstream,
        N,
        tapa::read_only_mmap<int>(input),
        tapa::write_only_mmap<int>(output),
        tapa::write_only_mmap<int>(result)
    );

    std::cout << "  Kernel completed in " << elapsed_ns << " ns" << std::endl;

    // --- Verify ---
    std::cout << "\nVerification:" << std::endl;

    // Check checksum
    unsigned int fpga_sum = static_cast<unsigned int>(result[0]);
    int fpga_count = result[1];
    std::cout << "  FPGA checksum: " << fpga_sum << "  expected: " << expected_sum << std::endl;
    std::cout << "  FPGA count:    " << fpga_count << "  expected: " << N << std::endl;

    bool pass = true;
    if (fpga_sum != expected_sum) {
        std::cerr << "  FAIL: checksum mismatch!" << std::endl;
        pass = false;
    }
    if (fpga_count != N) {
        std::cerr << "  FAIL: count mismatch!" << std::endl;
        pass = false;
    }

    // Check element-by-element readback
    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        if (output[i] != input[i]) {
            if (mismatches < 10) {
                std::cerr << "  MISMATCH at [" << i << "]: got " << output[i]
                          << " expected " << input[i] << std::endl;
            }
            mismatches++;
        }
    }
    if (mismatches > 0) {
        std::cerr << "  FAIL: " << mismatches << " / " << N << " elements differ" << std::endl;
        pass = false;
    } else {
        std::cout << "  All " << N << " elements match" << std::endl;
    }

    std::cout << "\n=== " << (pass ? "PASS" : "FAIL") << " ===" << std::endl;
    return pass ? 0 : 1;
}
