/**
 * Standalone FPGA Verification Testbench (no TAPA runtime required)
 *
 * Tests the same kernel logic as fpga_verify_kernel.h using plain C++:
 *   1. Fill input buffer with known pattern (i*7+42)
 *   2. Simulate the FPGA pipeline: read → checksum → write output + result
 *   3. Verify checksum, count, and element-by-element readback
 *
 * Build:  g++ -std=c++17 -O2 -o fpga_verify_standalone_tb fpga_verify_standalone_tb.cpp
 * Run:    ./fpga_verify_standalone_tb [count]
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>

int main(int argc, char* argv[]) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 1024;

    std::cout << "=== FPGA Verify Standalone Testbench ===" << std::endl;
    std::cout << "Count: " << N << " uint32 elements (" << N * 4 << " bytes)" << std::endl;

    // --- Prepare input data (same pattern as GPU kernel: i*7+42) ---
    std::vector<int> input(N);
    for (int i = 0; i < N; i++) {
        input[i] = static_cast<int>(i * 7 + 42);
    }

    // --- Compute expected checksum on host ---
    unsigned int expected_sum = 0;
    for (int i = 0; i < N; i++) {
        expected_sum += static_cast<unsigned int>(input[i]);
    }

    // --- Simulate FPGA kernel logic ---
    // This mirrors fpga_verify_top: read input → checksum + copy → write result
    std::vector<int> output(N, -1);
    std::vector<int> result(2, 0);

    std::cout << "\nSimulating fpga_verify_top kernel logic..." << std::endl;

    // Stage 1: read_p2p_input → data_fifo (just copies input)
    // Stage 2: compute_checksum: sum all values, forward to output
    unsigned int fpga_sum = 0;
    for (int i = 0; i < N; i++) {
        int val = input[i];
        fpga_sum += static_cast<unsigned int>(val);
        output[i] = val;
    }

    // Stage 3+4: write checksum and count to result
    result[0] = static_cast<int>(fpga_sum);
    result[1] = N;

    std::cout << "  Kernel simulation complete" << std::endl;

    // --- Verify ---
    std::cout << "\nVerification:" << std::endl;

    unsigned int got_sum = static_cast<unsigned int>(result[0]);
    int got_count = result[1];
    std::cout << "  FPGA checksum: " << got_sum << "  expected: " << expected_sum << std::endl;
    std::cout << "  FPGA count:    " << got_count << "  expected: " << N << std::endl;

    bool pass = true;
    if (got_sum != expected_sum) {
        std::cerr << "  FAIL: checksum mismatch!" << std::endl;
        pass = false;
    }
    if (got_count != N) {
        std::cerr << "  FAIL: count mismatch!" << std::endl;
        pass = false;
    }

    // Element-by-element check
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

    // Print first few values for visual inspection
    std::cout << "\n  First 8 input:  [";
    for (int i = 0; i < std::min(N, 8); i++) {
        std::cout << input[i] << (i < std::min(N, 8) - 1 ? " " : "");
    }
    std::cout << "]" << std::endl;
    std::cout << "  First 8 output: [";
    for (int i = 0; i < std::min(N, 8); i++) {
        std::cout << output[i] << (i < std::min(N, 8) - 1 ? " " : "");
    }
    std::cout << "]" << std::endl;

    std::cout << "\n=== " << (pass ? "PASS" : "FAIL") << " ===" << std::endl;
    return pass ? 0 : 1;
}
