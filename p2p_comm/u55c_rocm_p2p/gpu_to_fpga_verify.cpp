/**
 * GPU→FPGA P2P End-to-End Verification
 *
 * End-to-end test: GPU writes data into a P2P buffer via the
 * write_indices_to_fpga HIP kernel, then the FPGA kernel reads and
 * verifies the data.
 *
 * Flow:
 *   1. Load xclbin, create XRT P2P buffers (input on specific mem_group)
 *   2. Map input buffer, register with HIP
 *   3. GPU writes known pattern to the P2P buffer by launching
 *      write_indices_to_fpga on the mapped device pointer
 *   4. FPGA kernel reads input, computes checksum, copies to output
 *   5. Host reads output and verifies against expected values
 *
 * Build:
 *   hipcc gpu_to_fpga_verify.cpp gpu_write_kernels.hip \
 *         -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil \
 *         -o gpu_to_fpga_verify
 *
 * Run:
 *   ./gpu_to_fpga_verify --xclbin fpga_verify.xclbin --fpga 81:00.1
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cassert>

// XRT
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

// HIP
#include <hip/hip_runtime.h>

// GPU kernel declarations
extern "C" __global__ void write_indices_to_fpga(
    const uint32_t* __restrict__ gpu_src,
    uint32_t* __restrict__ fpga_dst,
    uint32_t count
);

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " --xclbin <path> --fpga <BDF>" << std::endl;
    std::cerr << "  --xclbin   Path to fpga_verify.xclbin" << std::endl;
    std::cerr << "  --fpga     FPGA BDF (e.g., 81:00.1)" << std::endl;
    std::cerr << "  --count    Number of uint32 elements (default: 1024)" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string xclbin_path;
    std::string fpga_bdf;
    int count = 1024;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--xclbin" && i + 1 < argc) xclbin_path = argv[++i];
        else if (arg == "--fpga" && i + 1 < argc) fpga_bdf = argv[++i];
        else if (arg == "--count" && i + 1 < argc) count = std::atoi(argv[++i]);
        else if (arg == "--help") { print_usage(argv[0]); return 0; }
    }

    if (xclbin_path.empty() || fpga_bdf.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    const uint32_t N = static_cast<uint32_t>(count);
    const size_t data_bytes = N * sizeof(uint32_t);

    std::cout << "=== GPU→FPGA P2P End-to-End Verification ===" << std::endl;
    std::cout << "FPGA BDF:  " << fpga_bdf << std::endl;
    std::cout << "xclbin:    " << xclbin_path << std::endl;
    std::cout << "Count:     " << N << " uint32 (" << data_bytes << " bytes)" << std::endl;

    // ---- 1. Open FPGA and load xclbin ----
    std::string full_bdf = "0000:" + fpga_bdf;
    xrt::device fpga_dev(full_bdf);
    auto uuid = fpga_dev.load_xclbin(xclbin_path);
    xrt::kernel krnl(fpga_dev, uuid, "fpga_verify_top");
    std::cout << "\nLoaded xclbin and kernel 'fpga_verify_top'" << std::endl;

    // ---- 2. Allocate XRT buffers ----
    // Input: P2P buffer (GPU will write here)
    // Output + Result: normal buffers (FPGA writes, host reads)
    xrt::bo input_bo;
    bool input_is_p2p = false;
    constexpr unsigned kMaxGroups = 32;

    // Try P2P allocation for input
    bool input_allocated = false;
    for (unsigned mg = 0; mg < kMaxGroups; mg++) {
        try {
            input_bo = xrt::bo(fpga_dev, data_bytes, xrt::bo::flags::p2p, mg);
            input_is_p2p = true;
            input_allocated = true;
            std::cout << "  Input buffer: P2P, mem_group=" << mg << std::endl;
            break;
        } catch (...) {}
    }
    if (!input_is_p2p) {
        // Fallback to normal
        for (unsigned mg = 0; mg < kMaxGroups; mg++) {
            try {
                input_bo = xrt::bo(fpga_dev, data_bytes, xrt::bo::flags::normal, mg);
                input_allocated = true;
                std::cout << "  Input buffer: normal (non-P2P fallback), mem_group=" << mg << std::endl;
                break;
            } catch (...) {}
        }
    }
    if (!input_allocated) {
        std::cerr << "Fatal: failed to allocate input buffer in any memory group" << std::endl;
        return 1;
    }

    xrt::bo output_bo;
    bool output_allocated = false;
    for (unsigned mg = 0; mg < kMaxGroups; mg++) {
        try {
            output_bo = xrt::bo(fpga_dev, data_bytes, xrt::bo::flags::normal, mg);
            output_allocated = true;
            std::cout << "  Output buffer: mem_group=" << mg << std::endl;
            break;
        } catch (...) {}
    }
    if (!output_allocated) {
        std::cerr << "Fatal: failed to allocate output buffer in any memory group" << std::endl;
        return 1;
    }

    xrt::bo result_bo;
    bool result_allocated = false;
    for (unsigned mg = 0; mg < kMaxGroups; mg++) {
        try {
            result_bo = xrt::bo(fpga_dev, 2 * sizeof(uint32_t), xrt::bo::flags::normal, mg);
            result_allocated = true;
            std::cout << "  Result buffer: mem_group=" << mg << std::endl;
            break;
        } catch (...) {}
    }
    if (!result_allocated) {
        std::cerr << "Fatal: failed to allocate result buffer in any memory group" << std::endl;
        return 1;
    }

    // ---- 3. Map input buffer and register with HIP ----
    auto* input_host_ptr = input_bo.map<uint32_t*>();
    assert(input_host_ptr != nullptr);

    hipError_t err = hipHostRegister(
        input_host_ptr, data_bytes,
        hipHostRegisterMapped | hipHostRegisterIoMemory);
    if (err != hipSuccess) {
        std::cerr << "  Warning: IoMemory failed, trying Mapped only" << std::endl;
        input_is_p2p = false;
        err = hipHostRegister(input_host_ptr, data_bytes, hipHostRegisterMapped);
        if (err != hipSuccess) {
            std::cerr << "Fatal: hipHostRegister failed" << std::endl;
            return 1;
        }
    }

    void* dev_ptr = nullptr;
    HIP_CHECK(hipHostGetDevicePointer(&dev_ptr, input_host_ptr, 0));
    std::cout << "  Registered with HIP, device_ptr=" << dev_ptr
              << ", is_p2p=" << input_is_p2p << std::endl;

    // ---- 4. GPU writes data ----
    // Prepare data on GPU
    std::vector<uint32_t> host_data(N);
    for (uint32_t i = 0; i < N; i++) {
        host_data[i] = i * 7 + 42;
    }

    uint32_t* d_src;
    HIP_CHECK(hipMalloc(&d_src, data_bytes));
    HIP_CHECK(hipMemcpy(d_src, host_data.data(), data_bytes, hipMemcpyHostToDevice));

    // GPU kernel writes to FPGA P2P buffer
    dim3 threads(256);
    dim3 blocks((N + 255) / 256);
    hipLaunchKernelGGL(write_indices_to_fpga, blocks, threads, 0, 0,
                       d_src, static_cast<uint32_t*>(dev_ptr), N);
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "\n  GPU write completed" << std::endl;

    // For non-P2P, sync host shadow to FPGA
    if (!input_is_p2p) {
        std::cout << "  (non-P2P) syncing to device..." << std::endl;
        input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // ---- 5. Run FPGA kernel ----
    std::cout << "  Running FPGA kernel..." << std::endl;
    auto run = krnl(static_cast<int>(N), input_bo, output_bo, result_bo);
    run.wait();
    std::cout << "  FPGA kernel completed" << std::endl;

    // ---- 6. Read back results ----
    output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    result_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto* output_ptr = output_bo.map<uint32_t*>();
    auto* result_ptr = result_bo.map<uint32_t*>();

    unsigned int fpga_sum = result_ptr[0];
    int fpga_count = static_cast<int>(result_ptr[1]);

    // Compute expected checksum
    unsigned int expected_sum = 0;
    for (uint32_t i = 0; i < N; i++) {
        expected_sum += host_data[i];
    }

    // ---- 7. Verify ----
    std::cout << "\nVerification:" << std::endl;
    std::cout << "  Checksum: FPGA=" << fpga_sum << " expected=" << expected_sum << std::endl;
    std::cout << "  Count:    FPGA=" << fpga_count << " expected=" << N << std::endl;

    bool pass = true;
    if (fpga_sum != expected_sum) {
        std::cerr << "  FAIL: checksum mismatch!" << std::endl;
        pass = false;
    }

    int mismatches = 0;
    for (uint32_t i = 0; i < N; i++) {
        if (output_ptr[i] != host_data[i]) {
            if (mismatches < 10) {
                std::cerr << "  MISMATCH [" << i << "]: FPGA=" << output_ptr[i]
                          << " expected=" << host_data[i] << std::endl;
            }
            mismatches++;
        }
    }
    if (mismatches > 0) {
        std::cerr << "  FAIL: " << mismatches << "/" << N << " elements differ" << std::endl;
        pass = false;
    } else {
        std::cout << "  All " << N << " elements match" << std::endl;
    }

    // ---- Cleanup ----
    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipHostUnregister(input_host_ptr));

    std::cout << "\n=== " << (pass ? "PASS" : "FAIL") << " ===" << std::endl;
    return pass ? 0 : 1;
}
