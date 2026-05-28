/**
 * GPU → FPGA P2P Transfer Demo
 *
 * Demonstrates the reverse P2P direction: GPU writes data into an
 * FPGA-resident buffer that is accessible to both devices.
 *
 * Flow:
 *   1. FPGA allocates a P2P buffer in HBM
 *   2. Buffer is mapped to host via xrt::bo::map()
 *   3. Mapped pointer is registered with GPU via hipHostRegister()
 *   4. GPU kernel writes data through the P2P device pointer
 *   5. FPGA reads the data (host-mapped pointer or FPGA kernel)
 *
 * Build:
 *   hipcc -c gpu_write_kernels.hip -o gpu_write_kernels.o
 *   hipcc gpu_to_fpga_demo.cpp gpu_write_kernels.o \
 *         -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil \
 *         -o gpu_to_fpga_demo
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cassert>
#include <numeric>
#include <random>
#include <set>

// XRT includes
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

// HIP includes
#include <hip/hip_runtime.h>

#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = (cmd); \
        if (error != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ============================================================================
// GPU kernel declarations (defined in gpu_write_kernels.hip)
// ============================================================================

extern "C" __global__ void write_indices_to_fpga(
    const uint32_t* __restrict__ gpu_indices,
    uint32_t* __restrict__ fpga_buffer,
    uint32_t count);

extern "C" __global__ void write_floats_to_fpga(
    const float* __restrict__ gpu_data,
    float* __restrict__ fpga_buffer,
    uint32_t count);

extern "C" __global__ void generate_pattern_to_fpga(
    uint32_t* __restrict__ fpga_buffer,
    uint32_t count,
    uint32_t start,
    uint32_t stride);

extern "C" __global__ void scatter_write_to_fpga(
    const uint32_t* __restrict__ gpu_data,
    const uint32_t* __restrict__ scatter_indices,
    uint32_t* __restrict__ fpga_buffer,
    uint32_t count,
    uint32_t buffer_size);

// ============================================================================
// Helper: allocate a P2P buffer on the FPGA
// ============================================================================

struct P2PAllocation {
    xrt::bo    bo;
    void*      host_ptr    = nullptr;
    void*      device_ptr  = nullptr;   // HIP device pointer
    size_t     size_bytes  = 0;
    bool       is_p2p      = false;     // true = P2P BO, false = normal fallback
    bool       gpu_registered = false;
};

/**
 * Allocate an xrt::bo with the P2P flag, map it, and register it
 * with HIP so that GPU kernels can access it via device_ptr.
 */
P2PAllocation alloc_p2p_buffer(xrt::device& fpga_dev, size_t size_bytes) {
    P2PAllocation alloc;
    alloc.size_bytes = size_bytes;

    // --- 1. Allocate FPGA buffer (try P2P, then fall back to normal) ---
    bool allocated = false;
    constexpr unsigned kMaxGroups = 32;

    for (unsigned g = 0; g < kMaxGroups && !allocated; ++g) {
        try {
            alloc.bo    = xrt::bo(fpga_dev, size_bytes, xrt::bo::flags::p2p, g);
            alloc.is_p2p = true;
            allocated   = true;
        } catch (...) {}
    }
    if (!allocated) {
        for (unsigned g = 0; g < kMaxGroups && !allocated; ++g) {
            try {
                alloc.bo    = xrt::bo(fpga_dev, size_bytes, xrt::bo::flags::normal, g);
                alloc.is_p2p = false;
                allocated   = true;
            } catch (...) {}
        }
    }
    if (!allocated) {
        std::cerr << "Fatal: could not allocate FPGA buffer in any memory group" << std::endl;
        exit(EXIT_FAILURE);
    }

    // --- 2. Map to host virtual address ---
    alloc.host_ptr = alloc.bo.map();
    if (!alloc.host_ptr) {
        std::cerr << "Fatal: xrt::bo::map() returned nullptr" << std::endl;
        exit(EXIT_FAILURE);
    }

    // --- 3. Register with HIP ---
    hipError_t err = hipHostRegister(
        alloc.host_ptr, size_bytes,
        hipHostRegisterMapped | hipHostRegisterIoMemory);

    if (err != hipSuccess) {
        // IoMemory may fail if IOMMU blocks it; fall back.
        // This is no longer true P2P — syncs will be required.
        std::cerr << "  Warning: hipHostRegisterIoMemory failed ("
                  << hipGetErrorString(err) << "), retrying with Mapped only" << std::endl;
        alloc.is_p2p = false;
        err = hipHostRegister(alloc.host_ptr, size_bytes, hipHostRegisterMapped);
        if (err != hipSuccess) {
            std::cerr << "Fatal: hipHostRegister failed: "
                      << hipGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    HIP_CHECK(hipHostGetDevicePointer(&alloc.device_ptr, alloc.host_ptr, 0));
    alloc.gpu_registered = true;

    return alloc;
}

void free_p2p_buffer(P2PAllocation& alloc) {
    if (alloc.gpu_registered) {
        HIP_CHECK(hipHostUnregister(alloc.host_ptr));
        alloc.gpu_registered = false;
    }
}

// ============================================================================
// Test 1: Simple sequential write
// ============================================================================

bool test_sequential_write(xrt::device& fpga_dev) {
    std::cout << "\n--- Test 1: Sequential GPU→FPGA write ---" << std::endl;

    constexpr uint32_t COUNT = 1024;
    constexpr size_t   BYTES = COUNT * sizeof(uint32_t);

    // Allocate P2P buffer
    auto alloc = alloc_p2p_buffer(fpga_dev, BYTES);
    std::cout << "  P2P buffer: " << BYTES << " B, is_p2p="
              << alloc.is_p2p << std::endl;

    // Clear the buffer from the host side so we can verify writes later
    std::memset(alloc.host_ptr, 0, BYTES);
    if (!alloc.is_p2p) {
        alloc.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // Prepare source data in GPU global memory
    uint32_t* d_src;
    HIP_CHECK(hipMalloc(&d_src, BYTES));

    std::vector<uint32_t> src_host(COUNT);
    std::iota(src_host.begin(), src_host.end(), 100);   // 100, 101, 102, ...
    HIP_CHECK(hipMemcpy(d_src, src_host.data(), BYTES, hipMemcpyHostToDevice));

    // Launch GPU kernel: write from GPU mem → FPGA P2P buffer
    dim3 threads(256);
    dim3 blocks((COUNT + 255) / 256);

    auto t0 = std::chrono::high_resolution_clock::now();

    hipLaunchKernelGGL(write_indices_to_fpga, blocks, threads, 0, 0,
                       d_src,
                       static_cast<uint32_t*>(alloc.device_ptr),
                       COUNT);
    HIP_CHECK(hipDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "  GPU kernel + sync: " << us << " us" << std::endl;

    // --- Synchronise before verification ---
    // For non-P2P: push host shadow to FPGA, then pull back to verify.
    // For P2P: sync from device to ensure the host mapping observes
    // GPU writes on platforms where the BAR isn't CPU-coherent.
    if (!alloc.is_p2p) {
        std::cout << "  (non-P2P fallback) calling sync_to_device..." << std::endl;
        alloc.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    alloc.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto* result = static_cast<uint32_t*>(alloc.host_ptr);
    int errors = 0;
    for (uint32_t i = 0; i < COUNT; i++) {
        if (result[i] != src_host[i]) {
            if (errors < 5) {
                std::cerr << "    MISMATCH [" << i << "]: got " << result[i]
                          << ", expected " << src_host[i] << std::endl;
            }
            errors++;
        }
    }

    HIP_CHECK(hipFree(d_src));
    free_p2p_buffer(alloc);

    if (errors == 0) {
        std::cout << "  PASSED (" << COUNT << " elements verified)" << std::endl;
    } else {
        std::cerr << "  FAILED: " << errors << " mismatches" << std::endl;
    }
    return errors == 0;
}

// ============================================================================
// Test 2: GPU generates a pattern directly into FPGA buffer (no staging)
// ============================================================================

bool test_pattern_generation(xrt::device& fpga_dev) {
    std::cout << "\n--- Test 2: GPU pattern generation → FPGA ---" << std::endl;

    constexpr uint32_t COUNT  = 2048;
    constexpr size_t   BYTES  = COUNT * sizeof(uint32_t);
    constexpr uint32_t START  = 42;
    constexpr uint32_t STRIDE = 3;

    auto alloc = alloc_p2p_buffer(fpga_dev, BYTES);
    std::memset(alloc.host_ptr, 0, BYTES);
    if (!alloc.is_p2p) alloc.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    dim3 threads(256);
    dim3 blocks((COUNT + 255) / 256);

    hipLaunchKernelGGL(generate_pattern_to_fpga, blocks, threads, 0, 0,
                       static_cast<uint32_t*>(alloc.device_ptr),
                       COUNT, START, STRIDE);
    HIP_CHECK(hipDeviceSynchronize());

    if (!alloc.is_p2p) {
        alloc.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    alloc.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto* result = static_cast<uint32_t*>(alloc.host_ptr);
    int errors = 0;
    for (uint32_t i = 0; i < COUNT; i++) {
        uint32_t expected = START + i * STRIDE;
        if (result[i] != expected) {
            if (errors < 5) {
                std::cerr << "    MISMATCH [" << i << "]: got " << result[i]
                          << ", expected " << expected << std::endl;
            }
            errors++;
        }
    }

    free_p2p_buffer(alloc);

    if (errors == 0) {
        std::cout << "  PASSED (" << COUNT << " elements)" << std::endl;
    } else {
        std::cerr << "  FAILED: " << errors << " mismatches" << std::endl;
    }
    return errors == 0;
}

// ============================================================================
// Test 3: Scatter-write to FPGA
// ============================================================================

bool test_scatter_write(xrt::device& fpga_dev) {
    std::cout << "\n--- Test 3: GPU scatter-write → FPGA ---" << std::endl;

    constexpr uint32_t BUFFER_SIZE = 4096;
    constexpr uint32_t WRITE_COUNT = 512;
    constexpr size_t   BYTES = BUFFER_SIZE * sizeof(uint32_t);

    auto alloc = alloc_p2p_buffer(fpga_dev, BYTES);
    // Fill with sentinel value so we can verify only written positions changed
    std::memset(alloc.host_ptr, 0xFF, BYTES);
    if (!alloc.is_p2p) alloc.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Generate unique random scatter indices on host
    std::mt19937 rng(7);
    std::set<uint32_t> idx_set;
    std::uniform_int_distribution<uint32_t> dist(0, BUFFER_SIZE - 1);
    while (idx_set.size() < WRITE_COUNT) idx_set.insert(dist(rng));
    std::vector<uint32_t> indices(idx_set.begin(), idx_set.end());

    // Data to write: value = index * 10
    std::vector<uint32_t> data(WRITE_COUNT);
    for (uint32_t i = 0; i < WRITE_COUNT; i++) data[i] = indices[i] * 10;

    // Upload to GPU
    uint32_t *d_data, *d_indices;
    HIP_CHECK(hipMalloc(&d_data,    WRITE_COUNT * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_indices, WRITE_COUNT * sizeof(uint32_t)));
    HIP_CHECK(hipMemcpy(d_data,    data.data(),    WRITE_COUNT * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_indices, indices.data(), WRITE_COUNT * sizeof(uint32_t), hipMemcpyHostToDevice));

    dim3 threads(256);
    dim3 blocks((WRITE_COUNT + 255) / 256);

    hipLaunchKernelGGL(scatter_write_to_fpga, blocks, threads, 0, 0,
                       d_data, d_indices,
                       static_cast<uint32_t*>(alloc.device_ptr),
                       WRITE_COUNT, BUFFER_SIZE);
    HIP_CHECK(hipDeviceSynchronize());

    if (!alloc.is_p2p) {
        alloc.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    alloc.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto* result = static_cast<uint32_t*>(alloc.host_ptr);
    int errors = 0;
    for (uint32_t i = 0; i < WRITE_COUNT; i++) {
        uint32_t pos = indices[i];
        if (result[pos] != data[i]) {
            if (errors < 5) {
                std::cerr << "    MISMATCH at pos " << pos << ": got " << result[pos]
                          << ", expected " << data[i] << std::endl;
            }
            errors++;
        }
    }

    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_indices));
    free_p2p_buffer(alloc);

    if (errors == 0) {
        std::cout << "  PASSED (" << WRITE_COUNT << " scattered writes verified)" << std::endl;
    } else {
        std::cerr << "  FAILED: " << errors << " mismatches" << std::endl;
    }
    return errors == 0;
}

// ============================================================================
// Test 4: Bandwidth benchmark (GPU → FPGA)
// ============================================================================

void test_bandwidth(xrt::device& fpga_dev) {
    std::cout << "\n--- Test 4: GPU→FPGA write bandwidth ---" << std::endl;

    // Test several sizes
    const std::vector<size_t> sizes = {
        4 * 1024,           //   4 KB
        64 * 1024,          //  64 KB
        1024 * 1024,        //   1 MB
        16 * 1024 * 1024,   //  16 MB
    };

    for (size_t bytes : sizes) {
        uint32_t count = bytes / sizeof(uint32_t);
        auto alloc = alloc_p2p_buffer(fpga_dev, bytes);

        // Prepare GPU source
        uint32_t* d_src;
        HIP_CHECK(hipMalloc(&d_src, bytes));
        HIP_CHECK(hipMemset(d_src, 0xAB, bytes));

        dim3 threads(256);
        dim3 blocks((count + 255) / 256);

        constexpr int WARMUP = 5;
        constexpr int ITERS  = 50;

        // Warm up
        for (int i = 0; i < WARMUP; i++) {
            hipLaunchKernelGGL(write_indices_to_fpga, blocks, threads, 0, 0,
                               d_src, static_cast<uint32_t*>(alloc.device_ptr), count);
        }
        HIP_CHECK(hipDeviceSynchronize());

        if (!alloc.is_p2p) {
            std::cout << "  WARNING: non-P2P fallback — bandwidth includes XRT sync cost" << std::endl;
        }

        // Timed iterations
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            hipLaunchKernelGGL(write_indices_to_fpga, blocks, threads, 0, 0,
                               d_src, static_cast<uint32_t*>(alloc.device_ptr), count);
            HIP_CHECK(hipDeviceSynchronize());
            if (!alloc.is_p2p) {
                alloc.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double sec = std::chrono::duration<double>(t1 - t0).count();
        double gbps = (double(bytes) * ITERS) / sec / 1e9;
        double avg_us = (sec / ITERS) * 1e6;

        char label[32];
        if (bytes >= 1024 * 1024)
            snprintf(label, sizeof(label), "%4zu MB", bytes / (1024 * 1024));
        else
            snprintf(label, sizeof(label), "%4zu KB", bytes / 1024);

        std::cout << "  " << label << ":  " << gbps << " GB/s  (avg "
                  << avg_us << " us/iter)" << std::endl;

        HIP_CHECK(hipFree(d_src));
        free_p2p_buffer(alloc);
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== GPU → FPGA P2P Transfer Demo ===" << std::endl;

    // --- Find FPGA ---
    std::cout << "\n[Init] Scanning for FPGA devices..." << std::endl;
    xrt::device fpga_dev;
    bool found = false;
    for (unsigned i = 0; i < 16 && !found; i++) {
        try {
            xrt::device dev(i);
            std::string name = dev.get_info<xrt::info::device::name>();
            std::string bdf  = dev.get_info<xrt::info::device::bdf>();
            std::cout << "  Found: index=" << i << "  BDF=" << bdf
                      << "  name=" << name << std::endl;
            fpga_dev = std::move(dev);
            found = true;
        } catch (...) {}
    }
    if (!found) {
        std::cerr << "No FPGA device found." << std::endl;
        return EXIT_FAILURE;
    }

    // --- Find GPU ---
    std::cout << "\n[Init] Scanning for GPU devices..." << std::endl;
    int gpu_count = 0;
    HIP_CHECK(hipGetDeviceCount(&gpu_count));
    if (gpu_count == 0) {
        std::cerr << "No GPU found." << std::endl;
        return EXIT_FAILURE;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::cout << "  Using GPU 0: " << props.name << std::endl;

    // --- Run tests ---
    int pass = 0, fail = 0;

    test_sequential_write(fpga_dev)  ? pass++ : fail++;
    test_pattern_generation(fpga_dev) ? pass++ : fail++;
    test_scatter_write(fpga_dev)      ? pass++ : fail++;
    test_bandwidth(fpga_dev);

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "  Passed: " << pass << std::endl;
    std::cout << "  Failed: " << fail << std::endl;

    return fail > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}