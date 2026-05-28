/**
 * Simple FPGA-GPU P2P Demo (BAR-based)
 * 
 * This is a simplified version that directly maps the FPGA's PCIe BAR
 * and uses the GPU to read/write to it. This approach:
 * 
 * 1. Does NOT require building a custom FPGA xclbin
 * 2. Uses the P2P BAR exposed by the U55C shell
 * 3. Demonstrates raw P2P memory access
 * 
 * The FPGA's HBM is exposed through PCIe BAR4 when P2P is enabled.
 * The GPU can directly DMA to/from this BAR.
 * 
 * Build: make simple
 * Run: ./p2p_simple --fpga 81:00.1 --gpu 0
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <getopt.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

// XRT includes for P2P buffer access
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>

// HIP includes
#include <hip/hip_runtime.h>

// ============================================================================
// Error checking macros
// ============================================================================

#define HIP_CHECK(cmd)                                                         \
    do {                                                                       \
        hipError_t error = (cmd);                                              \
        if (error != hipSuccess) {                                             \
            std::cerr << "HIP Error: " << hipGetErrorString(error)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ============================================================================
// P2P Status Check Function
// ============================================================================

std::string check_p2p_status(const std::string& bdf) {
    // Try to read P2P status from sysfs (may not work on clusters)
    std::string sysfs_path = "/sys/bus/pci/devices/0000:" + bdf + "/p2p_enable";
    std::ifstream p2p_file(sysfs_path);
    if (p2p_file.is_open()) {
        std::string status;
        std::getline(p2p_file, status);
        return status;
    }
    // Return "unavailable" if we can't read sysfs (common on clusters)
    return "unavailable";
}

// ============================================================================
// GPU Kernels (inline for simplicity)
// ============================================================================

__global__ void gpu_fill_pattern(uint32_t* dst, uint32_t num_words, uint32_t pattern) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    for (uint32_t i = idx; i < num_words; i += stride) {
        dst[i] = pattern + i;
    }
}

__global__ void gpu_verify_pattern(const uint32_t* src, uint32_t num_words, 
                                   uint32_t pattern, uint32_t* errors) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    uint32_t local_errors = 0;
    for (uint32_t i = idx; i < num_words; i += stride) {
        if (src[i] != pattern + i) {
            local_errors++;
        }
    }
    
    if (local_errors > 0) {
        atomicAdd(errors, local_errors);
    }
}

__global__ void gpu_copy(const uint32_t* src, uint32_t* dst, uint32_t num_words) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    for (uint32_t i = idx; i < num_words; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void gpu_copy_vec4(const float4* src, float4* dst, uint32_t num_vec) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    for (uint32_t i = idx; i < num_vec; i += stride) {
        dst[i] = src[i];
    }
}

// ============================================================================
// Configuration
// ============================================================================

struct Config {
    std::string fpga_bdf = "81:00.1";
    int gpu_id = 0;
    size_t buffer_size_mb = 64;
    int num_iterations = 10;
    int warmup_iterations = 3;
    bool verbose = false;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --fpga <bdf>       FPGA PCIe BDF (default: 81:00.1)\n"
              << "  --gpu <id>         GPU device ID (default: 0)\n"
              << "  --size <mb>        Buffer size in MB (default: 64)\n"
              << "  --iterations <n>   Number of iterations (default: 10)\n"
              << "  --verbose          Verbose output\n"
              << "  --help             Show this help\n";
}

Config parse_args(int argc, char* argv[]) {
    Config cfg;
    
    static struct option long_options[] = {
        {"fpga",       required_argument, 0, 'f'},
        {"gpu",        required_argument, 0, 'g'},
        {"size",       required_argument, 0, 's'},
        {"iterations", required_argument, 0, 'i'},
        {"verbose",    no_argument,       0, 'v'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "f:g:s:i:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'f': cfg.fpga_bdf = optarg; break;
            case 'g': cfg.gpu_id = std::atoi(optarg); break;
            case 's': cfg.buffer_size_mb = std::atoi(optarg); break;
            case 'i': cfg.num_iterations = std::atoi(optarg); break;
            case 'v': cfg.verbose = true; break;
            case 'h':
                print_usage(argv[0]);
                exit(0);
            default:
                print_usage(argv[0]);
                exit(1);
        }
    }

    return cfg;
}

// ============================================================================
// Timer utility
// ============================================================================

class Timer {
public:
    void start() { 
        HIP_CHECK(hipEventRecord(start_event_));
    }
    
    void stop() { 
        HIP_CHECK(hipEventRecord(stop_event_));
        HIP_CHECK(hipEventSynchronize(stop_event_));
    }
    
    float elapsed_ms() const {
        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, start_event_, stop_event_));
        return ms;
    }
    
    double bandwidth_gbps(size_t bytes) const {
        float ms = elapsed_ms();
        double seconds = ms / 1000.0;
        return (bytes / (1024.0 * 1024.0 * 1024.0)) / seconds;
    }
    
    Timer() {
        HIP_CHECK(hipEventCreate(&start_event_));
        HIP_CHECK(hipEventCreate(&stop_event_));
    }
    
    ~Timer() {
        hipEventDestroy(start_event_);
        hipEventDestroy(stop_event_);
    }

private:
    hipEvent_t start_event_, stop_event_;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);
    
    std::cout << "========================================\n";
    std::cout << "Simple FPGA-GPU P2P Demo (XRT P2P BO)\n";
    std::cout << "========================================\n";
    std::cout << "FPGA BDF:      " << cfg.fpga_bdf << "\n";
    std::cout << "GPU ID:        " << cfg.gpu_id << "\n";
    std::cout << "Buffer Size:   " << cfg.buffer_size_mb << " MB\n";
    std::cout << "Iterations:    " << cfg.num_iterations << "\n";
    std::cout << "========================================\n\n";

    const size_t buffer_size = cfg.buffer_size_mb * 1024 * 1024;
    const size_t num_words = buffer_size / sizeof(uint32_t);
    const size_t num_vec4 = buffer_size / sizeof(float4);
    
    // ========================================================================
    // Step 1: Initialize GPU
    // ========================================================================
    std::cout << "[1/5] Initializing GPU...\n";
    
    int gpu_count;
    HIP_CHECK(hipGetDeviceCount(&gpu_count));
    if (cfg.gpu_id >= gpu_count) {
        std::cerr << "Error: GPU " << cfg.gpu_id << " not found\n";
        return 1;
    }
    
    HIP_CHECK(hipSetDevice(cfg.gpu_id));
    
    hipDeviceProp_t gpu_props;
    HIP_CHECK(hipGetDeviceProperties(&gpu_props, cfg.gpu_id));
    std::cout << "  GPU: " << gpu_props.name << "\n";
    std::cout << "  PCIe: " << std::hex << std::setfill('0') 
              << std::setw(2) << gpu_props.pciBusID << ":"
              << std::setw(2) << gpu_props.pciDeviceID << "."
              << gpu_props.pciDomainID << std::dec << "\n\n";

    // Allocate GPU local memory
    uint32_t* d_gpu_buffer;
    uint32_t* d_errors;
    HIP_CHECK(hipMalloc(&d_gpu_buffer, buffer_size));
    HIP_CHECK(hipMalloc(&d_errors, sizeof(uint32_t)));
    
    // ========================================================================
    // Step 2: Initialize FPGA and create P2P buffer
    // ========================================================================
    std::cout << "[2/5] Initializing FPGA and creating P2P buffer...\n";
    
    // Find FPGA device by BDF
    // Try device indices 0-7 (typical max number of FPGAs)
    unsigned int device_index = 0;
    bool found = false;
    
    std::cout << "  Scanning for FPGA devices...\n";
    for (unsigned int i = 0; i < 8; i++) {
        try {
            xrt::device test_dev(i);
            std::string bdf = test_dev.get_info<xrt::info::device::bdf>();
            std::string name = test_dev.get_info<xrt::info::device::name>();
            std::cout << "  Device " << i << ": " << bdf << " (" << name << ")\n";
            
            if (bdf.find(cfg.fpga_bdf) != std::string::npos) {
                device_index = i;
                found = true;
                std::cout << "  -> Selected device " << i << "\n";
            }
        } catch (const std::exception& e) {
            if (cfg.verbose) {
                std::cout << "  Device " << i << ": " << e.what() << "\n";
            }
            continue;
        }
    }
    
    if (!found) {
        std::cerr << "Error: FPGA with BDF " << cfg.fpga_bdf << " not found\n";
        std::cerr << "Try running: xbutil examine\n";
        return 1;
    }
    
    xrt::device fpga_device(device_index);
    std::cout << "  FPGA: " << fpga_device.get_info<xrt::info::device::name>() << "\n";
    
    // ========================================================================
    // Check P2P status before attempting buffer creation
    // ========================================================================
    std::cout << "  Checking P2P configuration...\n";
    
    std::string p2p_status = check_p2p_status(cfg.fpga_bdf);
    
    if (p2p_status == "unavailable") {
        std::cout << "  Note: Cannot read P2P status from sysfs (normal on clusters)\n";
        std::cout << "  Assuming P2P is enabled, proceeding with buffer creation...\n";
    } else if (p2p_status == "0" || p2p_status == "disabled") {
        std::cerr << "\nERROR: P2P is DISABLED. Enable with:\n";
        std::cerr << "  sudo xbutil configure --device 0000:" << cfg.fpga_bdf << " --p2p enable\n";
        return 1;
    } else if (p2p_status == "1" || p2p_status == "enabled") {
        std::cout << "  ✓ P2P is ENABLED\n";
    } else {
        std::cout << "  P2P status: " << p2p_status << "\n";
    }
    
    // Create P2P buffer on FPGA
    // For U55C with HBM, memory groups are typically:
    // 0-31 for HBM banks (pseudo channels)
    // Note: P2P buffer creation requires the shell to expose P2P BAR
    std::cout << "  Creating P2P buffer (" << buffer_size / (1024*1024) << " MB)...\n";
    
    xrt::bo p2p_buffer;
    void* p2p_ptr = nullptr;
    
    // Try different memory groups - U55C has 32 HBM pseudo-channels (0-31)
    // Also try with different buffer sizes if large allocation fails
    std::vector<int> mem_groups_to_try = {0, 1, 2, 8, 16, 31};
    std::vector<size_t> sizes_to_try = {buffer_size, buffer_size / 4, 4 * 1024 * 1024};  // Try smaller if fails
    bool buffer_created = false;
    size_t actual_size = buffer_size;
    
    for (size_t try_size : sizes_to_try) {
        if (buffer_created) break;
        
        for (int mem_group : mem_groups_to_try) {
            try {
                if (cfg.verbose) {
                    std::cout << "  Trying memory group " << mem_group 
                              << " with size " << try_size / (1024*1024) << " MB...\n";
                }
                p2p_buffer = xrt::bo(fpga_device, try_size, xrt::bo::flags::p2p, mem_group);
                p2p_ptr = p2p_buffer.map();
                if (p2p_ptr) {
                    std::cout << "  ✓ P2P buffer created: " << try_size / (1024*1024) 
                              << " MB on memory group " << mem_group << "\n";
                    buffer_created = true;
                    actual_size = try_size;
                    break;
                }
            } catch (const std::exception& e) {
                if (cfg.verbose) {
                    std::cout << "  Memory group " << mem_group << " failed: " << e.what() << "\n";
                }
                continue;
            } catch (...) {
                if (cfg.verbose) {
                    std::cout << "  Memory group " << mem_group << " failed with unknown error\n";
                }
                continue;
            }
        }
        
        if (!buffer_created && try_size != sizes_to_try.back()) {
            std::cout << "  Trying smaller buffer size...\n";
        }
    }
    
    // If P2P buffers failed, try a regular device buffer as fallback
    if (!buffer_created) {
        std::cout << "\n  P2P buffer creation failed. Trying regular device buffer...\n";
        std::cout << "  (This will use CPU-mediated transfer, not true P2P)\n\n";
        
        try {
            // Create a regular device buffer (not P2P)
            p2p_buffer = xrt::bo(fpga_device, buffer_size, xrt::bo::flags::normal, 0);
            p2p_ptr = p2p_buffer.map();
            if (!p2p_ptr) {
                std::cerr << "Error: Failed to create any buffer on FPGA\n";
                return 1;
            }
            std::cout << "  Created regular device buffer (not P2P)\n";
            buffer_created = true;
        } catch (const std::exception& e) {
            std::cerr << "Error: All buffer creation attempts failed: " << e.what() << "\n";
            std::cerr << "\nThis could be due to:\n";
            std::cerr << "  1. FPGA shell doesn't support P2P\n";
            std::cerr << "  2. P2P is not enabled\n";
            std::cerr << "  3. Insufficient permissions\n";
            std::cerr << "  4. No xclbin loaded with memory topology\n";
            std::cerr << "\nCheck with: xbutil examine -d 0000:" << cfg.fpga_bdf << " -r platform\n";
            return 1;
        }
    }
    
    // Update sizes based on actual allocation
    const size_t actual_num_words = actual_size / sizeof(uint32_t);
    const size_t actual_num_vec4 = actual_size / sizeof(float4);
    
    std::cout << "  P2P buffer: " << actual_size / (1024*1024) << " MB at " << p2p_ptr << "\n\n";
    
    // ========================================================================
    // Step 3: Register P2P memory with HIP
    // ========================================================================
    std::cout << "[3/5] Registering P2P buffer with ROCm...\n";
    
    // Try registering as I/O memory (for true P2P)
    hipError_t reg_status = hipHostRegister(p2p_ptr, actual_size, 
                                            hipHostRegisterMapped | hipHostRegisterIoMemory);
    
    bool true_p2p = (reg_status == hipSuccess);
    
    if (!true_p2p) {
        std::cout << "  Note: hipHostRegisterIoMemory failed (" 
                  << hipGetErrorString(reg_status) << ")\n";
        std::cout << "  Trying standard registration...\n";
        
        // Fall back to standard registration
        reg_status = hipHostRegister(p2p_ptr, actual_size, hipHostRegisterMapped);
        if (reg_status != hipSuccess) {
            std::cerr << "Error: hipHostRegister failed: " 
                      << hipGetErrorString(reg_status) << "\n";
            return 1;
        }
        std::cout << "  WARNING: Using CPU-mediated transfer (not true P2P)\n";
    } else {
        std::cout << "  ✓ True P2P enabled (hipHostRegisterIoMemory)\n";
    }
    
    // Get device-accessible pointer
    void* d_p2p_ptr;
    HIP_CHECK(hipHostGetDevicePointer(&d_p2p_ptr, p2p_ptr, 0));
    std::cout << "  GPU-accessible pointer: " << d_p2p_ptr << "\n\n";
    
    // ========================================================================
    // Step 4: Benchmark GPU -> FPGA (Write to FPGA HBM)
    // ========================================================================
    std::cout << "[4/5] Benchmarking GPU -> FPGA transfer...\n";
    
    Timer timer;
    const int block_size = 256;
    const int num_blocks = std::min((size_t)65535, (actual_num_vec4 + block_size - 1) / block_size);
    
    // Initialize GPU buffer with test pattern
    const uint32_t test_pattern = 0xDEAD0000;
    hipLaunchKernelGGL(gpu_fill_pattern, dim3(num_blocks), dim3(block_size), 0, 0,
                      d_gpu_buffer, actual_num_words, test_pattern);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Warmup
    for (int i = 0; i < cfg.warmup_iterations; i++) {
        hipLaunchKernelGGL(gpu_copy_vec4, dim3(num_blocks), dim3(block_size), 0, 0,
                          (float4*)d_gpu_buffer, (float4*)d_p2p_ptr, actual_num_vec4);
        HIP_CHECK(hipDeviceSynchronize());
    }
    
    // Benchmark GPU -> FPGA
    std::vector<float> write_times;
    for (int i = 0; i < cfg.num_iterations; i++) {
        timer.start();
        hipLaunchKernelGGL(gpu_copy_vec4, dim3(num_blocks), dim3(block_size), 0, 0,
                          (float4*)d_gpu_buffer, (float4*)d_p2p_ptr, actual_num_vec4);
        timer.stop();
        
        write_times.push_back(timer.elapsed_ms());
        
        if (cfg.verbose) {
            std::cout << "  Write " << i << ": " << std::fixed << std::setprecision(2)
                      << timer.elapsed_ms() << " ms, " 
                      << timer.bandwidth_gbps(actual_size) << " GB/s\n";
        }
    }
    
    // Calculate average
    float avg_write = 0;
    for (auto t : write_times) avg_write += t;
    avg_write /= write_times.size();
    double write_bw = (actual_size / (1024.0 * 1024.0 * 1024.0)) / (avg_write / 1000.0);
    
    std::cout << "  Average: " << std::fixed << std::setprecision(2) 
              << avg_write << " ms, " << write_bw << " GB/s\n\n";
    
    // Verify data on CPU (read from FPGA P2P buffer)
    std::cout << "  Verifying data...\n";
    p2p_buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    
    uint32_t* host_verify = (uint32_t*)p2p_ptr;
    int verify_errors = 0;
    for (size_t i = 0; i < std::min((size_t)1000, actual_num_words); i++) {
        if (host_verify[i] != test_pattern + i) {
            verify_errors++;
            if (verify_errors <= 5) {
                std::cout << "    Error at " << i << ": expected " << std::hex 
                          << (test_pattern + i) << ", got " << host_verify[i] 
                          << std::dec << "\n";
            }
        }
    }
    if (verify_errors == 0) {
        std::cout << "  ✓ Verification passed\n\n";
    } else {
        std::cout << "  ✗ Verification failed with " << verify_errors << " errors\n\n";
    }
    
    // ========================================================================
    // Step 5: Benchmark FPGA -> GPU (Read from FPGA HBM)
    // ========================================================================
    std::cout << "[5/5] Benchmarking FPGA -> GPU transfer...\n";
    
    // Write a different pattern to FPGA buffer via CPU
    const uint32_t fpga_pattern = 0xCAFE0000;
    for (size_t i = 0; i < actual_num_words; i++) {
        host_verify[i] = fpga_pattern + i;
    }
    p2p_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // Warmup
    for (int i = 0; i < cfg.warmup_iterations; i++) {
        hipLaunchKernelGGL(gpu_copy_vec4, dim3(num_blocks), dim3(block_size), 0, 0,
                          (float4*)d_p2p_ptr, (float4*)d_gpu_buffer, actual_num_vec4);
        HIP_CHECK(hipDeviceSynchronize());
    }
    
    // Benchmark FPGA -> GPU
    std::vector<float> read_times;
    for (int i = 0; i < cfg.num_iterations; i++) {
        timer.start();
        hipLaunchKernelGGL(gpu_copy_vec4, dim3(num_blocks), dim3(block_size), 0, 0,
                          (float4*)d_p2p_ptr, (float4*)d_gpu_buffer, actual_num_vec4);
        timer.stop();
        
        read_times.push_back(timer.elapsed_ms());
        
        if (cfg.verbose) {
            std::cout << "  Read " << i << ": " << std::fixed << std::setprecision(2)
                      << timer.elapsed_ms() << " ms, " 
                      << timer.bandwidth_gbps(actual_size) << " GB/s\n";
        }
    }
    
    // Calculate average
    float avg_read = 0;
    for (auto t : read_times) avg_read += t;
    avg_read /= read_times.size();
    double read_bw = (actual_size / (1024.0 * 1024.0 * 1024.0)) / (avg_read / 1000.0);
    
    std::cout << "  Average: " << std::fixed << std::setprecision(2) 
              << avg_read << " ms, " << read_bw << " GB/s\n\n";
    
    // Verify data on GPU
    std::cout << "  Verifying data...\n";
    HIP_CHECK(hipMemset(d_errors, 0, sizeof(uint32_t)));
    hipLaunchKernelGGL(gpu_verify_pattern, dim3(num_blocks), dim3(block_size), 0, 0,
                      d_gpu_buffer, actual_num_words, fpga_pattern, d_errors);
    
    uint32_t gpu_errors;
    HIP_CHECK(hipMemcpy(&gpu_errors, d_errors, sizeof(uint32_t), hipMemcpyDeviceToHost));
    
    if (gpu_errors == 0) {
        std::cout << "  ✓ Verification passed\n\n";
    } else {
        std::cout << "  ✗ Verification failed with " << gpu_errors << " errors\n\n";
    }
    
    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "========================================\n";
    std::cout << "Summary\n";
    std::cout << "========================================\n";
    std::cout << "Buffer size:        " << actual_size / (1024*1024) << " MB\n";
    std::cout << "P2P mode:           " << (true_p2p ? "True P2P" : "CPU-mediated") << "\n";
    std::cout << "GPU -> FPGA:        " << std::fixed << std::setprecision(2) 
              << write_bw << " GB/s (" << avg_write << " ms)\n";
    std::cout << "FPGA -> GPU:        " << std::fixed << std::setprecision(2) 
              << read_bw << " GB/s (" << avg_read << " ms)\n";
    std::cout << "========================================\n";
    
    // Cleanup
    HIP_CHECK(hipHostUnregister(p2p_ptr));
    HIP_CHECK(hipFree(d_gpu_buffer));
    HIP_CHECK(hipFree(d_errors));
    
    std::cout << "\nDone!\n";
    return 0;
}
