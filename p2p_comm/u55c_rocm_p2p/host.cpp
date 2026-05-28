/**
 * P2P Transfer Host Application (OpenCL Version)
 * 
 * This application demonstrates PCIe peer-to-peer data transfer between
 * a Xilinx U55C FPGA and an AMD MI210 GPU, bypassing system DRAM.
 * 
 * Uses OpenCL APIs with Xilinx extensions for P2P buffer creation:
 * 1. Load xclbin and create OpenCL context/program/kernel
 * 2. Create P2P buffer with XCL_MEM_EXT_P2P_BUFFER flag
 * 3. Set kernel argument and map P2P buffer to host space
 * 4. Register mapped pointer with ROCm using hipHostRegister
 * 5. GPU can then DMA directly to/from this address
 * 
 * Based on Xilinx XRT P2P documentation:
 * https://xilinx.github.io/XRT/2024.1/html/p2p.html
 * 
 * Build: See Makefile
 * Run: ./p2p_transfer --xclbin <file> --fpga <bdf> --gpu <id>
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
#include <sys/stat.h>

// OpenCL includes
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#include <CL/cl2.hpp>

// Xilinx OpenCL extensions for P2P
#include <CL/cl_ext_xilinx.h>

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

#define OCL_CHECK(err, call)                                                   \
    do {                                                                       \
        call;                                                                  \
        if (err != CL_SUCCESS) {                                               \
            std::cerr << "OpenCL Error: " << err << " at " << __FILE__         \
                      << ":" << __LINE__ << std::endl;                         \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ============================================================================
// GPU Kernel declarations (defined in gpu_kernel.hip)
// ============================================================================

extern "C" __global__ void gpu_write_pattern(uint32_t* dst, uint32_t num_words, uint32_t pattern);
extern "C" __global__ void gpu_read_verify(const uint32_t* src, uint32_t num_words, uint32_t expected, uint32_t* error_count);
extern "C" __global__ void gpu_memcpy(const uint32_t* src, uint32_t* dst, uint32_t num_words);
extern "C" __global__ void gpu_memcpy_vec4(const float4* src, float4* dst, uint32_t num_vec);

// ============================================================================
// Utility: Load binary file (xclbin)
// ============================================================================

std::vector<unsigned char> load_binary_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<unsigned char> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    file.close();
    
    return buffer;
}

// ============================================================================
// Configuration
// ============================================================================

struct Config {
    std::string xclbin_path;
    std::string kernel_name = "dummy_kernel";  // Default kernel name
    std::string fpga_bdf = "81:00.1";
    int gpu_id = 0;
    size_t buffer_size_kb = 64 * 1024;  // Default 64 MB in KB
    int num_iterations = 10;
    int warmup_iterations = 3;
    bool verify = true;
    bool verbose = false;
    bool use_p2p = true;  // Use P2P transfer by default, false = use system DRAM
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --xclbin <file>    FPGA xclbin file (REQUIRED)\n"
              << "  --kernel <name>    Kernel name in xclbin (default: dummy_kernel)\n"
              << "  --fpga <bdf>       FPGA PCIe BDF (default: 81:00.1)\n"
              << "  --gpu <id>         GPU device ID (default: 0)\n"
              << "  --size <kb>        Buffer size in KB (default: 65536 = 64MB)\n"
              << "  --iterations <n>   Number of iterations (default: 10)\n"
              << "  --no-verify        Skip data verification\n"
              << "  --no-p2p           Use non-P2P mode (transfer via system DRAM)\n"
              << "  --verbose          Verbose output\n"
              << "  --help             Show this help\n";
}

Config parse_args(int argc, char* argv[]) {
    Config cfg;
    
    static struct option long_options[] = {
        {"xclbin",     required_argument, 0, 'x'},
        {"kernel",     required_argument, 0, 'k'},
        {"fpga",       required_argument, 0, 'f'},
        {"gpu",        required_argument, 0, 'g'},
        {"size",       required_argument, 0, 's'},
        {"iterations", required_argument, 0, 'i'},
        {"no-verify",  no_argument,       0, 'n'},
        {"no-p2p",     no_argument,       0, 'p'},
        {"verbose",    no_argument,       0, 'v'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "x:k:f:g:s:i:npvh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'x': cfg.xclbin_path = optarg; break;
            case 'k': cfg.kernel_name = optarg; break;
            case 'f': cfg.fpga_bdf = optarg; break;
            case 'g': cfg.gpu_id = std::atoi(optarg); break;
            case 's': cfg.buffer_size_kb = std::atoi(optarg); break;
            case 'i': cfg.num_iterations = std::atoi(optarg); break;
            case 'n': cfg.verify = false; break;
            case 'p': cfg.use_p2p = false; break;
            case 'v': cfg.verbose = true; break;
            case 'h':
                print_usage(argv[0]);
                exit(0);
            default:
                print_usage(argv[0]);
                exit(1);
        }
    }

    if (cfg.xclbin_path.empty()) {
        std::cerr << "Error: --xclbin is required\n\n";
        print_usage(argv[0]);
        exit(1);
    }

    return cfg;
}

// ============================================================================
// Find Xilinx FPGA device by BDF
// ============================================================================

cl::Device find_fpga_device(const std::string& target_bdf, bool verbose) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    
    for (auto& platform : platforms) {
        std::string platform_name = platform.getInfo<CL_PLATFORM_NAME>();
        if (verbose) {
            std::cout << "  Platform: " << platform_name << "\n";
        }
        
        // Look for Xilinx platform
        if (platform_name.find("Xilinx") == std::string::npos) {
            continue;
        }
        
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
        
        for (auto& device : devices) {
            std::string device_name = device.getInfo<CL_DEVICE_NAME>();
            if (verbose) {
                std::cout << "  Device: " << device_name << "\n";
            }
            
            // Check if this is the target device (by BDF in name or any U55C)
            if (device_name.find(target_bdf) != std::string::npos ||
                device_name.find("u55c") != std::string::npos ||
                device_name.find("U55C") != std::string::npos) {
                return device;
            }
        }
        
        // If no specific match found, return first Xilinx accelerator
        if (!devices.empty()) {
            return devices[0];
        }
    }
    
    throw std::runtime_error("No Xilinx FPGA device found");
}

// ============================================================================
// Main P2P Transfer Demo
// ============================================================================

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);
    
    std::cout << "========================================\n";
    std::cout << "FPGA-GPU P2P Transfer Demo (OpenCL)\n";
    std::cout << "========================================\n";
    std::cout << "XCLBIN:       " << cfg.xclbin_path << "\n";
    std::cout << "Kernel:       " << cfg.kernel_name << "\n";
    std::cout << "FPGA BDF:     " << cfg.fpga_bdf << "\n";
    std::cout << "GPU ID:       " << cfg.gpu_id << "\n";
    std::cout << "Buffer Size:  " << cfg.buffer_size_kb << " KB (" 
              << (cfg.buffer_size_kb / 1024.0) << " MB)\n";
    std::cout << "Iterations:   " << cfg.num_iterations << "\n";
    std::cout << "Verification: " << (cfg.verify ? "enabled" : "disabled") << "\n";
    std::cout << "Transfer Mode: " << (cfg.use_p2p ? "P2P (direct)" : "Non-P2P (via system DRAM)") << "\n";
    std::cout << "========================================\n\n";

    const size_t buffer_size = cfg.buffer_size_kb * 1024;
    const size_t num_words = buffer_size / sizeof(uint32_t);
    const size_t num_vec4 = buffer_size / sizeof(float4);

    cl_int err;

    // ========================================================================
    // Step 1: Initialize GPU (HIP/ROCm)
    // ========================================================================
    std::cout << "[1/6] Initializing GPU...\n";
    
    int gpu_count;
    HIP_CHECK(hipGetDeviceCount(&gpu_count));
    if (cfg.gpu_id >= gpu_count) {
        std::cerr << "Error: GPU " << cfg.gpu_id << " not found. Available: " << gpu_count << "\n";
        return 1;
    }
    
    HIP_CHECK(hipSetDevice(cfg.gpu_id));
    
    hipDeviceProp_t gpu_props;
    HIP_CHECK(hipGetDeviceProperties(&gpu_props, cfg.gpu_id));
    std::cout << "  GPU: " << gpu_props.name << "\n";
    std::cout << "  PCIe Bus ID: " << std::hex << std::setfill('0') 
              << std::setw(2) << gpu_props.pciBusID << ":"
              << std::setw(2) << gpu_props.pciDeviceID << "."
              << gpu_props.pciDomainID << std::dec << "\n\n";

    // Allocate GPU device memory
    uint32_t* d_gpu_buffer;
    uint32_t* d_error_count;
    
    HIP_CHECK(hipMalloc(&d_gpu_buffer, buffer_size));
    HIP_CHECK(hipMalloc(&d_error_count, sizeof(uint32_t)));
    
    // Allocate host staging buffer for non-P2P mode
    uint32_t* h_staging_buffer = nullptr;
    void* d_staging_ptr = nullptr;  // GPU-accessible pointer to staging buffer
    if (!cfg.use_p2p) {
        // Use hipHostMallocMapped so GPU kernel can access the staging buffer
        HIP_CHECK(hipHostMalloc(&h_staging_buffer, buffer_size, hipHostMallocMapped));
        HIP_CHECK(hipHostGetDevicePointer(&d_staging_ptr, h_staging_buffer, 0));
        std::cout << "  Allocated host staging buffer: " << buffer_size / 1024 << " KB\n";
        std::cout << "  Staging buffer GPU-accessible address: " << d_staging_ptr << "\n";
    }
    
    // ========================================================================
    // Step 2: Initialize FPGA (OpenCL)
    // ========================================================================
    std::cout << "[2/6] Initializing FPGA...\n";
    
    cl::Device fpga_device;
    try {
        fpga_device = find_fpga_device(cfg.fpga_bdf, cfg.verbose);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    std::cout << "  FPGA: " << fpga_device.getInfo<CL_DEVICE_NAME>() << "\n\n";
    
    // Create OpenCL context
    cl::Context context(fpga_device, nullptr, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to create OpenCL context, error " << err << "\n";
        return 1;
    }
    
    // Create command queue
    cl::CommandQueue queue(context, fpga_device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to create command queue, error " << err << "\n";
        return 1;
    }
    
    // ========================================================================
    // Step 3: Load xclbin and create kernel
    // ========================================================================
    std::cout << "[3/6] Loading xclbin...\n";
    
    std::vector<unsigned char> xclbin_data = load_binary_file(cfg.xclbin_path);
    cl::Program::Binaries binaries{{xclbin_data.data(), xclbin_data.size()}};
    
    std::vector<cl::Device> devices{fpga_device};
    cl::Program program(context, devices, binaries, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to load xclbin, error " << err << "\n";
        return 1;
    }
    
    std::cout << "  xclbin loaded successfully\n";
    
    // Create kernel
    cl::Kernel kernel(program, cfg.kernel_name.c_str(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to create kernel '" << cfg.kernel_name << "', error " << err << "\n";
        std::cerr << "  Hint: Use --kernel to specify the correct kernel name in your xclbin\n";
        return 1;
    }
    std::cout << "  Kernel '" << cfg.kernel_name << "' created\n\n";
    
    // ========================================================================
    // Step 4: Create FPGA buffer (P2P or regular based on mode)
    // ========================================================================
    std::cout << "[4/6] Creating FPGA buffer...\n";
    
    cl::Buffer fpga_buffer;
    cl::Buffer fpga_buffer_out;
    
    if (cfg.use_p2p) {
        // Create P2P buffer using XCL_MEM_EXT_P2P_BUFFER flag
        // This is the key for P2P - the buffer will be mapped to PCIe BAR
        cl_mem_ext_ptr_t p2p_ext = {0, nullptr, 0};
        p2p_ext.flags = XCL_MEM_EXT_P2P_BUFFER;
        
        fpga_buffer = cl::Buffer(context, 
                              CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX,
                              buffer_size, 
                              &p2p_ext, 
                              &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Error: Failed to create P2P buffer, error " << err << "\n";
            std::cerr << "  This may indicate P2P is not enabled on your FPGA\n";
            return 1;
        }
        
        std::cout << "  P2P buffer created: " << buffer_size / 1024 << " KB\n";
        
        // Create a second P2P buffer for output (fpga_memcpy needs in and out buffers)
        cl_mem_ext_ptr_t p2p_ext_out = {0, nullptr, 0};
        p2p_ext_out.flags = XCL_MEM_EXT_P2P_BUFFER;
        
        fpga_buffer_out = cl::Buffer(context, 
                                  CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX,
                                  buffer_size, 
                                  &p2p_ext_out, 
                                  &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Error: Failed to create P2P output buffer, error " << err << "\n";
            return 1;
        }
        std::cout << "  P2P output buffer created: " << buffer_size / 1024 << " KB\n";
    } else {
        // Create regular device buffers for non-P2P mode
        fpga_buffer = cl::Buffer(context, 
                              CL_MEM_READ_WRITE,
                              buffer_size, 
                              nullptr, 
                              &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Error: Failed to create FPGA buffer, error " << err << "\n";
            return 1;
        }
        
        std::cout << "  Regular FPGA buffer created: " << buffer_size / 1024 << " KB\n";
        
        fpga_buffer_out = cl::Buffer(context, 
                                  CL_MEM_READ_WRITE,
                                  buffer_size, 
                                  nullptr, 
                                  &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Error: Failed to create FPGA output buffer, error " << err << "\n";
            return 1;
        }
        std::cout << "  Regular FPGA output buffer created: " << buffer_size / 1024 << " KB\n";
    }
    
    // Set kernel arguments for fpga_memcpy(in, out, num_beats)
    // Argument 0: in (input buffer)
    err = kernel.setArg(0, fpga_buffer);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to set kernel argument 0 (in), error " << err << "\n";
        return 1;
    }
    
    // Argument 1: out (output buffer)
    err = kernel.setArg(1, fpga_buffer_out);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to set kernel argument 1 (out), error " << err << "\n";
        return 1;
    }
    
    // Argument 2: num_beats (number of 64-byte beats to copy)
    unsigned int num_beats = buffer_size / 64;  // 64 bytes per beat (512 bits)
    err = kernel.setArg(2, num_beats);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to set kernel argument 2 (num_beats), error " << err << "\n";
        return 1;
    }
    
    std::cout << "  Kernel arguments set (in, out, num_beats=" << num_beats << ")\n";
    
    void* fpga_mapped_ptr = nullptr;
    void* d_fpga_ptr = nullptr;
    bool true_p2p = false;
    
    if (cfg.use_p2p) {
        // Map P2P buffer to host address space
        // This gives us a pointer that can be registered with HIP
        fpga_mapped_ptr = queue.enqueueMapBuffer(fpga_buffer, 
                                                CL_TRUE,
                                                CL_MAP_READ | CL_MAP_WRITE,
                                                0, 
                                                buffer_size,
                                                nullptr, nullptr, &err);
        
        if (fpga_mapped_ptr == nullptr || err != CL_SUCCESS) {
            std::cerr << "Error: Failed to map P2P buffer, error " << err << "\n";
            return 1;
        }
        
        std::cout << "  P2P mapped address: " << fpga_mapped_ptr << "\n\n";

        // ========================================================================
        // Step 5: Register P2P buffer with ROCm
        // ========================================================================
        std::cout << "[5/6] Registering P2P buffer with ROCm...\n";
        
        // Register the FPGA P2P memory with HIP so GPU can access it
        // Try with hipHostRegisterIoMemory for true P2P
        hipError_t reg_result = hipHostRegister(fpga_mapped_ptr, buffer_size, 
                                                hipHostRegisterMapped | hipHostRegisterIoMemory);
        
        true_p2p = (reg_result == hipSuccess);
        
        if (!true_p2p) {
            std::cout << "  Note: hipHostRegisterIoMemory failed: " << hipGetErrorString(reg_result) << "\n";
            std::cout << "  Trying standard registration...\n";
            
            // Fall back to regular registration
            reg_result = hipHostRegister(fpga_mapped_ptr, buffer_size, hipHostRegisterMapped);
            if (reg_result != hipSuccess) {
                std::cerr << "Error: hipHostRegister failed: " << hipGetErrorString(reg_result) << "\n";
                std::cerr << "  This may be due to:\n";
                std::cerr << "  - IOMMU blocking P2P access\n";
                std::cerr << "  - Devices on different PCIe root complexes\n";
                std::cerr << "  - Insufficient permissions\n";
                return 1;
            }
            std::cout << "  Using CPU-mediated transfer (not true P2P)\n";
        } else {
            std::cout << "  ✓ True P2P enabled (hipHostRegisterIoMemory)\n";
        }
        
        // Get the device pointer that GPU kernels will use
        HIP_CHECK(hipHostGetDevicePointer(&d_fpga_ptr, fpga_mapped_ptr, 0));
        
        std::cout << "  GPU-accessible address: " << d_fpga_ptr << "\n\n";
    } else {
        // Non-P2P mode: use host staging buffer
        std::cout << "[5/6] Setting up non-P2P transfer path...\n";
        std::cout << "  Using system DRAM staging buffer\n";
        std::cout << "  Data path: GPU <-> System DRAM <-> FPGA\n\n";
    }

    // ========================================================================
    // Step 6: Run P2P Transfer Tests
    // ========================================================================
    
    // Use HIP events for accurate GPU timing
    hipEvent_t start_event, stop_event;
    HIP_CHECK(hipEventCreate(&start_event));
    HIP_CHECK(hipEventCreate(&stop_event));
    
    const int block_size = 256;
    const int num_blocks = std::min((size_t)65535, (num_vec4 + block_size - 1) / block_size);
    
    // ------------------------------------------------------------------------
    // Test A: GPU -> FPGA (Write to FPGA HBM)
    // ------------------------------------------------------------------------
    std::cout << "[6/6] Running P2P Transfer Tests...\n\n";
    std::cout << "--- Test A: GPU -> FPGA Transfer ---\n";
    
    // First, initialize GPU buffer with a test pattern
    const uint32_t gpu_pattern = 0xCAFE0000;
    hipLaunchKernelGGL(gpu_write_pattern, dim3(num_blocks), dim3(block_size), 0, 0,
                      d_gpu_buffer, num_words, gpu_pattern);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Warmup and benchmark based on transfer mode
    std::vector<float> write_times;
    
    if (cfg.use_p2p) {
        // P2P mode: direct GPU kernel writes to FPGA memory
        // Warmup
        for (int i = 0; i < cfg.warmup_iterations; i++) {
            hipLaunchKernelGGL(gpu_memcpy_vec4, dim3(num_blocks), dim3(block_size), 0, 0,
                              (const float4*)d_gpu_buffer, (float4*)d_fpga_ptr, num_vec4);
            HIP_CHECK(hipDeviceSynchronize());
        }
        
        // Benchmark GPU -> FPGA
        for (int iter = 0; iter < cfg.num_iterations; iter++) {
            HIP_CHECK(hipEventRecord(start_event));
            
            hipLaunchKernelGGL(gpu_memcpy_vec4, dim3(num_blocks), dim3(block_size), 0, 0,
                              (const float4*)d_gpu_buffer, (float4*)d_fpga_ptr, num_vec4);
            
            HIP_CHECK(hipEventRecord(stop_event));
            HIP_CHECK(hipEventSynchronize(stop_event));
            
            float ms;
            HIP_CHECK(hipEventElapsedTime(&ms, start_event, stop_event));
            write_times.push_back(ms);
            
            if (cfg.verbose) {
                double bw = (buffer_size / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
                std::cout << "  Iter " << iter << ": " << std::fixed << std::setprecision(5) 
                          << ms << " ms, " << bw << " GB/s\n";
            }
        }
    } else {
        // Non-P2P mode: GPU -> Host DRAM -> FPGA
        // Warmup
        for (int i = 0; i < cfg.warmup_iterations; i++) {
            // GPU -> Host DRAM
            hipLaunchKernelGGL(gpu_memcpy_vec4, dim3(num_blocks), dim3(block_size), 0, 0,
                              (const float4*)d_gpu_buffer, (float4*)d_staging_ptr, num_vec4);
            HIP_CHECK(hipDeviceSynchronize());
            // Host DRAM -> FPGA
            OCL_CHECK(err, err = queue.enqueueWriteBuffer(fpga_buffer, CL_TRUE, 0, buffer_size, h_staging_buffer));
        }
        
        // Benchmark GPU -> FPGA (via DRAM)
        for (int iter = 0; iter < cfg.num_iterations; iter++) {
            HIP_CHECK(hipEventRecord(start_event));
            
            // GPU -> Host DRAM
            hipLaunchKernelGGL(gpu_memcpy_vec4, dim3(num_blocks), dim3(block_size), 0, 0,
                              (const float4*)d_gpu_buffer, (float4*)d_staging_ptr, num_vec4);
            
            HIP_CHECK(hipEventRecord(stop_event));
            HIP_CHECK(hipEventSynchronize(stop_event));
            
            // Host DRAM -> FPGA (not timed with GPU events, use wall clock)
            auto fpga_start = std::chrono::high_resolution_clock::now();
            OCL_CHECK(err, err = queue.enqueueWriteBuffer(fpga_buffer, CL_TRUE, 0, buffer_size, h_staging_buffer));
            auto fpga_end = std::chrono::high_resolution_clock::now();
            
            float gpu_ms;
            HIP_CHECK(hipEventElapsedTime(&gpu_ms, start_event, stop_event));
            float fpga_ms = std::chrono::duration<float, std::milli>(fpga_end - fpga_start).count();
            float total_ms = gpu_ms + fpga_ms;
            write_times.push_back(total_ms);
            
            if (cfg.verbose) {
                double bw = (buffer_size / (1024.0 * 1024.0 * 1024.0)) / (total_ms / 1000.0);
                std::cout << "  Iter " << iter << ": " << std::fixed << std::setprecision(5) 
                          << total_ms << " ms (GPU: " << gpu_ms << " + FPGA: " << fpga_ms << "), " << bw << " GB/s\n";
            }
        }
    }
    
    // Calculate average
    float avg_write = 0;
    for (auto t : write_times) avg_write += t;
    avg_write /= write_times.size();
    double write_bw = (buffer_size / (1024.0 * 1024.0 * 1024.0)) / (avg_write / 1000.0);
    
    std::cout << "  Average: " << std::fixed << std::setprecision(5) 
              << avg_write << " ms, " << write_bw << " GB/s\n";
    
    // Verify data on CPU (read from FPGA buffer)
    if (cfg.verify) {
        std::cout << "  Verifying data...\n";
        
        uint32_t* host_verify;
        if (cfg.use_p2p) {
            host_verify = (uint32_t*)fpga_mapped_ptr;
        } else {
            // Read back from FPGA to staging buffer for verification
            OCL_CHECK(err, err = queue.enqueueReadBuffer(fpga_buffer, CL_TRUE, 0, buffer_size, h_staging_buffer));
            host_verify = h_staging_buffer;
        }
        
        int verify_errors = 0;
        for (size_t i = 0; i < std::min((size_t)1000, num_words); i++) {
            uint32_t expected = gpu_pattern + (i / 16);  // Match GPU pattern (base + block index)
            if (host_verify[i] != expected) {
                verify_errors++;
                if (verify_errors <= 5) {
                    std::cout << "    Error at " << i << ": expected 0x" << std::hex 
                              << expected << ", got 0x" << host_verify[i] 
                              << std::dec << "\n";
                }
            }
        }
        if (verify_errors == 0) {
            std::cout << "  ✓ Verification passed\n";
        } else {
            std::cout << "  ✗ Verification failed with " << verify_errors << "+ errors\n";
        }
    }
    std::cout << "\n";
    
    // ------------------------------------------------------------------------
    // Test B: FPGA -> GPU (Read from FPGA HBM)
    // ------------------------------------------------------------------------
    std::cout << "--- Test B: FPGA -> GPU Transfer ---\n";
    
    // Initialize FPGA buffer with a different pattern
    const uint32_t fpga_pattern = 0xDEAD0000;
    if (cfg.use_p2p) {
        uint32_t* fpga_data = (uint32_t*)fpga_mapped_ptr;
        for (size_t i = 0; i < num_words; i++) {
            fpga_data[i] = fpga_pattern + (i / 16);
        }
    } else {
        // Initialize in staging buffer and write to FPGA
        for (size_t i = 0; i < num_words; i++) {
            h_staging_buffer[i] = fpga_pattern + (i / 16);
        }
        OCL_CHECK(err, err = queue.enqueueWriteBuffer(fpga_buffer, CL_TRUE, 0, buffer_size, h_staging_buffer));
    }
    
    // Warmup and benchmark based on transfer mode
    std::vector<float> read_times;
    
    if (cfg.use_p2p) {
        // P2P mode: direct GPU kernel reads from FPGA memory
        // Warmup
        for (int i = 0; i < cfg.warmup_iterations; i++) {
            hipLaunchKernelGGL(gpu_memcpy_vec4, dim3(num_blocks), dim3(block_size), 0, 0,
                              (const float4*)d_fpga_ptr, (float4*)d_gpu_buffer, num_vec4);
            HIP_CHECK(hipDeviceSynchronize());
        }
        
        // Benchmark FPGA -> GPU
        for (int iter = 0; iter < cfg.num_iterations; iter++) {
            HIP_CHECK(hipEventRecord(start_event));
            
            hipLaunchKernelGGL(gpu_memcpy_vec4, dim3(num_blocks), dim3(block_size), 0, 0,
                              (const float4*)d_fpga_ptr, (float4*)d_gpu_buffer, num_vec4);
            
            HIP_CHECK(hipEventRecord(stop_event));
            HIP_CHECK(hipEventSynchronize(stop_event));
            
            float ms;
            HIP_CHECK(hipEventElapsedTime(&ms, start_event, stop_event));
            read_times.push_back(ms);
            
            if (cfg.verbose) {
                double bw = (buffer_size / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
                std::cout << "  Iter " << iter << ": " << std::fixed << std::setprecision(5) 
                          << ms << " ms, " << bw << " GB/s\n";
            }
        }
    } else {
        // Non-P2P mode: FPGA -> Host DRAM -> GPU
        // Warmup
        for (int i = 0; i < cfg.warmup_iterations; i++) {
            // FPGA -> Host DRAM
            OCL_CHECK(err, err = queue.enqueueReadBuffer(fpga_buffer, CL_TRUE, 0, buffer_size, h_staging_buffer));
            // Host DRAM -> GPU
            hipLaunchKernelGGL(gpu_memcpy_vec4, dim3(num_blocks), dim3(block_size), 0, 0,
                              (const float4*)d_staging_ptr, (float4*)d_gpu_buffer, num_vec4);
            HIP_CHECK(hipDeviceSynchronize());
        }
        
        // Benchmark FPGA -> GPU (via DRAM)
        for (int iter = 0; iter < cfg.num_iterations; iter++) {
            // FPGA -> Host DRAM (timed with wall clock)
            auto fpga_start = std::chrono::high_resolution_clock::now();
            OCL_CHECK(err, err = queue.enqueueReadBuffer(fpga_buffer, CL_TRUE, 0, buffer_size, h_staging_buffer));
            auto fpga_end = std::chrono::high_resolution_clock::now();
            
            HIP_CHECK(hipEventRecord(start_event));
            
            // Host DRAM -> GPU
            hipLaunchKernelGGL(gpu_memcpy_vec4, dim3(num_blocks), dim3(block_size), 0, 0,
                              (const float4*)d_staging_ptr, (float4*)d_gpu_buffer, num_vec4);
            
            HIP_CHECK(hipEventRecord(stop_event));
            HIP_CHECK(hipEventSynchronize(stop_event));
            auto process_end = std::chrono::high_resolution_clock::now();
            
            float gpu_ms;
            HIP_CHECK(hipEventElapsedTime(&gpu_ms, start_event, stop_event));
            float fpga_ms = std::chrono::duration<float, std::milli>(fpga_end - fpga_start).count();
            float total_ms = std::chrono::duration<float, std::milli>(process_end - fpga_start).count();
            read_times.push_back(total_ms);
            
            if (cfg.verbose) {
                double bw = (buffer_size / (1024.0 * 1024.0 * 1024.0)) / (total_ms / 1000.0);
                std::cout << "  Iter " << iter << ": " << std::fixed << std::setprecision(5) 
                          << total_ms << " ms (FPGA: " << fpga_ms << " + GPU: " << gpu_ms << "), " << bw << " GB/s\n";
            }
        }
    }
    
    // Calculate average
    float avg_read = 0;
    for (auto t : read_times) avg_read += t;
    avg_read /= read_times.size();
    double read_bw = (buffer_size / (1024.0 * 1024.0 * 1024.0)) / (avg_read / 1000.0);
    
    std::cout << "  Average: " << std::fixed << std::setprecision(5) 
              << avg_read << " ms, " << read_bw << " GB/s\n";
    
    // Verify data on GPU
    if (cfg.verify) {
        std::cout << "  Verifying data on GPU...\n";
        HIP_CHECK(hipMemset(d_error_count, 0, sizeof(uint32_t)));
        
        hipLaunchKernelGGL(gpu_read_verify, dim3(num_blocks), dim3(block_size), 0, 0,
                          d_gpu_buffer, num_words, fpga_pattern, d_error_count);
        
        uint32_t gpu_errors;
        HIP_CHECK(hipMemcpy(&gpu_errors, d_error_count, sizeof(uint32_t), hipMemcpyDeviceToHost));
        
        if (gpu_errors == 0) {
            std::cout << "  ✓ Verification passed\n";
        } else {
            std::cout << "  ✗ Verification failed with " << gpu_errors << " errors\n";
        }
    }
    std::cout << "\n";

    // ========================================================================
    // Optional: Run FPGA kernel (if it does something useful)
    // ========================================================================
    std::cout << "--- Running FPGA kernel ---\n";
    
    // Enqueue kernel execution
    OCL_CHECK(err, err = queue.enqueueTask(kernel));
    queue.finish();
    
    std::cout << "  FPGA kernel executed\n\n";

    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "========================================\n";
    std::cout << "Summary\n";
    std::cout << "========================================\n";
    std::cout << "Buffer size:        " << cfg.buffer_size_kb << " KB (" 
              << (cfg.buffer_size_kb / 1024.0) << " MB)\n";
    if (cfg.use_p2p) {
        std::cout << "Transfer mode:      " << (true_p2p ? "True P2P" : "P2P (CPU-mediated)") << "\n";
    } else {
        std::cout << "Transfer mode:      Non-P2P (via system DRAM)\n";
    }
    std::cout << "GPU -> FPGA:        " << std::fixed << std::setprecision(5) 
              << write_bw << " GB/s (" << avg_write << " ms)\n";
    std::cout << "FPGA -> GPU:        " << std::fixed << std::setprecision(5) 
              << read_bw << " GB/s (" << avg_read << " ms)\n";
    std::cout << "========================================\n";

    // ========================================================================
    // Cleanup
    // ========================================================================
    HIP_CHECK(hipEventDestroy(start_event));
    HIP_CHECK(hipEventDestroy(stop_event));
    
    if (cfg.use_p2p) {
        HIP_CHECK(hipHostUnregister(fpga_mapped_ptr));
        // Unmap P2P buffer
        queue.enqueueUnmapMemObject(fpga_buffer, fpga_mapped_ptr);
        queue.finish();
    } else {
        // Free staging buffer
        HIP_CHECK(hipHostFree(h_staging_buffer));
    }
    
    HIP_CHECK(hipFree(d_gpu_buffer));
    HIP_CHECK(hipFree(d_error_count));
    
    std::cout << "\nP2P transfer test completed successfully!\n";
    
    return 0;
}
