/**
 * P2P Manager - C++ Wrapper Classes for Python Bindings
 * 
 * Provides high-level C++ interface for FPGA-GPU P2P operations
 * that will be exposed to Python via pybind11.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <chrono>

// XRT includes
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

// HIP includes
#include <hip/hip_runtime.h>

namespace heteromem {

// ============================================================================
// Exception Classes
// ============================================================================

class P2PException : public std::runtime_error {
public:
    explicit P2PException(const std::string& msg) : std::runtime_error(msg) {}
};

class FPGAException : public P2PException {
public:
    explicit FPGAException(const std::string& msg) : P2PException("FPGA Error: " + msg) {}
};

class GPUException : public P2PException {
public:
    explicit GPUException(const std::string& msg) : P2PException("GPU Error: " + msg) {}
};

// ============================================================================
// P2P Buffer
// ============================================================================

class P2PBuffer {
public:
    enum class DataType {
        UINT32,
        INT32,
        FLOAT32,
        FLOAT64
    };

    P2PBuffer(xrt::device& fpga_device, size_t count, DataType dtype, int mem_group = -1);
    ~P2PBuffer();

    // Disable copy, allow move
    P2PBuffer(const P2PBuffer&) = delete;
    P2PBuffer& operator=(const P2PBuffer&) = delete;
    P2PBuffer(P2PBuffer&&) = default;
    P2PBuffer& operator=(P2PBuffer&&) = default;

    // Properties
    size_t size() const { return size_bytes_; }
    size_t count() const { return count_; }
    DataType dtype() const { return dtype_; }
    void* host_ptr() const { return host_ptr_; }
    void* device_ptr() const { return device_ptr_; }
    
    // Data access
    void write(const void* data, size_t size);
    void read(void* data, size_t size) const;
    
    // Sync operations
    void sync_to_device();
    void sync_to_host();
    
    // XRT buffer access (for kernel arguments)
    xrt::bo& get_xrt_bo() { return buffer_; }
    
    // Register with GPU
    void register_with_gpu();
    void unregister_from_gpu();
    bool is_gpu_registered() const { return gpu_registered_; }
    bool is_p2p_buffer() const { return is_p2p_; }

private:
    xrt::bo buffer_;
    void* host_ptr_ = nullptr;
    void* device_ptr_ = nullptr;
    size_t size_bytes_;
    size_t count_;
    DataType dtype_;
    bool gpu_registered_;
    bool is_p2p_ = false;
    
    size_t get_dtype_size(DataType dtype) const;
};

// ============================================================================
// FPGA Device
// ============================================================================

class FPGADevice {
public:
    // Constructors
    explicit FPGADevice(const std::string& bdf);
    explicit FPGADevice(unsigned int device_index);
    static FPGADevice auto_detect(); // Find first U55C
    
    ~FPGADevice() = default;

    // Device info
    std::string name() const;
    std::string bdf() const;
    unsigned int device_index() const { return device_index_; }
    
    // Bitstream management
    void load_xclbin(const std::string& xclbin_path);
    bool is_xclbin_loaded() const { return xclbin_loaded_; }
    
    // Kernel access
    xrt::kernel get_kernel(const std::string& kernel_name);
    
    // Buffer creation
    std::shared_ptr<P2PBuffer> create_buffer(size_t count, P2PBuffer::DataType dtype, int mem_group = -1);
    
    // High-level operations
    void generate_indices(
        P2PBuffer& buffer,
        uint32_t count,
        const std::string& mode,
        uint32_t stride = 1,
        uint32_t start_offset = 0,
        const std::string& kernel_name = "fpga_index_generator"
    );
    
    // Direct device access
    xrt::device& get_device() { return device_; }

private:
    xrt::device device_;
    xrt::uuid xclbin_uuid_;
    unsigned int device_index_;
    std::string bdf_;
    bool xclbin_loaded_;
    
    static unsigned int find_u55c_device();
};

// ============================================================================
// GPU Device
// ============================================================================

class GPUDevice {
public:
    // Constructors
    explicit GPUDevice(int device_id = 0);
    static GPUDevice auto_detect(); // Use default GPU
    
    ~GPUDevice();

    // Device info
    std::string name() const;
    int device_id() const { return device_id_; }
    std::string pcie_id() const;
    
    // Memory operations
    void read_buffer(const P2PBuffer& buffer, void* output, size_t size);
    void write_buffer(P2PBuffer& buffer, const void* input, size_t size);
    template<typename T>
    void memcpy_write_to_fpga(P2PBuffer& buffer, const T* gpu_src_ptr, uint32_t count);
    
    // SpMV operations (will be implemented with GPU kernels)
    void spmv_csr(
        const float* values,
        const uint32_t* col_indices,
        const uint32_t* row_ptrs,
        const float* x,
        float* y,
        const uint32_t* row_indices,
        uint32_t num_rows
    );
    
    // Synchronization
    void synchronize();

private:
    int device_id_;
    hipDeviceProp_t props_;
    
    void check_hip_error(hipError_t error, const std::string& msg);
};

// ============================================================================
// P2P Manager - High-level interface
// ============================================================================

class P2PManager {
public:
    P2PManager(FPGADevice& fpga, GPUDevice& gpu);
    ~P2PManager() = default;

    // Transfer operations
    void transfer_fpga_to_gpu(P2PBuffer& buffer);
    void transfer_gpu_to_fpga(P2PBuffer& buffer);
    void memcpy_transfer_gpu_to_fpga(P2PBuffer& buffer, const void* gpu_src_ptr, uint32_t count);
    
    // Benchmark
    struct BenchmarkResult {
        double bandwidth_gbps;
        double latency_ms;
        size_t bytes_transferred;
    };
    
    BenchmarkResult benchmark_transfer(
        P2PBuffer& buffer,
        bool fpga_to_gpu,
        int iterations = 100
    );

    BenchmarkResult benchmark_gpu_write(P2PBuffer& buffer, const void* gpu_src_ptr, uint32_t count, int iterations = 100);


private:
    FPGADevice& fpga_;
    GPUDevice& gpu_;
};

// ============================================================================
// Utility Functions
// ============================================================================

namespace utils {
    // Device enumeration
    std::vector<std::string> list_fpga_devices();
    std::vector<int> list_gpu_devices();
    
    // P2P capability check
    bool check_p2p_support(const std::string& fpga_bdf);
    
    // Performance monitoring
    class Timer {
    public:
        Timer();
        void start();
        void stop();
        double elapsed_ms() const;
        
    private:
        std::chrono::high_resolution_clock::time_point start_time_;
        std::chrono::high_resolution_clock::time_point end_time_;
    };
}

} // namespace heteromem
