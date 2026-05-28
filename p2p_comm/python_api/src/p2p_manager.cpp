/**
 * P2P Manager Implementation
 */

#include "p2p_manager.hpp"
#include <iostream>
#include <chrono>
#include <cstring>

namespace heteromem {

// ============================================================================
// P2P Buffer Implementation
// ============================================================================

P2PBuffer::P2PBuffer(xrt::device& fpga_device, size_t count, DataType dtype, int mem_group)
    : count_(count)
    , dtype_(dtype)
    , gpu_registered_(false)
{
    size_t element_size = 0;
    switch (dtype) {
        case DataType::UINT32: element_size = sizeof(uint32_t); break;
        case DataType::INT32: element_size = sizeof(int32_t); break;
        case DataType::FLOAT32: element_size = sizeof(float); break;
        case DataType::FLOAT64: element_size = sizeof(double); break;
    }
    
    size_bytes_ = count * element_size;
    
    bool allocated = false;

    // If a specific memory group was requested, try only that group.
    // This is important when the FPGA kernel argument is bound to a
    // particular HBM bank — the buffer must match.
    if (mem_group >= 0) {
        unsigned mg = static_cast<unsigned>(mem_group);
        try {
            buffer_ = xrt::bo(fpga_device, size_bytes_, xrt::bo::flags::p2p, mg);
            is_p2p_ = true;
            allocated = true;
        } catch (...) {
            try {
                buffer_ = xrt::bo(fpga_device, size_bytes_, xrt::bo::flags::normal, mg);
                is_p2p_ = false;
                allocated = true;
            } catch (...) {
                throw P2PException("Failed to allocate FPGA buffer in memory group " +
                                   std::to_string(mem_group));
            }
        }
    } else {
        // Probe all 32 groups because the Python API is platform-agnostic
        // and doesn't know a priori which HBM banks are valid on each board.
        // The exception-based probing is only done once at allocation time.
        constexpr unsigned kMaxMemGroups = 32;
        
        // First, try to allocate a P2P BO across a range of memory groups
        for (unsigned mg = 0; mg < kMaxMemGroups && !allocated; ++mg) {
            try {
                buffer_ = xrt::bo(fpga_device, size_bytes_, xrt::bo::flags::p2p, mg);
                is_p2p_ = true;
                allocated = true;
            } catch (...) {
                // Try next memory group
            }
        }
        
        // If P2P allocation failed for all groups, fall back to normal BOs
        if (!allocated) {
            for (unsigned mg = 0; mg < kMaxMemGroups && !allocated; ++mg) {
                try {
                    buffer_ = xrt::bo(fpga_device, size_bytes_, xrt::bo::flags::normal, mg);
                    is_p2p_ = false;
                    allocated = true;
                } catch (...) {
                    // Try next memory group
                }
            }
        }
        
        if (!allocated) {
            throw P2PException("Failed to allocate FPGA buffer in any memory group");
        }
    }
    
    host_ptr_ = buffer_.map();
    if (!host_ptr_) {
        throw P2PException("Failed to map buffer to host");
    }
}

P2PBuffer::~P2PBuffer() {
    if (gpu_registered_) {
        unregister_from_gpu();
    }
}

void P2PBuffer::write(const void* data, size_t size) {
    if (size > size_bytes_) {
        throw P2PException("Write size exceeds buffer size");
    }
    std::memcpy(host_ptr_, data, size);
}

void P2PBuffer::read(void* data, size_t size) const {
    if (size > size_bytes_) {
        throw P2PException("Read size exceeds buffer size");
    }
    std::memcpy(data, host_ptr_, size);
}

void P2PBuffer::sync_to_device() {
    buffer_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

void P2PBuffer::sync_to_host() {
    buffer_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
}

void P2PBuffer::register_with_gpu() {
    if (gpu_registered_) return;
    
    hipError_t err = hipHostRegister(
        host_ptr_, 
        size_bytes_,
        hipHostRegisterMapped | hipHostRegisterIoMemory
    );
    
    if (err != hipSuccess) {
        // Try without IoMemory flag — this means we don't have true P2P;
        // transfers will go through the host shadow and require explicit syncs.
        err = hipHostRegister(host_ptr_, size_bytes_, hipHostRegisterMapped);
        if (err != hipSuccess) {
            throw GPUException(std::string("Failed to register buffer with GPU: ") + 
                             hipGetErrorString(err) + 
                             " (tried both hipHostRegisterIoMemory and hipHostRegisterMapped)");
        }
        is_p2p_ = false;
    }
    
    err = hipHostGetDevicePointer(&device_ptr_, host_ptr_, 0);
    if (err != hipSuccess) {
        hipHostUnregister(host_ptr_);
        throw GPUException("Failed to get device pointer");
    }
    
    gpu_registered_ = true;
}

void P2PBuffer::unregister_from_gpu() {
    if (!gpu_registered_) return;
    
    hipHostUnregister(host_ptr_);
    gpu_registered_ = false;
    device_ptr_ = nullptr;
}

size_t P2PBuffer::get_dtype_size(DataType dtype) const {
    switch (dtype) {
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::FLOAT32:
            return 4;
        case DataType::FLOAT64:
            return 8;
        default:
            throw P2PException("Unknown data type");
    }
}

// ============================================================================
// FPGA Device Implementation
// ============================================================================

FPGADevice::FPGADevice(const std::string& bdf)
    : bdf_(bdf)
    , xclbin_loaded_(false)
{
    // Find device by BDF - try up to 16 devices
    for (unsigned int i = 0; i < 16; i++) {
        try {
            xrt::device dev(i);
            std::string dev_bdf = dev.get_info<xrt::info::device::bdf>();
            if (dev_bdf == bdf) {
                device_ = std::move(dev);
                device_index_ = i;
                return;
            }
        } catch (...) {
            continue;
        }
    }
    throw FPGAException("Device with BDF " + bdf + " not found");
}

FPGADevice::FPGADevice(unsigned int device_index)
    : device_(device_index)
    , device_index_(device_index)
    , xclbin_loaded_(false)
{
    bdf_ = device_.get_info<xrt::info::device::bdf>();
}

FPGADevice FPGADevice::auto_detect() {
    unsigned int idx = find_u55c_device();
    return FPGADevice(idx);
}

std::string FPGADevice::name() const {
    return device_.get_info<xrt::info::device::name>();
}

std::string FPGADevice::bdf() const {
    return bdf_;
}

void FPGADevice::load_xclbin(const std::string& xclbin_path) {
    xclbin_uuid_ = device_.load_xclbin(xclbin_path);
    xclbin_loaded_ = true;
}

xrt::kernel FPGADevice::get_kernel(const std::string& kernel_name) {
    if (!xclbin_loaded_) {
        throw FPGAException("No xclbin loaded");
    }
    return xrt::kernel(device_, xclbin_uuid_, kernel_name);
}

std::shared_ptr<P2PBuffer> FPGADevice::create_buffer(size_t count, P2PBuffer::DataType dtype, int mem_group) {
    return std::make_shared<P2PBuffer>(device_, count, dtype, mem_group);
}

void FPGADevice::generate_indices(
    P2PBuffer& buffer,
    uint32_t count,
    const std::string& mode,
    uint32_t stride,
    uint32_t start_offset,
    const std::string& kernel_name
) {
    if (!xclbin_loaded_) {
        throw FPGAException("No xclbin loaded - cannot run kernel");
    }
    
    // Validate buffer size
    if (count > buffer.count()) {
        throw FPGAException("Requested count (" + std::to_string(count) + 
                          ") exceeds buffer capacity (" + std::to_string(buffer.count()) + ")");
    }
    
    // Validate buffer dtype (kernel writes uint32_t)
    if (buffer.dtype() != P2PBuffer::DataType::UINT32) {
        std::string dtype_str;
        switch (buffer.dtype()) {
            case P2PBuffer::DataType::UINT32: dtype_str = "uint32"; break;
            case P2PBuffer::DataType::INT32: dtype_str = "int32"; break;
            case P2PBuffer::DataType::FLOAT32: dtype_str = "float32"; break;
            case P2PBuffer::DataType::FLOAT64: dtype_str = "float64"; break;
        }
        throw FPGAException("Buffer dtype must be uint32 for index generation, got: " + dtype_str);
    }
    
    // Get kernel
    xrt::kernel kernel = get_kernel(kernel_name);
    xrt::run run(kernel);
    
    // Determine mode value
    uint32_t mode_val = 0;
    if (mode == "sequential") mode_val = 0;
    else if (mode == "strided") mode_val = 1;
    else if (mode == "pattern") mode_val = 2;
    else throw FPGAException("Unknown mode: " + mode);
    
    // Set arguments
    run.set_arg(0, buffer.get_xrt_bo());
    run.set_arg(1, count);
    run.set_arg(2, mode_val);
    run.set_arg(3, stride);
    run.set_arg(4, start_offset);
    
    // Execute
    run.start();
    run.wait();
    
    // Ensure buffer contents are synchronized back to host memory
    buffer.sync_to_host();
}

unsigned int FPGADevice::find_u55c_device() {
    // Try up to 16 devices
    for (unsigned int i = 0; i < 16; i++) {
        try {
            xrt::device dev(i);
            std::string name = dev.get_info<xrt::info::device::name>();
            if (name.find("u55c") != std::string::npos) {
                return i;
            }
        } catch (...) {
            continue;
        }
    }
    throw FPGAException("No U55C device found");
}

// ============================================================================
// GPU Device Implementation
// ============================================================================

GPUDevice::GPUDevice(int device_id)
    : device_id_(device_id)
{
    hipError_t err = hipSetDevice(device_id);
    check_hip_error(err, "Failed to set GPU device");
    
    err = hipGetDeviceProperties(&props_, device_id);
    check_hip_error(err, "Failed to get device properties");
}

GPUDevice::~GPUDevice() {
    // Cleanup if needed
}

GPUDevice GPUDevice::auto_detect() {
    return GPUDevice(0);
}

std::string GPUDevice::name() const {
    return std::string(props_.name);
}

std::string GPUDevice::pcie_id() const {
    // Format as domain:bus:device.function (standard BDF format)
    // HIP doesn't expose function, so we assume .0
    char bdf[32];
    snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.0", 
             props_.pciDomainID, props_.pciBusID, props_.pciDeviceID);
    return std::string(bdf);
}

void GPUDevice::read_buffer(const P2PBuffer& buffer, void* output, size_t size) {
    if (!buffer.is_gpu_registered()) {
        throw GPUException("Buffer not registered with GPU");
    }
    
    // Validate size bounds
    if (size > buffer.size()) {
        throw GPUException("Read size (" + std::to_string(size) + 
                          ") exceeds buffer capacity (" + std::to_string(buffer.size()) + ")");
    }
    
    hipError_t err = hipMemcpy(output, buffer.device_ptr(), size, hipMemcpyDeviceToHost);
    check_hip_error(err, "Failed to read from P2P buffer");
}

void GPUDevice::write_buffer(P2PBuffer& buffer, const void* input, size_t size) {
    if (!buffer.is_gpu_registered()) {
        throw GPUException("Buffer not registered with GPU");
    }
    
    // Validate size bounds
    if (size > buffer.size()) {
        throw GPUException("Write size (" + std::to_string(size) + 
                          ") exceeds buffer capacity (" + std::to_string(buffer.size()) + ")");
    }
    
    hipError_t err = hipMemcpy(buffer.device_ptr(), input, size, hipMemcpyHostToDevice);
    check_hip_error(err, "Failed to write to P2P buffer");
}

template<typename T>
void GPUDevice::memcpy_write_to_fpga(P2PBuffer& buffer, const T* gpu_src_ptr, uint32_t count) {
    if (!buffer.is_gpu_registered()) {
        throw GPUException("Buffer not registered with GPU");
    }
    if (count > buffer.count()) {
        throw GPUException("Write count (" + std::to_string(count) + 
        ") exceeds buffer capacity (" + std::to_string(buffer.count()) + ")");
    }
    size_t bytes = count * sizeof(T);
    hipError_t err = hipMemcpy(buffer.device_ptr(), gpu_src_ptr, bytes, hipMemcpyDeviceToDevice);
    check_hip_error(err, "memcpy_write_to_fpga: hipMemcpy failed");
    // For non-P2P buffers the hipMemcpy lands in the host shadow;
    // push to FPGA HBM so the data is visible on the device side.
    if (!buffer.is_p2p_buffer()) {
        buffer.sync_to_device();
    }
}

// Explicit template instantiations
template void GPUDevice::memcpy_write_to_fpga<uint32_t>(P2PBuffer&, const uint32_t*, uint32_t);
template void GPUDevice::memcpy_write_to_fpga<float>(P2PBuffer&, const float*, uint32_t);

void GPUDevice::synchronize() {
    hipError_t err = hipDeviceSynchronize();
    check_hip_error(err, "GPU synchronization failed");
}

void GPUDevice::check_hip_error(hipError_t error, const std::string& msg) {
    if (error != hipSuccess) {
        throw GPUException(msg + ": " + hipGetErrorString(error));
    }
}

// ============================================================================
// P2P Manager Implementation
// ============================================================================

P2PManager::P2PManager(FPGADevice& fpga, GPUDevice& gpu)
    : fpga_(fpga)
    , gpu_(gpu)
{
}

void P2PManager::transfer_fpga_to_gpu(P2PBuffer& buffer) {
    buffer.sync_to_host();
    // Data is now accessible to GPU via P2P
}

void P2PManager::transfer_gpu_to_fpga(P2PBuffer& buffer) {
    gpu_.synchronize();
    buffer.sync_to_device();
}

void P2PManager::memcpy_transfer_gpu_to_fpga(P2PBuffer& buffer, const void* gpu_src_ptr, uint32_t count) {
    // Dispatch to the correct template instantiation based on buffer dtype.
    // memcpy_write_to_fpga already handles non-P2P sync internally.
    switch (buffer.dtype()) {
        case P2PBuffer::DataType::FLOAT32:
            gpu_.memcpy_write_to_fpga(buffer, static_cast<const float*>(gpu_src_ptr), count);
            break;
        case P2PBuffer::DataType::UINT32:
        case P2PBuffer::DataType::INT32:
        default:
            gpu_.memcpy_write_to_fpga(buffer, static_cast<const uint32_t*>(gpu_src_ptr), count);
            break;
    }
}
 
P2PManager::BenchmarkResult P2PManager::benchmark_gpu_write(
    P2PBuffer& buffer,
    const void* gpu_src_ptr,
    uint32_t count,
    int iterations
) {
    if (iterations <= 0) {
        throw P2PException("benchmark_gpu_write: iterations must be > 0");
    }
    BenchmarkResult result;
    size_t elem_size = (buffer.count() > 0) ? buffer.size() / buffer.count() : sizeof(uint32_t);
    size_t bytes_per_iter = count * elem_size;
    result.bytes_transferred = bytes_per_iter;
    for (int i = 0; i < 5; i++) {
        memcpy_transfer_gpu_to_fpga(buffer, gpu_src_ptr, count);
    }
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        memcpy_transfer_gpu_to_fpga(buffer, gpu_src_ptr, count);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    result.latency_ms    = duration.count() / 1000.0 / iterations;
    result.bandwidth_gbps = (double(bytes_per_iter) * iterations) /
                            (duration.count() / 1e6) / 1e9;
    return result;
}

P2PManager::BenchmarkResult P2PManager::benchmark_transfer(
    P2PBuffer& buffer,
    bool fpga_to_gpu,
    int iterations
) {
    BenchmarkResult result;
    result.bytes_transferred = buffer.size();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        if (fpga_to_gpu) {
            transfer_fpga_to_gpu(buffer);
        } else {
            transfer_gpu_to_fpga(buffer);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    result.latency_ms = duration.count() / 1000.0 / iterations;
    result.bandwidth_gbps = (result.bytes_transferred * iterations) / 
                           (duration.count() / 1e6) / 1e9;
    
    return result;
}

// ============================================================================
// Utility Functions
// ============================================================================

namespace utils {

std::vector<std::string> list_fpga_devices() {
    std::vector<std::string> devices;
    
    // Try up to 16 devices
    for (unsigned int i = 0; i < 16; i++) {
        try {
            xrt::device dev(i);
            std::string bdf = dev.get_info<xrt::info::device::bdf>();
            std::string name = dev.get_info<xrt::info::device::name>();
            devices.push_back(bdf + " (" + name + ")");
        } catch (...) {
            // Device may not exist or have permission errors, continue searching
            continue;
        }
    }
    
    return devices;
}

std::vector<int> list_gpu_devices() {
    std::vector<int> devices;
    int count;
    
    if (hipGetDeviceCount(&count) == hipSuccess) {
        for (int i = 0; i < count; i++) {
            devices.push_back(i);
        }
    }
    
    return devices;
}

bool check_p2p_support(const std::string& fpga_bdf) {
    // This would need to parse xbutil output or check sysfs
    // For now, return true and let runtime handle it
    return true;
}

Timer::Timer() {}

void Timer::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
}

void Timer::stop() {
    end_time_ = std::chrono::high_resolution_clock::now();
}

double Timer::elapsed_ms() const {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time_ - start_time_
    );
    return duration.count() / 1000.0;
}

} // namespace utils

} // namespace heteromem
