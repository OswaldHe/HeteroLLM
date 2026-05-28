#include "gpu_mem_control.h"
#include <cstdlib>
#include <iostream>

void* registerTargetIOMemBuffer(void* fpga_mapped_ptr, size_t buffer_size) {
    hipError_t reg_result = hipHostRegister(fpga_mapped_ptr, buffer_size, 
                                                hipHostRegisterMapped | hipHostRegisterIoMemory);
        
    bool true_p2p = (reg_result == hipSuccess);
    
    if (!true_p2p) {
        std::cout << "  Note: hipHostRegisterIoMemory failed: " << hipGetErrorString(reg_result) << "\n";
        std::cout << "  Trying standard registration...\n";
        
        // Fall back to regular registration
        reg_result = hipHostRegister(fpga_mapped_ptr, buffer_size, hipHostRegisterMapped);
        if (reg_result != hipSuccess) {
            std::cerr << "Error: hipHostRegister failed: " << hipGetErrorString(reg_result) << "\n";
            std::exit(1);
        }
        std::cout << "  Using CPU-mediated transfer (not true P2P)\n";
    } else {
        std::cout << "  True P2P enabled (hipHostRegisterIoMemory)\n";
    }

    void* d_fpga_ptr = nullptr;
    
    // Get the device pointer that GPU kernels will use
    HIP_CHECK(hipHostGetDevicePointer(&d_fpga_ptr, fpga_mapped_ptr, 0));
    
    return d_fpga_ptr;
}

uint32_t* registerGPUBuffer(int gpu_id, size_t buffer_size) {
    int gpu_count;
    HIP_CHECK(hipGetDeviceCount(&gpu_count));
    if (gpu_id >= gpu_count) {
        std::cerr << "Error: GPU " << gpu_id << " not found. Available: " << gpu_count << "\n";
        std::exit(1);
    }
    
    HIP_CHECK(hipSetDevice(gpu_id));
    
    hipDeviceProp_t gpu_props;
    HIP_CHECK(hipGetDeviceProperties(&gpu_props, gpu_id));
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
}
