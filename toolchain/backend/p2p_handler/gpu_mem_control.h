#ifndef GPU_MEM_CONTROL_H
#define GPU_MEM_CONTROL_H
// HIP includes

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

#include <hip/hip_runtime.h>
#include <iomanip>
#include <cstdlib>

void* registerTargetIOMemBuffer(void* fpga_mapped_ptr, size_t buffer_size);
uint32_t* registerGPUBuffer(int gpu_id, size_t buffer_size);

// Usage
// void* fpga_p2p_ptr = registerTargetIOMemBuffer(fpga_mapped_ptr, buffer_size);
// uint32_t* d_gpu_buffer = registerGPUBuffer(gpu_id, buffer_size);
// hipLaunchKernelGGL(gpu_kernel_c, dim3(num_blocks), dim3(block_size), 0, 0,
//                              (const dtype*)d_gpu_buffer, (dtype*)fpga_p2p_ptr);
// HIP_CHECK(hipDeviceSynchronize());



#endif // GPU_MEM_CONTROL_H
