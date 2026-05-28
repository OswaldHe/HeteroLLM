#ifndef FPGA_UTIL_H
#define FPGA_UTIL_H

#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#include <CL/cl2.hpp>

// Xilinx OpenCL extensions for P2P
#include <CL/cl_ext_xilinx.h>

#define OCL_CHECK(err, call)                                                   \
    do {                                                                       \
        call;                                                                  \
        if (err != CL_SUCCESS) {                                               \
            std::cerr << "OpenCL Error: " << err << " at " << __FILE__         \
                      << ":" << __LINE__ << std::endl;                         \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/**
 * @brief FPGA resources container holding OpenCL context, queue, and kernel
 */
struct FPGAResources {
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel kernel;
};

/**
 * @brief Load a binary file (e.g., xclbin) into a byte vector
 * @param filename Path to the binary file
 * @return Vector of unsigned chars containing the file contents
 */
std::vector<unsigned char> load_binary_file(const std::string& filename);

/**
 * @brief Find an FPGA device matching the target BDF or any U55C device
 * @param target_bdf Target Bus:Device.Function string to match
 * @param verbose Enable verbose output
 * @return OpenCL device object for the found FPGA
 * @throws std::runtime_error if no Xilinx FPGA device is found
 */
cl::Device find_fpga_device(const std::string& target_bdf, bool verbose);

/**
 * @brief Setup OpenCL resources for FPGA execution
 * @param fpga_bdf FPGA Bus:Device.Function identifier
 * @param xclbin_path Path to the xclbin file
 * @param kernel_top Name of the kernel to create
 * @return FPGAResources struct containing context, queue, and kernel
 */
FPGAResources setup_opencl_resource(
    std::string& fpga_bdf,
    std::string& xclbin_path,
    std::string& kernel_top
);

/**
 * @brief Register a P2P (peer-to-peer) buffer for GPU-FPGA direct communication
 * @param resources FPGA resources containing context, queue, and kernel
 * @param arg_id Kernel argument index for the buffer
 * @param buffer_size Size of the P2P buffer in bytes
 * @return Pointer to the mapped P2P buffer
 */
void* register_p2p_buffer(FPGAResources& resources, int arg_id, size_t buffer_size);

#endif // FPGA_UTIL_H
