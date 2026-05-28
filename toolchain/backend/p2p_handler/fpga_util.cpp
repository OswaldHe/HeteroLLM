#include "fpga_util.h"

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
            if (device_name.find(target_bdf) != std::string::npos) {
                return device;
            } else if (verbose){
                std::clog << "  Cannot get device with BDF '" << target_bdf << "'.";
            }

            if(device_name.find("u55c") != std::string::npos ||
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

FPGAResources setup_opencl_resource(
    std::string& fpga_bdf,
    std::string& xclbin_path,
    std::string& kernel_top
) {
    cl::Device fpga_device;
    try {
        fpga_device = find_fpga_device(fpga_bdf, false);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::exit(1);
    }

    cl_int err;
    cl::Context context(fpga_device, nullptr, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to create OpenCL context, error " << err << "\n";
        std::exit(1);
    }
    
    cl::CommandQueue queue(context, fpga_device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to create command queue, error " << err << "\n";
        std::exit(1);
    }

    std::vector<unsigned char> xclbin_data = load_binary_file(xclbin_path);
    cl::Program::Binaries binaries{{xclbin_data.data(), xclbin_data.size()}};
    
    std::vector<cl::Device> devices{fpga_device};
    cl::Program program(context, devices, binaries, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to load xclbin, error " << err << "\n";
        std::exit(1);
    }
    std::cout << "  xclbin loaded successfully\n";
    
    // Create kernel
    cl::Kernel kernel(program, kernel_top.c_str(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to create kernel '" << kernel_top << "', error " << err << "\n";
        std::cerr << "  Hint: Use --kernel to specify the correct kernel name in your xclbin\n";
        std::exit(1);
    }
    std::cout << "  Kernel '" << kernel_top << "' created\n\n";

    return {context, queue, kernel};
}

void* register_p2p_buffer(FPGAResources& resources, int arg_id, size_t buffer_size) {
    cl_int err;
    cl_mem_ext_ptr_t p2p_ext = {0, nullptr, 0};
    p2p_ext.flags = XCL_MEM_EXT_P2P_BUFFER;
    
    cl::Buffer fpga_buffer = cl::Buffer(resources.context, 
                            CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX,
                            buffer_size, 
                            &p2p_ext, 
                            &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to create P2P buffer, error " << err << "\n";
        std::exit(1);
    }
    
    std::cout << "  P2P buffer created: " << buffer_size / 1024 << " KB\n";

    err = resources.kernel.setArg(arg_id, fpga_buffer);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to set kernel argument " << arg_id << ", error " << err << "\n";
        std::cerr << "The device does not support P2P buffer allocation." << std::endl;
        std::exit(1);
    }

    void* fpga_mapped_ptr = nullptr;
    fpga_mapped_ptr = resources.queue.enqueueMapBuffer(fpga_buffer, 
                                            CL_TRUE,
                                            CL_MAP_READ | CL_MAP_WRITE,
                                            0, 
                                            buffer_size,
                                            nullptr, nullptr, &err);
    
    if (fpga_mapped_ptr == nullptr || err != CL_SUCCESS) {
        std::cerr << "Error: Failed to map P2P buffer, error " << err << "\n";
        std::exit(1);
    }
    
    std::cout << "  P2P mapped address: " << fpga_mapped_ptr << "\n\n";

    return fpga_mapped_ptr;
}
