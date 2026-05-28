#ifndef HETEROMM_DEV_STEPS_UTIL_H_
#define HETEROMM_DEV_STEPS_UTIL_H_

#include <string>

namespace heteromm {
namespace step {

/**
 * @brief Enum representing the kernel execution target
 */
enum class KernelType {
    CPU,     // CPU/GPU kernel (standard compute)
    GPU,     // Explicitly GPU kernel
    FPGA,    // FPGA kernel
};

/**
 * @brief Convert KernelType to string for logging
 */
inline std::string kernel_type_to_string(KernelType type) {
    switch (type) {
        case KernelType::CPU: return "CPU";
        case KernelType::GPU: return "GPU";
        case KernelType::FPGA: return "FPGA";
        default: return "UNKNOWN";
    }
}

KernelType string_to_kernel_type(const std::string& device) {
    if (device == "cpu") {
        return KernelType::CPU;
    } else if (device == "gpu") {
        return KernelType::GPU;
    } else if (device == "fpga") {
        return KernelType::FPGA;
    }
    return KernelType::CPU;  // Default
}

//[TODO]: add communication handler

}  // namespace step
}  // namespace heteromm

#endif  // HETEROMM_DEV_STEPS_UTIL_H_