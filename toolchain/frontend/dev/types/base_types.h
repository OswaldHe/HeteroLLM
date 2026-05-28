/**
 * @file base_types.h
 * @brief Base type definitions for data communication between pipeline steps
 * 
 * This file defines the fundamental types used in the HeteroMM framework.
 * All user-defined types must inherit from one of these base types to enable
 * compile-time type checking and automatic testing.
 */

#ifndef HETEROMM_DEV_TYPES_BASE_TYPES_H_
#define HETEROMM_DEV_TYPES_BASE_TYPES_H_

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <type_traits>

namespace heteromm {
namespace data_type {

/**
 * @brief Enum representing the device type for memory allocation
 */
enum class DataLocation {
    SYS_DRAM,  ///< System RAM (CPU)
    GPU_MEM,   ///< GPU Device Memory
    FPGA_MEM   ///< FPGA Device Memory
};

/**
 * @brief Base class for all data types in the framework
 * 
 * Provides common functionality for device tracking, serialization,
 * and type identification. All derived types must implement the
 * required virtual methods.
 */
template<typename T>
class BaseType {
public:
    virtual ~BaseType() = default;
    
    /**
     * @brief Get the type name for runtime identification
     */
    virtual std::string type_name() const = 0;
    
    /**
     * @brief Get the size of the data in bytes
     */
    virtual size_t get_size() const = 0;
    
    /**
     * @brief Get the current device location of the data
     */
    DataLocation get_device() const { return device_; }

    void set_device(DataLocation device) { device_ = device; }

    /**
     * @brief export and set the contents 
     */
    T export_data() const {
        return data;
    }

    void set_data(const T& new_data) {
        data = new_data;
    }
    

protected:
    DataLocation device_ = DataLocation::SYS_DRAM;
    T data;
};

}  // namespace data_type
}  // namespace heteromm

#endif  // HETEROMM_DEV_TYPES_BASE_TYPES_H_
