/**
 * Python Bindings for HeteroMem P2P API
 * 
 * Uses pybind11 to expose C++ classes to Python
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "p2p_manager.hpp"

namespace py = pybind11;
using namespace heteromem;

// Helper to convert numpy dtype to P2PBuffer::DataType
P2PBuffer::DataType numpy_to_dtype(const py::dtype& dtype) {
    if (dtype.is(py::dtype::of<uint32_t>())) {
        return P2PBuffer::DataType::UINT32;
    } else if (dtype.is(py::dtype::of<int32_t>())) {
        return P2PBuffer::DataType::INT32;
    } else if (dtype.is(py::dtype::of<float>())) {
        return P2PBuffer::DataType::FLOAT32;
    } else if (dtype.is(py::dtype::of<double>())) {
        return P2PBuffer::DataType::FLOAT64;
    } else {
        throw P2PException("Unsupported numpy dtype");
    }
}

PYBIND11_MODULE(heteromem_p2p, m) {
    m.doc() = "HeteroMem FPGA-GPU P2P Communication Library";

    // ========================================================================
    // Exceptions
    // ========================================================================
    
    py::register_exception<P2PException>(m, "P2PException");
    py::register_exception<FPGAException>(m, "FPGAException");
    py::register_exception<GPUException>(m, "GPUException");

    // ========================================================================
    // P2PBuffer
    // ========================================================================
    
    py::enum_<P2PBuffer::DataType>(m, "DataType")
        .value("UINT32", P2PBuffer::DataType::UINT32)
        .value("INT32", P2PBuffer::DataType::INT32)
        .value("FLOAT32", P2PBuffer::DataType::FLOAT32)
        .value("FLOAT64", P2PBuffer::DataType::FLOAT64)
        .export_values();
    
    py::class_<P2PBuffer, std::shared_ptr<P2PBuffer>>(m, "P2PBuffer")
        .def_property_readonly("size", &P2PBuffer::size)
        .def_property_readonly("count", &P2PBuffer::count)
        .def_property_readonly("dtype", &P2PBuffer::dtype)
        .def_property_readonly("is_gpu_registered", &P2PBuffer::is_gpu_registered)
        
        .def("write", [](P2PBuffer& self, py::array data) {
            // Ensure array is C-contiguous before memcpy
            if (!(data.flags() & py::array::c_style)) {
                throw P2PException("Input array must be C-contiguous");
            }
            
            // Validate that numpy dtype matches the buffer's dtype
            py::dtype np_dtype = data.dtype();
            P2PBuffer::DataType arr_dtype = numpy_to_dtype(np_dtype);
            if (arr_dtype != self.dtype()) {
                throw P2PException("Input array dtype does not match P2PBuffer dtype");
            }
            
            py::buffer_info info = data.request();
            
            // Validate element count and total size
            const size_t expected_count = self.count();
            const size_t expected_size_bytes = self.size();
            const size_t actual_count = static_cast<size_t>(info.size);
            const size_t actual_size_bytes = 
                static_cast<size_t>(info.size) * static_cast<size_t>(info.itemsize);
            
            if (actual_count != expected_count) {
                throw P2PException("Input array length does not match P2PBuffer element count");
            }
            if (actual_size_bytes != expected_size_bytes) {
                throw P2PException("Input array byte size does not match P2PBuffer size");
            }
            
            self.write(info.ptr, actual_size_bytes);
        }, "Write data from numpy array to buffer")
        
        .def("read", [](P2PBuffer& self) -> py::array {
            // Determine numpy dtype
            py::dtype dtype;
            switch (self.dtype()) {
                case P2PBuffer::DataType::UINT32:
                    dtype = py::dtype::of<uint32_t>();
                    break;
                case P2PBuffer::DataType::INT32:
                    dtype = py::dtype::of<int32_t>();
                    break;
                case P2PBuffer::DataType::FLOAT32:
                    dtype = py::dtype::of<float>();
                    break;
                case P2PBuffer::DataType::FLOAT64:
                    dtype = py::dtype::of<double>();
                    break;
            }
            
            // Create numpy array with proper shape
            std::vector<ssize_t> shape = {static_cast<ssize_t>(self.count())};
            py::array result(dtype, shape);
            py::buffer_info info = result.request();
            self.read(info.ptr, self.size());
            return result;
        }, "Read data from buffer to numpy array")
        
        .def("sync_to_device", &P2PBuffer::sync_to_device)
        .def("sync_to_host", &P2PBuffer::sync_to_host)
        .def("register_with_gpu", &P2PBuffer::register_with_gpu)
        .def("unregister_from_gpu", &P2PBuffer::unregister_from_gpu)
        .def_property_readonly(
            "is_p2p",
            &P2PBuffer::is_p2p_buffer,
            "Indicates whether this buffer is configured for direct P2P access "
            "between FPGA and GPU. This value is only reliable after "
            "register_with_gpu() has completed. It will be False if allocation or "
            "registration fell back to host-staged mode (requires explicit "
            "sync_to_device/sync_to_host)."
        );

    // ========================================================================
    // FPGADevice
    // ========================================================================
    
    py::class_<FPGADevice>(m, "FPGADevice")
        .def(py::init<const std::string&>(), py::arg("bdf"),
             "Create FPGA device from BDF string (e.g., '81:00.1')")
        .def(py::init<unsigned int>(), py::arg("device_index"),
             "Create FPGA device from device index")
        .def_static("auto_detect", &FPGADevice::auto_detect,
                   "Auto-detect first U55C FPGA device")
        
        .def_property_readonly("name", &FPGADevice::name)
        .def_property_readonly("bdf", &FPGADevice::bdf)
        .def_property_readonly("device_index", &FPGADevice::device_index)
        .def_property_readonly("is_xclbin_loaded", &FPGADevice::is_xclbin_loaded)
        
        .def("load_xclbin", &FPGADevice::load_xclbin, py::arg("xclbin_path"),
             "Load FPGA bitstream from xclbin file")
        
        .def("create_buffer", [](FPGADevice& self, size_t count, py::dtype dtype, int mem_group) {
            P2PBuffer::DataType dt = numpy_to_dtype(dtype);
            return self.create_buffer(count, dt, mem_group);
        }, py::arg("count"), py::arg("dtype"), py::arg("mem_group") = -1,
           "Create P2P buffer with specified count and numpy dtype. "
           "Pass mem_group >= 0 to target a specific HBM bank (must match FPGA kernel argument).")
        
        .def("generate_indices", &FPGADevice::generate_indices,
             py::arg("buffer"),
             py::arg("count"),
             py::arg("mode") = "sequential",
             py::arg("stride") = 1,
             py::arg("start_offset") = 0,
             py::arg("kernel_name") = "fpga_index_generator",
             "Generate indices using FPGA kernel");

    // ========================================================================
    // GPUDevice
    // ========================================================================
    
    py::class_<GPUDevice>(m, "GPUDevice")
        .def(py::init<int>(), py::arg("device_id") = 0,
             "Create GPU device with specified device ID")
        .def_static("auto_detect", &GPUDevice::auto_detect,
                   "Auto-detect default GPU device")
        
        .def_property_readonly("name", &GPUDevice::name)
        .def_property_readonly("device_id", &GPUDevice::device_id)
        .def_property_readonly("pcie_id", &GPUDevice::pcie_id)
        
        .def("read_buffer", [](GPUDevice& self, const P2PBuffer& buffer) -> py::array {
            // Determine numpy dtype
            py::dtype dtype;
            switch (buffer.dtype()) {
                case P2PBuffer::DataType::UINT32:
                    dtype = py::dtype::of<uint32_t>();
                    break;
                case P2PBuffer::DataType::INT32:
                    dtype = py::dtype::of<int32_t>();
                    break;
                case P2PBuffer::DataType::FLOAT32:
                    dtype = py::dtype::of<float>();
                    break;
                case P2PBuffer::DataType::FLOAT64:
                    dtype = py::dtype::of<double>();
                    break;
            }
            
            std::vector<ssize_t> shape = {static_cast<ssize_t>(buffer.count())};
            py::array result(dtype, shape);
            py::buffer_info info = result.request();
            self.read_buffer(buffer, info.ptr, buffer.size());
            return result;
        }, py::arg("buffer"), "Read data from P2P buffer to numpy array")
        
        .def("write_buffer", [](GPUDevice& self, P2PBuffer& buffer, py::array data) {
            // Ensure array is C-contiguous before memcpy
            if (!(data.flags() & py::array::c_style)) {
                throw GPUException("Input array must be C-contiguous");
            }
            py::buffer_info info = data.request();
            self.write_buffer(buffer, info.ptr, static_cast<size_t>(info.size) * static_cast<size_t>(info.itemsize));
        }, py::arg("buffer"), py::arg("data"),
           "Write data from numpy array to P2P buffer")
        
        .def("synchronize", &GPUDevice::synchronize,
             "Synchronize GPU operations")
        .def("memcpy_write_to_fpga",
             [](GPUDevice& self, P2PBuffer& buffer, py::array_t<uint32_t, py::array::c_style | py::array::forcecast> gpu_data) {
                 // Convenience wrapper: copy numpy data to GPU memory,
                 // then hipMemcpy to FPGA P2P buffer.
                 auto info = gpu_data.request();
                 uint32_t count = static_cast<uint32_t>(info.size);
                 size_t bytes = count * sizeof(uint32_t);
 
                 // Allocate GPU staging buffer
                 uint32_t* d_src;
                 hipError_t err = hipMalloc(&d_src, bytes);
                 if (err != hipSuccess) {
                     throw GPUException("Failed to allocate GPU staging buffer");
                 }
 
                 err = hipMemcpy(d_src, info.ptr, bytes, hipMemcpyHostToDevice);
                 if (err != hipSuccess) {
                     hipFree(d_src);
                     throw GPUException("Failed to copy data to GPU");
                 }
 
                 try {
                     self.memcpy_write_to_fpga(buffer, d_src, count);
                 } catch (...) {
                     hipFree(d_src);
                     throw;
                 }
 
                 hipFree(d_src);
             },
             py::arg("buffer"), py::arg("data"),
             "Write uint32 data to FPGA buffer via hipMemcpy through P2P. "
             "Data flows: numpy → GPU global mem → FPGA P2P buffer")
 
        .def("memcpy_write_floats_to_fpga",
             [](GPUDevice& self, P2PBuffer& buffer, py::array_t<float, py::array::c_style | py::array::forcecast> gpu_data) {
                 auto info = gpu_data.request();
                 uint32_t count = static_cast<uint32_t>(info.size);
                 size_t bytes = count * sizeof(float);
                 float* d_src;
                 hipError_t err = hipMalloc(&d_src, bytes);
                 if (err != hipSuccess) {
                     throw GPUException("Failed to allocate GPU staging buffer");
                 }
                 err = hipMemcpy(d_src, info.ptr, bytes, hipMemcpyHostToDevice);
                 if (err != hipSuccess) {
                     hipFree(d_src);
                     throw GPUException("Failed to copy data to GPU");
                 }
                 try {
                     self.memcpy_write_to_fpga(buffer, d_src, count);
                 } catch (...) {
                     hipFree(d_src);
                     throw;
                 }
                 hipFree(d_src);
             },
             py::arg("buffer"), py::arg("data"),
             "Write float data to FPGA buffer via hipMemcpy through P2P");

    // ========================================================================
    // P2PManager
    // ========================================================================
    
    py::class_<P2PManager::BenchmarkResult>(m, "BenchmarkResult")
        .def_readonly("bandwidth_gbps", &P2PManager::BenchmarkResult::bandwidth_gbps)
        .def_readonly("latency_ms", &P2PManager::BenchmarkResult::latency_ms)
        .def_readonly("bytes_transferred", &P2PManager::BenchmarkResult::bytes_transferred)
        .def("__repr__", [](const P2PManager::BenchmarkResult& r) {
            return "BenchmarkResult(bandwidth=" + std::to_string(r.bandwidth_gbps) + 
                   " GB/s, latency=" + std::to_string(r.latency_ms) + " ms)";
        });
    
    py::class_<P2PManager>(m, "P2PManager")
        .def(py::init<FPGADevice&, GPUDevice&>(),
             py::arg("fpga"), py::arg("gpu"))
        
        .def("transfer_fpga_to_gpu", &P2PManager::transfer_fpga_to_gpu,
             py::arg("buffer"))
        .def("transfer_gpu_to_fpga", &P2PManager::transfer_gpu_to_fpga,
             py::arg("buffer"))
        
        .def("benchmark_transfer", &P2PManager::benchmark_transfer,
             py::arg("buffer"),
             py::arg("fpga_to_gpu") = true,
             py::arg("iterations") = 100,
             "Benchmark P2P transfer performance")
        .def("memcpy_transfer_gpu_to_fpga",
             [](P2PManager& self, P2PBuffer& buffer, py::array_t<uint32_t, py::array::c_style | py::array::forcecast> data) {
                 auto info = data.request();
                 uint32_t count = static_cast<uint32_t>(info.size);
                 size_t bytes = count * sizeof(uint32_t);
 
                 uint32_t* d_src;
                 hipError_t err = hipMalloc(&d_src, bytes);
                 if (err != hipSuccess) {
                     throw GPUException("Failed to allocate GPU staging buffer");
                 }
 
                 err = hipMemcpy(d_src, info.ptr, bytes, hipMemcpyHostToDevice);
                 if (err != hipSuccess) {
                     hipFree(d_src);
                     throw GPUException("hipMemcpy to GPU failed");
                 }
 
                 try {
                     self.memcpy_transfer_gpu_to_fpga(buffer, d_src, count);
                 } catch (...) {
                     hipFree(d_src);
                     throw;
                 }
 
                 hipFree(d_src);
             },
             py::arg("buffer"), py::arg("data"),
             "Transfer data GPU→FPGA via hipMemcpy P2P write. "
             "Handles sync automatically for both P2P and fallback buffers.")
 
        .def("benchmark_gpu_write",
             [](P2PManager& self, P2PBuffer& buffer,
                py::array_t<uint32_t, py::array::c_style | py::array::forcecast> data, int iterations) {
                 auto info = data.request();
                 uint32_t count = static_cast<uint32_t>(info.size);
                 size_t bytes = count * sizeof(uint32_t);
 
                 uint32_t* d_src;
                 hipError_t err = hipMalloc(&d_src, bytes);
                 if (err != hipSuccess) {
                     throw GPUException("Failed to allocate GPU staging buffer");
                 }
 
                 err = hipMemcpy(d_src, info.ptr, bytes, hipMemcpyHostToDevice);
                 if (err != hipSuccess) {
                     hipFree(d_src);
                     throw GPUException("hipMemcpy to GPU failed");
                 }
 
                 P2PManager::BenchmarkResult result;
                 try {
                     result = self.benchmark_gpu_write(buffer, d_src, count, iterations);
                 } catch (...) {
                     hipFree(d_src);
                     throw;
                 }
 
                 hipFree(d_src);
                 return result;
             },
             py::arg("buffer"), py::arg("data"), py::arg("iterations") = 100,
             "Benchmark GPU→FPGA write bandwidth using hipMemcpy-based path");

    // ========================================================================
    // Utility Functions
    // ========================================================================
    
    m.def("list_fpga_devices", &utils::list_fpga_devices,
          "List all available FPGA devices");
    
    m.def("list_gpu_devices", &utils::list_gpu_devices,
          "List all available GPU devices");
    
    m.def("check_p2p_support", &utils::check_p2p_support,
          py::arg("fpga_bdf"),
          "Check if P2P is supported for FPGA device");
    
    // Version info
    m.attr("__version__") = "0.1.0";
}
