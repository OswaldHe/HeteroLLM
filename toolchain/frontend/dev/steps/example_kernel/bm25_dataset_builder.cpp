/**
 * @file bm25_dataset_builder.cpp
 * @brief Implementation of BM25DatasetBuilder kernels
 * 
 * Uses the embedded Python interpreter (Python.h) to call
 * bm25_loader_xrt and launch_bm25 Python functions for
 * dataset preparation and FPGA device initialization.
 *
 * The PY_FUNC attribute is placed here (not in the header) because
 * different users may provide a pure-C++ implementation instead.
 * The PythonDispatchPass scans these .cpp files to detect Python calls.
 */

#include "../build_memory.h"
#include <Python.h>
#include <stdexcept>
#include <iostream>
#include <sstream>

#ifndef PY_FUNC
#define PY_FUNC(func_name) __attribute__((annotate("py_func=" func_name)))
#endif

namespace heteromm {
namespace step {

BM25DatasetBuilder::~BM25DatasetBuilder() {
    if (fpga_setup_object_) {
        Py_XDECREF(static_cast<PyObject*>(fpga_setup_object_));
        fpga_setup_object_ = nullptr;
    }
}

void BM25DatasetBuilder::ensure_python_initialized() {
    if (py_initialized_) return;

    if (!Py_IsInitialized()) {
        Py_Initialize();
    }

    // Add the module path to sys.path so we can import bm25_loader_xrt and launch_bm25
    PyObject* sys_path = PySys_GetObject("path");
    if (sys_path) {
        PyObject* path_str = PyUnicode_FromString(python_module_path_.c_str());
        PyList_Append(sys_path, path_str);
        Py_DECREF(path_str);
    }

    py_initialized_ = true;
}

PY_FUNC("bm25_loader_xrt.fpga_retriever_setup")
void BM25DatasetBuilder::run_cpu_kernel(
    const data_type::TextDBData& raw_data,
    data_type::BM25IndexMemory& memory
) {
    ensure_python_initialized();

    // Import the launch_bm25 module (which re-exports from bm25_loader_xrt)
    PyObject* py_module = PyImport_ImportModule("launch_bm25");
    if (!py_module) {
        PyErr_Print();
        throw std::runtime_error("[BM25DatasetBuilder] Failed to import launch_bm25 module");
    }

    // Call fpga_retriever_setup(bitstream, export_dir)
    PyObject* py_setup_func = PyObject_GetAttrString(py_module, "fpga_retriever_setup");
    if (!py_setup_func || !PyCallable_Check(py_setup_func)) {
        Py_XDECREF(py_setup_func);
        Py_DECREF(py_module);
        PyErr_Print();
        throw std::runtime_error("[BM25DatasetBuilder] fpga_retriever_setup not found or not callable");
    }

    // Build arguments: (bitstream_path, export_dir)
    PyObject* py_args = PyTuple_New(2);
    PyTuple_SetItem(py_args, 0, PyUnicode_FromString(bitstream_path_.c_str()));
    PyTuple_SetItem(py_args, 1, PyUnicode_FromString(export_dir_.c_str()));

    // Call the function
    PyObject* py_result = PyObject_CallObject(py_setup_func, py_args);
    Py_DECREF(py_args);
    Py_DECREF(py_setup_func);

    if (!py_result) {
        Py_DECREF(py_module);
        PyErr_Print();
        throw std::runtime_error("[BM25DatasetBuilder] fpga_retriever_setup() call failed");
    }

    if (py_result == Py_None) {
        Py_DECREF(py_result);
        Py_DECREF(py_module);
        throw std::runtime_error("[BM25DatasetBuilder] fpga_retriever_setup() returned None - FPGA setup failed");
    }

    // Store the FPGA setup tuple for use by the fused retrieval step
    // The tuple contains: (kernel, L, L_doc_total, bo_query_bitmap, bo_df_buffer,
    //                      bo_inst_mem, bo_doc_mem_0..3, bo_topk_id, buf_topk)
    if (fpga_setup_object_) {
        Py_XDECREF(static_cast<PyObject*>(fpga_setup_object_));
    }
    fpga_setup_object_ = py_result;  // takes ownership
    Py_INCREF(static_cast<PyObject*>(fpga_setup_object_));

    // Also build the BM25IndexMemory from the raw_data for the C++ side
    const auto& documents = raw_data.export_data();
    data_type::BM25IndexData index_data;

    for (size_t doc_idx = 0; doc_idx < documents.size(); ++doc_idx) {
        std::unordered_map<int, int> doc_freq;
        for (int token_id : documents[doc_idx]) {
            doc_freq[token_id]++;
        }
        index_data.doc_freqs.push_back(doc_freq);
        for (const auto& [token_id, freq] : doc_freq) {
            index_data.df_map[token_id]++;
        }
    }

    memory = data_type::BM25IndexMemory(index_data);

    Py_DECREF(py_result);
    Py_DECREF(py_module);

    std::clog << "[BM25DatasetBuilder] CPU kernel completed. "
              << "Loaded " << documents.size() << " documents, "
              << "FPGA setup stored." << std::endl;
}

void BM25DatasetBuilder::run_gpu_kernel(
    const data_type::TextDBData& raw_data,
    data_type::BM25IndexMemory& memory
) {
    // GPU is not used for build_memory in the RAG pipeline.
    std::clog << "[BM25DatasetBuilder] GPU kernel not implemented for build_memory. "
              << "Use CPU kernel for dataset preparation." << std::endl;
    run_cpu_kernel(raw_data, memory);
}

void BM25DatasetBuilder::run_fpga_kernel(
    const data_type::TextDBData& raw_data,
    data_type::BM25IndexMemory& memory
) {
    // FPGA is not directly used for build_memory.
    // The CPU kernel handles FPGA initialization as part of dataset preparation.
    std::clog << "[BM25DatasetBuilder] FPGA kernel not implemented for build_memory. "
              << "Use CPU kernel which initializes FPGA as part of setup." << std::endl;
    run_cpu_kernel(raw_data, memory);
}

void BM25DatasetBuilder::run_test_kernel(
    const data_type::TextDBData& raw_data,
    data_type::BM25IndexMemory& memory
) {
    // For testing, build the BM25 index without FPGA setup
    const auto& documents = raw_data.export_data();
    data_type::BM25IndexData index_data;

    for (size_t doc_idx = 0; doc_idx < documents.size(); ++doc_idx) {
        std::unordered_map<int, int> doc_freq;
        for (int token_id : documents[doc_idx]) {
            doc_freq[token_id]++;
        }
        index_data.doc_freqs.push_back(doc_freq);
        for (const auto& [token_id, freq] : doc_freq) {
            index_data.df_map[token_id]++;
        }
    }

    memory = data_type::BM25IndexMemory(index_data);
}

}  // namespace step
}  // namespace heteromm
