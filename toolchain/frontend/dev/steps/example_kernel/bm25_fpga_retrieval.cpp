/**
 * @file bm25_fpga_retrieval.cpp
 * @brief Implementation of FusedBM25Retrieval kernels
 * 
 * Uses embedded Python interpreter to call launch_bm25.fpga_retriver_launch
 * for fused BM25 score computation and top-K retrieval on FPGA.
 *
 * The PY_FUNC attribute is placed here (not in the header) because
 * different users may provide a pure-C++ implementation instead.
 * The PythonDispatchPass scans these .cpp files to detect Python calls.
 */

#include "../fused_steps/template.h"
#include <Python.h>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>

#ifndef PY_FUNC
#define PY_FUNC(func_name) __attribute__((annotate("py_func=" func_name)))
#endif

namespace heteromm {
namespace step {

FusedBM25Retrieval::~FusedBM25Retrieval() {
    if (fpga_setup_object_) {
        Py_XDECREF(static_cast<PyObject*>(fpga_setup_object_));
        fpga_setup_object_ = nullptr;
    }
}

PY_FUNC("launch_bm25.fpga_retriver_launch")
void FusedBM25Retrieval::run_fpga_kernel(
    const data_type::BM25IndexMemory& memory,
    const data_type::BM25Query& query,
    data_type::TopKIndex& index
) {
    PyObject* py_setup = static_cast<PyObject*>(fpga_setup_object_);
    if (!py_setup || py_setup == Py_None) {
        throw std::runtime_error(
            "[FusedBM25Retrieval] FPGA setup object not set. "
            "Call BM25DatasetBuilder::run_cpu_kernel first.");
    }

    if (!Py_IsInitialized()) {
        throw std::runtime_error("[FusedBM25Retrieval] Python interpreter not initialized");
    }

    // Import launch_bm25 module
    PyObject* py_module = PyImport_ImportModule("launch_bm25");
    if (!py_module) {
        PyErr_Print();
        throw std::runtime_error("[FusedBM25Retrieval] Failed to import launch_bm25 module");
    }

    // Get fpga_retriver_launch function
    PyObject* py_launch_func = PyObject_GetAttrString(py_module, "fpga_retriver_launch");
    if (!py_launch_func || !PyCallable_Check(py_launch_func)) {
        Py_XDECREF(py_launch_func);
        Py_DECREF(py_module);
        PyErr_Print();
        throw std::runtime_error("[FusedBM25Retrieval] fpga_retriver_launch not callable");
    }

    // Build query token string from BM25Query (token_id -> freq map)
    const auto& query_data = query.export_data();
    std::ostringstream query_str;
    bool first = true;
    for (const auto& [token_id, freq] : query_data) {
        for (int i = 0; i < freq; ++i) {
            if (!first) query_str << ",";
            query_str << token_id;
            first = false;
        }
    }

    // Unpack FPGA setup tuple
    if (!PyTuple_Check(py_setup) || PyTuple_Size(py_setup) < 12) {
        Py_DECREF(py_launch_func);
        Py_DECREF(py_module);
        throw std::runtime_error("[FusedBM25Retrieval] Invalid FPGA setup object format");
    }

    // Build arguments for fpga_retriver_launch
    PyObject* py_args = PyTuple_New(13);
    for (int i = 0; i < 12; ++i) {
        PyObject* item = PyTuple_GetItem(py_setup, i);
        Py_INCREF(item);
        PyTuple_SetItem(py_args, i, item);
    }
    PyTuple_SetItem(py_args, 12, PyUnicode_FromString(query_str.str().c_str()));

    // Call fpga_retriver_launch
    PyObject* py_result = PyObject_CallObject(py_launch_func, py_args);
    Py_DECREF(py_args);
    Py_DECREF(py_launch_func);

    if (!py_result) {
        Py_DECREF(py_module);
        PyErr_Print();
        throw std::runtime_error("[FusedBM25Retrieval] fpga_retriver_launch() call failed");
    }

    // Result is a tuple: (topk_ids_list, kernel_latency_ms)
    if (!PyTuple_Check(py_result) || PyTuple_Size(py_result) < 2) {
        Py_DECREF(py_result);
        Py_DECREF(py_module);
        throw std::runtime_error("[FusedBM25Retrieval] Unexpected return format from fpga_retriver_launch");
    }

    PyObject* py_topk_list = PyTuple_GetItem(py_result, 0);

    // Convert Python list of top-K indices to C++ vector
    std::vector<int> topk_indices;
    if (PyList_Check(py_topk_list)) {
        Py_ssize_t list_size = PyList_Size(py_topk_list);
        // Exclude the last 3 padding entries as in rag_pipeline.py
        Py_ssize_t effective_size = list_size > 3 ? list_size - 3 : list_size;
        topk_indices.reserve(effective_size);
        for (Py_ssize_t i = 0; i < effective_size; ++i) {
            PyObject* item = PyList_GetItem(py_topk_list, i);
            topk_indices.push_back(static_cast<int>(PyLong_AsLong(item)));
        }
    }

    index = data_type::TopKIndex(topk_indices);

    Py_DECREF(py_result);
    Py_DECREF(py_module);

    std::clog << "[FusedBM25Retrieval] FPGA kernel completed. "
              << "Retrieved " << topk_indices.size() << " document indices." << std::endl;
}

void FusedBM25Retrieval::run_cpu_kernel(
    const data_type::BM25IndexMemory& memory,
    const data_type::BM25Query& query,
    data_type::TopKIndex& index
) {
    // Software BM25 fallback implementation
    const auto& mem_data = memory.export_data();
    const auto& query_data = query.export_data();
    
    size_t num_docs = mem_data.doc_freqs.size();
    if (num_docs == 0) {
        index = data_type::TopKIndex({});
        return;
    }

    // BM25 parameters
    const double k1 = 1.2;
    const double b = 0.75;
    const double N = static_cast<double>(num_docs);

    // Compute average document length
    double avg_dl = 0.0;
    std::vector<double> doc_lengths(num_docs, 0.0);
    for (size_t i = 0; i < num_docs; ++i) {
        double dl = 0.0;
        for (const auto& [tid, freq] : mem_data.doc_freqs[i]) {
            dl += freq;
        }
        doc_lengths[i] = dl;
        avg_dl += dl;
    }
    avg_dl /= N;

    // Compute BM25 scores for each document
    std::vector<double> scores(num_docs, 0.0);
    for (const auto& [query_token, query_freq] : query_data) {
        auto df_it = mem_data.df_map.find(query_token);
        if (df_it == mem_data.df_map.end()) continue;
        
        double df = static_cast<double>(df_it->second);
        double idf = std::log((N - df + 0.5) / (df + 0.5) + 1.0);

        for (size_t d = 0; d < num_docs; ++d) {
            auto tf_it = mem_data.doc_freqs[d].find(query_token);
            if (tf_it == mem_data.doc_freqs[d].end()) continue;
            
            double tf = static_cast<double>(tf_it->second);
            double dl = doc_lengths[d];
            double score = idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avg_dl));
            scores[d] += score;
        }
    }

    // Top-K selection (k=64 matching FPGA)
    const size_t top_k = 64;
    std::vector<int> indices(num_docs);
    std::iota(indices.begin(), indices.end(), 0);
    
    size_t k = std::min(top_k, num_docs);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [&scores](int a, int b_idx) { return scores[a] > scores[b_idx]; });
    
    indices.resize(k);
    index = data_type::TopKIndex(indices);

    std::clog << "[FusedBM25Retrieval] CPU kernel completed. "
              << "Retrieved top-" << k << " documents." << std::endl;
}

void FusedBM25Retrieval::run_gpu_kernel(
    const data_type::BM25IndexMemory& memory,
    const data_type::BM25Query& query,
    data_type::TopKIndex& index
) {
    // GPU not implemented for BM25 retrieval, fallback to CPU
    std::clog << "[FusedBM25Retrieval] GPU kernel not implemented. Falling back to CPU." << std::endl;
    run_cpu_kernel(memory, query, index);
}

void FusedBM25Retrieval::run_test_kernel(
    const data_type::BM25IndexMemory& memory,
    const data_type::BM25Query& query,
    data_type::TopKIndex& index
) {
    // For testing, use the CPU software BM25 implementation
    run_cpu_kernel(memory, query, index);
}

}  // namespace step
}  // namespace heteromm
