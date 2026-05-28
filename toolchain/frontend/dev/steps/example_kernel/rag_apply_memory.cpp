/**
 * @file rag_apply_memory.cpp
 * @brief Implementation of RAGApplyMemory kernels
 * 
 * Uses embedded Python interpreter to call rag_apply_memory for
 * building the RAG prompt from retrieved documents and query.
 *
 * The PY_FUNC attribute is placed here (not in the header) because
 * different users may provide a pure-C++ implementation instead.
 * The PythonDispatchPass scans these .cpp files to detect Python calls.
 */

#include "../apply_memory.h"
#include <Python.h>
#include <stdexcept>
#include <sstream>

#ifndef PY_FUNC
#define PY_FUNC(func_name) __attribute__((annotate("py_func=" func_name)))
#endif

namespace heteromm {
namespace step {

PY_FUNC("rag_pipeline.rag_apply_memory")
void RAGApplyMemory::run_gpu_kernel(
    const data_type::TextDBData& retrieved_data,
    const data_type::TopKIndex& index,
    const data_type::TextInputOutputData<int>& input,
    data_type::TextInputOutputData<int>& output
) {
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }

    // Add module path to sys.path
    PyObject* sys_path = PySys_GetObject("path");
    if (sys_path) {
        PyObject* path_str = PyUnicode_FromString(python_module_path_.c_str());
        PyList_Append(sys_path, path_str);
        Py_DECREF(path_str);
    }

    // Import rag_pipeline module
    PyObject* py_module = PyImport_ImportModule("rag_pipeline");
    if (!py_module) {
        PyErr_Print();
        throw std::runtime_error("[RAGApplyMemory] Failed to import rag_pipeline module");
    }

    // Get rag_apply_memory function
    PyObject* py_func = PyObject_GetAttrString(py_module, "rag_apply_memory");
    if (!py_func || !PyCallable_Check(py_func)) {
        Py_XDECREF(py_func);
        Py_DECREF(py_module);
        PyErr_Print();
        throw std::runtime_error("[RAGApplyMemory] rag_apply_memory not found or not callable");
    }

    // Build Python list of document token lists from TextDBData
    const auto& all_docs = retrieved_data.export_data();
    const auto& topk_indices = index.export_data();
    const auto& input_tokens = input.export_data();

    // Create Python list of retrieved document token lists
    PyObject* py_doc_list = PyList_New(topk_indices.size());
    for (size_t i = 0; i < topk_indices.size(); ++i) {
        int doc_idx = topk_indices[i];
        if (doc_idx >= 0 && doc_idx < static_cast<int>(all_docs.size())) {
            const auto& doc_tokens = all_docs[doc_idx];
            PyObject* py_doc_tokens = PyList_New(doc_tokens.size());
            for (size_t j = 0; j < doc_tokens.size(); ++j) {
                PyList_SetItem(py_doc_tokens, j, PyLong_FromLong(doc_tokens[j]));
            }
            PyList_SetItem(py_doc_list, i, py_doc_tokens);
        } else {
            PyList_SetItem(py_doc_list, i, PyList_New(0));
        }
    }

    // Create Python list of input tokens (query)
    PyObject* py_input_tokens = PyList_New(input_tokens.size());
    for (size_t i = 0; i < input_tokens.size(); ++i) {
        PyList_SetItem(py_input_tokens, i, PyLong_FromLong(input_tokens[i]));
    }

    // Create Python string for model name
    PyObject* py_model_name = PyUnicode_FromString(model_name_.c_str());

    // Call rag_apply_memory(doc_list, input_tokens, model_name)
    PyObject* py_args = PyTuple_New(3);
    PyTuple_SetItem(py_args, 0, py_doc_list);
    PyTuple_SetItem(py_args, 1, py_input_tokens);
    PyTuple_SetItem(py_args, 2, py_model_name);

    PyObject* py_result = PyObject_CallObject(py_func, py_args);
    Py_DECREF(py_args);
    Py_DECREF(py_func);

    if (!py_result) {
        Py_DECREF(py_module);
        PyErr_Print();
        throw std::runtime_error("[RAGApplyMemory] rag_apply_memory() call failed");
    }

    // Result is a Python list of token IDs (the RAG prompt)
    std::vector<int> output_tokens;
    if (PyList_Check(py_result)) {
        Py_ssize_t list_size = PyList_Size(py_result);
        output_tokens.reserve(list_size);
        for (Py_ssize_t i = 0; i < list_size; ++i) {
            PyObject* item = PyList_GetItem(py_result, i);
            output_tokens.push_back(static_cast<int>(PyLong_AsLong(item)));
        }
    }

    output = data_type::TextInputOutputData<int>(output_tokens);

    Py_DECREF(py_result);
    Py_DECREF(py_module);

    std::clog << "[RAGApplyMemory] GPU kernel completed. "
              << "Built RAG prompt with " << output_tokens.size() << " tokens." << std::endl;
}

void RAGApplyMemory::run_cpu_kernel(
    const data_type::TextDBData& retrieved_data,
    const data_type::TopKIndex& index,
    const data_type::TextInputOutputData<int>& input,
    data_type::TextInputOutputData<int>& output
) {
    // Simple CPU concatenation: input tokens + retrieved doc tokens
    const auto& all_docs = retrieved_data.export_data();
    const auto& topk_indices = index.export_data();
    const auto& input_tokens = input.export_data();

    std::vector<int> output_tokens;
    output_tokens.reserve(input_tokens.size() + topk_indices.size() * 100);

    // Start with input tokens (query)
    output_tokens.insert(output_tokens.end(), input_tokens.begin(), input_tokens.end());

    // Append retrieved document tokens
    for (int doc_idx : topk_indices) {
        if (doc_idx >= 0 && doc_idx < static_cast<int>(all_docs.size())) {
            const auto& doc_tokens = all_docs[doc_idx];
            output_tokens.insert(output_tokens.end(), doc_tokens.begin(), doc_tokens.end());
        }
    }

    output = data_type::TextInputOutputData<int>(output_tokens);

    std::clog << "[RAGApplyMemory] CPU kernel completed. "
              << "Concatenated " << topk_indices.size() << " documents with query. "
              << "Output size: " << output_tokens.size() << " tokens." << std::endl;
}

void RAGApplyMemory::run_fpga_kernel(
    const data_type::TextDBData& retrieved_data,
    const data_type::TopKIndex& index,
    const data_type::TextInputOutputData<int>& input,
    data_type::TextInputOutputData<int>& output
) {
    // FPGA not used for apply_memory in RAG
    std::clog << "[RAGApplyMemory] FPGA kernel not implemented. Falling back to CPU." << std::endl;
    run_cpu_kernel(retrieved_data, index, input, output);
}

void RAGApplyMemory::run_test_kernel(
    const data_type::TextDBData& retrieved_data,
    const data_type::TopKIndex& index,
    const data_type::TextInputOutputData<int>& input,
    data_type::TextInputOutputData<int>& output
) {
    // For testing, use the CPU implementation
    run_cpu_kernel(retrieved_data, index, input, output);
}

}  // namespace step
}  // namespace heteromm
