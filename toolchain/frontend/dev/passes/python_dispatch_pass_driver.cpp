/**
 * @file python_dispatch_pass_driver.cpp
 * @brief Driver program for the PythonDispatchPass source-level pass
 * 
 * Scans .cpp implementation files for PY_FUNC annotations AND deploy headers
 * for MemoryManager API structure, then generates a Python pipeline script.
 * 
 * Usage:
 *   ./python_dispatch_pass --base-dir ../steps \
 *       --source example_kernel/bm25_dataset_builder.cpp \
 *       --source example_kernel/bm25_fpga_retrieval.cpp \
 *       --source example_kernel/rag_apply_memory.cpp \
 *       --deploy-dir ../../deploy \
 *       --deploy simple_rag.h \
 *       --deploy memory_manager.h \
 *       --output-dir ./generated
 * 
 * Output:
 *   generated/heteromm_pipeline.py
 *     Mixed mode  : calls PY_FUNC Python functions + Pybind11 C++ modules
 *     Pure C++ mode: calls a single Pybind11 module wrapping the pipeline
 */

#include "python_dispatch_pass.h"
#include <iostream>
#include <string>
#include <vector>

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "Options:\n"
        << "  --base-dir <dir>     Base directory for resolving source paths\n"
        << "  --source  <file>     .cpp file to scan (repeatable)\n"
        << "  --deploy-dir <dir>   Base directory for resolving deploy paths\n"
        << "  --deploy  <file>     Deploy header to scan (repeatable)\n"
        << "  --output-dir <dir>   Directory for generated files  [./generated]\n"
        << "  --output-name <name> Name of generated Python file  [heteromm_pipeline.py]\n"
        << "  --verbose            Print extra diagnostics\n"
        << "  --help               Show this message\n";
}

int main(int argc, char* argv[]) {
    std::string base_dir;
    std::string deploy_dir;
    std::vector<std::string> source_files;
    std::vector<std::string> deploy_headers;
    std::string output_dir  = "./generated";
    std::string output_name;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--base-dir" && i + 1 < argc) {
            base_dir = argv[++i];
        } else if (arg == "--source" && i + 1 < argc) {
            source_files.push_back(argv[++i]);
        } else if (arg == "--deploy-dir" && i + 1 < argc) {
            deploy_dir = argv[++i];
        } else if (arg == "--deploy" && i + 1 < argc) {
            deploy_headers.push_back(argv[++i]);
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--output-name" && i + 1 < argc) {
            output_name = argv[++i];
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (source_files.empty()) {
        std::cerr << "Error: No --source files specified.\n";
        print_usage(argv[0]);
        return 1;
    }

    // Configure pass
    heteromm::passes::PythonDispatchPass pass;
    pass.set_base_dir(base_dir);
    pass.set_deploy_dir(deploy_dir);
    if (!output_name.empty()) pass.set_output_filename(output_name);

    for (const auto& f : source_files) {
        pass.add_source_file(f);
        if (verbose) std::clog << "  Added source: " << f << "\n";
    }

    for (const auto& f : deploy_headers) {
        pass.add_deploy_header(f);
        if (verbose) std::clog << "  Added deploy header: " << f << "\n";
    }

    // Run
    std::clog << "[PythonDispatchPass Driver] Starting analysis...\n";
    auto result = pass.run(output_dir);

    // Summary
    std::clog << "\n[PythonDispatchPass Driver] Summary:\n";
    std::clog << "  Steps analysed: " << result.steps.size() << "\n";

    for (const auto& step : result.steps) {
        std::clog << "  " << step.class_name
                  << " (" << step.step_type << "):\n";
        for (const auto& k : step.kernels) {
            std::clog << "    " << k.kernel_type << ": ";
            if (k.is_python_delegated())
                std::clog << "Python -> " << k.python_function;
            else
                std::clog << "C++";
            std::clog << "\n";
        }
    }

    if (!result.deploy.deploy_class.empty()) {
        std::clog << "\n  Deploy class: " << result.deploy.deploy_class
                  << " (base: " << result.deploy.base_class << ")\n";
        for (const auto& fm : result.deploy.factory_mappings) {
            std::clog << "    " << fm.step_role << " -> " << fm.step_class << "\n";
        }
        if (result.deploy.has_fused_retrieval)
            std::clog << "    fused_retrieve: yes\n";
    }

    if (result.has_python_kernels()) {
        std::clog << "\n  Strategy: MIXED MODE\n"
                  << "  - Python steps use original scripts directly\n"
                  << "  - C++ steps exported via Pybind11\n"
                  << "  - Connected by generated Python pipeline\n";

        std::clog << "\n  Python dependencies:\n";
        for (const auto& [mod, funcs] : result.python_dependencies) {
            std::clog << "    " << mod << ":";
            for (const auto& f : funcs) std::clog << " " << f;
            std::clog << "\n";
        }
    } else {
        std::clog << "\n  Strategy: PURE C++ MODE\n"
                  << "  - All steps compiled as C++\n"
                  << "  - Pipeline exported via single Pybind11 module\n";
    }

    std::clog << "\n  Output directory: " << output_dir
              << "\n[PythonDispatchPass Driver] Done.\n";

    return 0;
}
