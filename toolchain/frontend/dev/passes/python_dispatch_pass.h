/**
 * @file python_dispatch_pass.h
 * @brief Source-level pass for detecting PY_FUNC annotations and generating
 *        a Python pipeline script
 * 
 * This pass scans:
 *   1. .cpp implementation files (example_kernel/ *.cpp) for PY_FUNC annotations
 *   2. Deploy headers (simple_rag.h, memory_manager.h) to understand the
 *      MemoryManager API and how steps are wired together
 * 
 * Based on the scan results it emits a single Python file that provides the
 * same three public functions as MemoryManager:
 *   - build_memory
 *   - manage_memory_and_apply
 *   - build_and_apply_memory
 * 
 * Kernels annotated with PY_FUNC("module.function") are called directly as
 * Python functions.  Kernels without PY_FUNC are assumed compiled to C++ and
 * exported via Pybind11.
 * 
 * No LLVM, MLIR, or libclang dependency — purely regex-based source parsing.
 * 
 * Usage:
 * @code
 *   PythonDispatchPass pass;
 *   pass.set_base_dir("../steps");
 *   pass.add_source_file("example_kernel/bm25_dataset_builder.cpp");
 *   pass.add_source_file("example_kernel/bm25_fpga_retrieval.cpp");
 *   pass.add_source_file("example_kernel/rag_apply_memory.cpp");
 *   pass.add_deploy_header("../../deploy/simple_rag.h");
 *   pass.add_deploy_header("../../deploy/memory_manager.h");
 *   pass.run("./generated");  // → ./generated/heteromm_pipeline.py
 * @endcode
 */

#ifndef HETEROMM_DEV_PASSES_PYTHON_DISPATCH_PASS_H_
#define HETEROMM_DEV_PASSES_PYTHON_DISPATCH_PASS_H_

#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <filesystem>

namespace heteromm {
namespace passes {

// =====================================================================
// Data structures
// =====================================================================

/**
 * @brief Information about a single kernel implementation method
 */
struct KernelInfo {
    std::string method_name;      ///< e.g. "BM25DatasetBuilder::run_cpu_kernel"
    std::string kernel_type;      ///< "cpu", "gpu", "fpga"
    std::string step_class;       ///< e.g. "BM25DatasetBuilder"
    std::string python_function;  ///< PY_FUNC reference or empty
    std::string source_file;

    bool is_python_delegated() const { return !python_function.empty(); }
};

/**
 * @brief Information about a step class and its kernel implementations
 */
struct StepInfo {
    std::string class_name;
    std::string step_type;    ///< "BuildMemory", "FusedComputeScoreAndRetrieval", "ApplyMemory", ...
    std::string source_file;
    std::vector<KernelInfo> kernels;

    bool has_python_kernel() const {
        for (const auto& k : kernels) if (k.is_python_delegated()) return true;
        return false;
    }

    const KernelInfo* get_python_kernel() const {
        for (const auto& k : kernels) if (k.is_python_delegated()) return &k;
        return nullptr;
    }
};

/**
 * @brief Factory method mapping discovered from a deploy header
 * 
 * Maps: factory method name → step class it instantiates
 * e.g.  "create_build_memory_step" → "BM25DatasetBuilder"
 */
struct FactoryMapping {
    std::string factory_method;  ///< e.g. "create_build_memory_step"
    std::string step_class;      ///< e.g. "BM25DatasetBuilder"
    std::string step_role;       ///< "build_memory", "compute_score", "memory_retrieval", "apply_memory"
};

/**
 * @brief Information extracted from deploy headers (simple_rag.h + memory_manager.h)
 */
struct DeployInfo {
    std::string deploy_class;            ///< e.g. "SimpleRAG"
    std::string base_class;              ///< e.g. "SimpleRAGManager" or "MemoryManager"
    std::vector<FactoryMapping> factory_mappings;
    std::vector<std::string> extra_methods;  ///< e.g. "fused_retrieve"
    bool has_fused_retrieval = false;

    /// MemoryManager public API methods (always present)
    std::vector<std::string> api_methods = {
        "build_memory",
        "manage_memory_and_apply",
        "build_and_apply_memory"
    };

    /// Get the step class for a given role, or empty
    std::string step_for_role(const std::string& role) const {
        for (const auto& fm : factory_mappings)
            if (fm.step_role == role) return fm.step_class;
        return "";
    }
};

/**
 * @brief Result of the analysis pass
 */
struct AnalysisResult {
    std::vector<StepInfo> steps;
    std::map<std::string, std::set<std::string>> python_dependencies;
    DeployInfo deploy;

    bool has_python_kernels() const {
        for (const auto& s : steps) if (s.has_python_kernel()) return true;
        return false;
    }

    std::vector<const StepInfo*> get_python_steps() const {
        std::vector<const StepInfo*> r;
        for (const auto& s : steps) if (s.has_python_kernel()) r.push_back(&s);
        return r;
    }

    std::vector<const StepInfo*> get_cpp_steps() const {
        std::vector<const StepInfo*> r;
        for (const auto& s : steps) if (!s.has_python_kernel()) r.push_back(&s);
        return r;
    }

    /// Find a StepInfo by class_name
    const StepInfo* find_step(const std::string& class_name) const {
        for (const auto& s : steps)
            if (s.class_name == class_name) return &s;
        return nullptr;
    }
};

// =====================================================================
// Pass implementation
// =====================================================================

class PythonDispatchPass {
public:
    PythonDispatchPass() = default;
    ~PythonDispatchPass() = default;

    void add_source_file(const std::string& file_path) {
        source_files_.push_back(file_path);
    }

    void add_deploy_header(const std::string& file_path) {
        deploy_headers_.push_back(file_path);
    }

    void set_base_dir(const std::string& base_dir) {
        base_dir_ = base_dir;
    }

    void set_deploy_dir(const std::string& deploy_dir) {
        deploy_dir_ = deploy_dir;
    }

    void set_output_filename(const std::string& name) {
        output_filename_ = name;
    }

    // -----------------------------------------------------------------
    // Phase 1 — Analysis
    // -----------------------------------------------------------------

    AnalysisResult analyze() const {
        AnalysisResult result;

        // 1. Scan .cpp files for PY_FUNC annotations
        for (const auto& file_path : source_files_) {
            std::string full_path = resolve_path(file_path, base_dir_);
            std::string content = read_file_contents(full_path);
            if (content.empty()) {
                std::cerr << "[PythonDispatchPass] Warning: Could not read "
                          << full_path << std::endl;
                continue;
            }
            auto file_steps = parse_cpp_file(content, file_path);
            for (auto& s : file_steps)
                result.steps.push_back(std::move(s));
        }

        // 2. Build Python dependency map
        for (const auto& step : result.steps) {
            for (const auto& kernel : step.kernels) {
                if (kernel.is_python_delegated()) {
                    auto [mod, func] = split_python_ref(kernel.python_function);
                    result.python_dependencies[mod].insert(func);
                }
            }
        }

        // 3. Scan deploy headers for MemoryManager structure
        for (const auto& file_path : deploy_headers_) {
            std::string full_path = resolve_path(file_path, deploy_dir_);
            std::string content = read_file_contents(full_path);
            if (content.empty()) continue;
            parse_deploy_header(content, result.deploy);
        }

        return result;
    }

    // -----------------------------------------------------------------
    // Phase 2 — Code generation
    // -----------------------------------------------------------------

    AnalysisResult run(const std::string& output_dir) {
        auto result = analyze();
        log_summary(result);
        std::filesystem::create_directories(output_dir);

        if (result.has_python_kernels()) {
            emit_mixed_pipeline(result, output_dir);
        } else {
            emit_pure_cpp_pipeline(result, output_dir);
        }

        return result;
    }

private:
    std::vector<std::string> source_files_;
    std::vector<std::string> deploy_headers_;
    std::string base_dir_;
    std::string deploy_dir_;
    std::string output_filename_ = "heteromm_pipeline.py";

    // ===== Utility helpers =====

    static std::string resolve_path(const std::string& path, const std::string& dir) {
        if (dir.empty() || (!path.empty() && path[0] == '/')) return path;
        return dir + "/" + path;
    }

    static std::string read_file_contents(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return "";
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    static std::pair<std::string, std::string> split_python_ref(const std::string& ref) {
        auto dot = ref.rfind('.');
        if (dot == std::string::npos) return {"__main__", ref};
        return {ref.substr(0, dot), ref.substr(dot + 1)};
    }

    /**
     * @brief Convert CamelCase to snake_case, handling acronyms and digits.
     */
    static std::string to_snake_case(const std::string& camel) {
        std::string r;
        for (size_t i = 0; i < camel.size(); ++i) {
            char c = camel[i];
            if (std::isupper(c)) {
                if (i > 0) {
                    char prev = camel[i - 1];
                    bool next_lower = (i + 1 < camel.size() && std::islower(camel[i + 1]));
                    if (std::islower(prev) || std::isdigit(prev) || next_lower)
                        r += '_';
                }
                r += static_cast<char>(std::tolower(c));
            } else {
                if (std::isdigit(c) && i > 0 && std::islower(camel[i - 1]))
                    r += '_';
                r += c;
            }
        }
        return r;
    }

    // ===== .cpp file parsing =====

    static std::vector<StepInfo> parse_cpp_file(
        const std::string& content,
        const std::string& file_path
    ) {
        std::map<std::string, StepInfo> step_map;

        // 1. Annotated kernels
        std::regex annotated_kernel(
            R"_re_(PY_FUNC\("([^"]+)"\)\s*\n\s*void\s+(\w+)::run_(cpu|gpu|fpga)_kernel\s*\()_re_",
            std::regex::multiline
        );

        auto it_end = std::sregex_iterator();
        for (auto it = std::sregex_iterator(content.begin(), content.end(), annotated_kernel);
             it != it_end; ++it) {
            std::string py_ref = (*it)[1].str();
            std::string cls    = (*it)[2].str();
            std::string ktype  = (*it)[3].str();

            auto& step = step_map[cls];
            step.class_name = cls;
            step.source_file = file_path;

            KernelInfo ki;
            ki.method_name     = cls + "::run_" + ktype + "_kernel";
            ki.kernel_type     = ktype;
            ki.step_class      = cls;
            ki.python_function = py_ref;
            ki.source_file     = file_path;
            step.kernels.push_back(ki);
        }

        // 2. Plain (non-annotated) kernels
        std::regex plain_kernel(
            R"_re_(void\s+(\w+)::run_(cpu|gpu|fpga)_kernel\s*\()_re_",
            std::regex::multiline
        );

        for (auto it = std::sregex_iterator(content.begin(), content.end(), plain_kernel);
             it != it_end; ++it) {
            std::string cls   = (*it)[1].str();
            std::string ktype = (*it)[2].str();
            std::string mname = cls + "::run_" + ktype + "_kernel";

            auto& step = step_map[cls];
            bool already = false;
            for (const auto& k : step.kernels)
                if (k.method_name == mname) { already = true; break; }
            if (already) continue;

            step.class_name = cls;
            step.source_file = file_path;

            KernelInfo ki;
            ki.method_name = mname;
            ki.kernel_type = ktype;
            ki.step_class  = cls;
            ki.source_file = file_path;
            step.kernels.push_back(ki);
        }

        // 3. Infer step_type
        for (auto& [cls, step] : step_map)
            step.step_type = infer_step_type(content, cls);

        std::vector<StepInfo> out;
        for (auto& [_, s] : step_map) out.push_back(std::move(s));
        return out;
    }

    static std::string infer_step_type(const std::string& content,
                                        const std::string& class_name) {
        if (content.find("build_memory.h") != std::string::npos)
            return "BuildMemory";
        if (content.find("fused_steps/template.h") != std::string::npos ||
            content.find("fused_steps\\template.h") != std::string::npos)
            return "FusedComputeScoreAndRetrieval";
        if (content.find("apply_memory.h") != std::string::npos)
            return "ApplyMemory";

        std::string lower = to_snake_case(class_name);
        if (lower.find("build") != std::string::npos || lower.find("dataset") != std::string::npos)
            return "BuildMemory";
        if (lower.find("fused") != std::string::npos || lower.find("retrieval") != std::string::npos)
            return "FusedComputeScoreAndRetrieval";
        if (lower.find("apply") != std::string::npos)
            return "ApplyMemory";
        if (lower.find("score") != std::string::npos)
            return "ComputeScore";
        return "Unknown";
    }

    // ===== Deploy header parsing =====

    /**
     * @brief Parse deploy headers to extract MemoryManager API structure
     * 
     * Discovers:
     * - The concrete deploy class (e.g. SimpleRAG) and its base
     * - Factory method → step class mappings (create_*_step → make_shared<StepClass>)
     * - Extra public methods like fused_retrieve
     */
    static void parse_deploy_header(const std::string& content, DeployInfo& deploy) {
        // Detect deploy class: class SimpleRAG : public SimpleRAGManager
        // or: class SimpleRAG : public MemoryManager<...>
        std::regex deploy_class_re(
            R"_re_(class\s+(\w+)\s*:\s*public\s+(\w+))_re_"
        );
        for (auto it = std::sregex_iterator(content.begin(), content.end(), deploy_class_re);
             it != std::sregex_iterator(); ++it) {
            std::string cls  = (*it)[1].str();
            std::string base = (*it)[2].str();
            // Skip builder classes
            if (cls.find("Builder") != std::string::npos) continue;
            // Pick the one that inherits from *Manager* or a using alias thereof
            if (base.find("Manager") != std::string::npos ||
                base.find("manager") != std::string::npos ||
                deploy.deploy_class.empty()) {
                deploy.deploy_class = cls;
                deploy.base_class = base;
            }
        }

        // Detect factory methods:
        //   create_build_memory_step()   → make_shared<BM25DatasetBuilder>(...)
        //   create_compute_score_step()  → nullptr / make_shared<...>
        //   create_memory_retrieval_step() → nullptr / make_shared<...>
        //   create_apply_memory_step()   → make_shared<RAGApplyMemory>(...)
        std::regex factory_re(
            R"_re_(create_(build_memory|compute_score|memory_retrieval|apply_memory)_step\b)_re_"
        );
        std::regex make_shared_re(
            R"_re_(make_shared<\w+::(\w+)>)_re_"
        );
        std::regex nullptr_re(
            R"_re_(return\s+nullptr)_re_"
        );

        // For each factory method, find the body and check what it returns
        auto fact_begin = std::sregex_iterator(content.begin(), content.end(), factory_re);
        for (auto it = fact_begin; it != std::sregex_iterator(); ++it) {
            std::string role = (*it)[1].str();
            std::string factory_name = "create_" + role + "_step";

            // Find the function body after the match
            size_t pos = it->position() + it->length();
            // Find the opening brace
            size_t brace_start = content.find('{', pos);
            if (brace_start == std::string::npos) continue;
            // Find the matching closing brace (simplified — works for simple methods)
            int depth = 1;
            size_t brace_end = brace_start + 1;
            while (brace_end < content.size() && depth > 0) {
                if (content[brace_end] == '{') depth++;
                else if (content[brace_end] == '}') depth--;
                brace_end++;
            }
            std::string body = content.substr(brace_start, brace_end - brace_start);

            // Check if it returns nullptr
            std::smatch null_match;
            if (std::regex_search(body, null_match, nullptr_re)) {
                // No step for this role — skip
                continue;
            }

            // Check for make_shared<step::ClassName>
            std::smatch ms_match;
            if (std::regex_search(body, ms_match, make_shared_re)) {
                FactoryMapping fm;
                fm.factory_method = factory_name;
                fm.step_class = ms_match[1].str();
                fm.step_role = role;
                deploy.factory_mappings.push_back(fm);
            }
        }

        // Detect fused_retrieve method
        if (content.find("fused_retrieve") != std::string::npos) {
            deploy.has_fused_retrieval = true;
            deploy.extra_methods.push_back("fused_retrieve");
        }
    }

    // ===== Logging =====

    void log_summary(const AnalysisResult& result) const {
        std::clog << "[PythonDispatchPass] Analysis complete:\n"
                  << "  Steps found: " << result.steps.size() << "\n"
                  << "  Python-delegated: " << result.get_python_steps().size() << "\n"
                  << "  Pure C++: " << result.get_cpp_steps().size() << "\n";

        if (!result.deploy.deploy_class.empty()) {
            std::clog << "  Deploy class: " << result.deploy.deploy_class
                      << " (base: " << result.deploy.base_class << ")\n";
            for (const auto& fm : result.deploy.factory_mappings) {
                std::clog << "    " << fm.step_role << " → " << fm.step_class << "\n";
            }
            if (result.deploy.has_fused_retrieval)
                std::clog << "    fused_retrieve: yes\n";
        }

        if (result.has_python_kernels()) {
            std::clog << "  Strategy: MIXED MODE\n";
            for (const auto& [mod, funcs] : result.python_dependencies) {
                std::clog << "    import " << mod << " → ";
                for (const auto& f : funcs) std::clog << f << " ";
                std::clog << "\n";
            }
        } else {
            std::clog << "  Strategy: PURE C++ MODE\n";
        }
    }

    // ===== Code generation helpers =====

    /// Emit the Python call for a step (Python PY_FUNC or C++ pybind11)
    static void emit_step_call(std::ofstream& out,
                                const StepInfo* step,
                                const std::string& indent,
                                const std::string& args) {
        if (!step) {
            out << indent << "pass  # step not configured\n";
            return;
        }
        auto pk = step->get_python_kernel();
        if (pk) {
            auto [mod, func] = split_python_ref(pk->python_function);
            out << indent << "result = " << func << "(" << args << ")\n";
        } else {
            std::string cpp_func = step->class_name + "_"
                + (step->kernels.empty() ? "cpu" : step->kernels[0].kernel_type);
            out << indent << "if heteromm_cpp_kernels is None:\n";
            out << indent << "    raise RuntimeError('C++ kernel "
                << step->class_name << " not available')\n";
            out << indent << "result = heteromm_cpp_kernels."
                << cpp_func << "(" << args << ")\n";
        }
    }

    // -----------------------------------------------------------------
    // Mixed mode code generation
    // -----------------------------------------------------------------

    void emit_mixed_pipeline(
        const AnalysisResult& result,
        const std::string& output_dir
    ) const {
        std::string filepath = output_dir + "/" + output_filename_;
        std::ofstream out(filepath);

        const auto& deploy = result.deploy;

        // ---- header ----
        out << "#!/usr/bin/env python3\n";
        out << "\"\"\"\n";
        out << "Auto-generated HeteroMM pipeline.\n";
        out << "Generated by PythonDispatchPass (mixed mode).\n\n";
        out << "Provides the same three public API methods as MemoryManager:\n";
        out << "  - build_memory\n";
        out << "  - manage_memory_and_apply\n";
        out << "  - build_and_apply_memory\n";
        if (deploy.has_fused_retrieval)
            out << "  - fused_retrieve  (extra helper)\n";
        out << "\n";
        out << "Steps with PY_FUNC annotations call the original Python functions.\n";
        out << "Steps without PY_FUNC call C++ kernels exported via Pybind11.\n";
        out << "\"\"\"\n\n";

        // ---- imports ----
        out << "import os\n";
        out << "import logging\n";
        out << "import sys\n";
        out << "import time\n\n";

        out << "# Ensure Python modules from backend/lib/python are importable\n";
        out << "_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))\n";
        out << "_LIB_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "
            << "'../../../backend/lib/python'))\n";
        out << "if _LIB_DIR not in sys.path:\n";
        out << "    sys.path.insert(0, _LIB_DIR)\n\n";

        out << "logging.basicConfig(level=logging.INFO,\n";
        out << "                    format='%(asctime)s - %(levelname)s - %(message)s')\n";
        out << "logger = logging.getLogger(__name__)\n\n";

        // PY_FUNC imports
        for (const auto& [module, functions] : result.python_dependencies) {
            out << "from " << module << " import ";
            bool first = true;
            for (const auto& func : functions) {
                if (!first) out << ", ";
                out << func;
                first = false;
            }
            out << "\n";
        }
        if (!result.python_dependencies.empty()) out << "\n";

        // Pybind11 import
        auto cpp_steps = result.get_cpp_steps();
        if (!cpp_steps.empty()) {
            out << "try:\n";
            out << "    import heteromm_cpp_kernels\n";
            out << "except ImportError:\n";
            out << "    heteromm_cpp_kernels = None\n";
            out << "    logger.warning('heteromm_cpp_kernels not built; "
                << "C++ kernels unavailable.')\n\n";
        }

        // ---- LatencyTracker ----
        out << "\nclass LatencyTracker:\n";
        out << "    \"\"\"Lightweight latency tracker.\"\"\"\n";
        out << "    def __init__(self):\n";
        out << "        self.metrics = {}\n";
        out << "        self._starts = {}\n\n";
        out << "    def start(self, name):\n";
        out << "        self._starts[name] = time.perf_counter()\n\n";
        out << "    def stop(self, name):\n";
        out << "        elapsed = time.perf_counter() - self._starts.pop(name, time.perf_counter())\n";
        out << "        self.metrics.setdefault(name, []).append(elapsed)\n";
        out << "        return elapsed\n\n";
        out << "    def summary(self):\n";
        out << "        for name, times in self.metrics.items():\n";
        out << "            avg = sum(times) / len(times) * 1000\n";
        out << "            logger.info(f'{name}: avg={avg:.2f} ms  count={len(times)}')\n\n";

        // ---- Resolve step pointers for each role ----
        // Use deploy info if available, otherwise fall back to step_type heuristics
        const StepInfo* build_step = nullptr;
        const StepInfo* score_step = nullptr;
        const StepInfo* retrieval_step = nullptr;
        const StepInfo* apply_step = nullptr;
        const StepInfo* fused_step = nullptr;

        if (!deploy.factory_mappings.empty()) {
            // Use deploy header info
            std::string bm_cls = deploy.step_for_role("build_memory");
            std::string cs_cls = deploy.step_for_role("compute_score");
            std::string mr_cls = deploy.step_for_role("memory_retrieval");
            std::string am_cls = deploy.step_for_role("apply_memory");

            build_step = result.find_step(bm_cls);
            score_step = result.find_step(cs_cls);
            retrieval_step = result.find_step(mr_cls);
            apply_step = result.find_step(am_cls);
        } else {
            // Fallback: heuristic matching by step_type
            for (const auto& s : result.steps) {
                if (s.step_type == "BuildMemory") build_step = &s;
                if (s.step_type == "ComputeScore") score_step = &s;
                if (s.step_type == "MemoryRetrieval") retrieval_step = &s;
                if (s.step_type == "ApplyMemory") apply_step = &s;
            }
        }
        // Fused step
        for (const auto& s : result.steps)
            if (s.step_type == "FusedComputeScoreAndRetrieval") fused_step = &s;

        bool use_fused = (fused_step != nullptr) &&
                         (score_step == nullptr) && (retrieval_step == nullptr);

        // ---- Pipeline class ----
        std::string py_class = deploy.deploy_class.empty()
            ? "HeteroMMPipeline" : deploy.deploy_class + "Pipeline";

        out << "\nclass " << py_class << ":\n";
        out << "    \"\"\"\n";
        out << "    Auto-generated pipeline matching MemoryManager API.\n\n";
        out << "    Steps:\n";
        for (const auto& step : result.steps) {
            auto pk = step.get_python_kernel();
            out << "      " << step.class_name << " (" << step.step_type << ") ";
            if (pk) out << "[Python: " << pk->python_function << "]";
            else    out << "[C++/Pybind11]";
            out << "\n";
        }
        out << "    \"\"\"\n\n";

        // __init__
        out << "    def __init__(self, **kwargs):\n";
        out << "        self.config = kwargs\n";
        out << "        self.latency = LatencyTracker()\n";
        out << "        self._memory = None\n";
        out << "        self._fpga_setup = None\n";
        out << "        self._memory_built = False\n\n";

        // ---- Internal step methods ----
        auto emit_step_method = [&](const StepInfo* step) {
            if (!step) return;
            std::string method = "_run_" + to_snake_case(step->class_name);

            out << "    def " << method << "(self, *args, **kwargs):\n";
            out << "        \"\"\"Internal: " << step->class_name
                << " (" << step->step_type << ")\"\"\"\n";
            out << "        self.latency.start('" << step->class_name << "')\n";

            auto pk = step->get_python_kernel();
            if (pk) {
                auto [mod, func] = split_python_ref(pk->python_function);
                out << "        result = " << func << "(*args, **kwargs)\n";
            } else {
                std::string cpp_func = step->class_name + "_"
                    + (step->kernels.empty() ? "cpu" : step->kernels[0].kernel_type);
                out << "        if heteromm_cpp_kernels is None:\n";
                out << "            raise RuntimeError('C++ kernel "
                    << step->class_name << " not available')\n";
                out << "        result = heteromm_cpp_kernels."
                    << cpp_func << "(*args, **kwargs)\n";
            }

            out << "        elapsed = self.latency.stop('" << step->class_name << "')\n";
            out << "        logger.info(f'" << step->class_name
                << " completed in {elapsed*1000:.2f} ms')\n";
            out << "        return result\n\n";
        };

        emit_step_method(build_step);
        if (fused_step) emit_step_method(fused_step);
        if (score_step) emit_step_method(score_step);
        if (retrieval_step) emit_step_method(retrieval_step);
        emit_step_method(apply_step);

        // ================================================================
        // Public API: build_memory
        // ================================================================
        out << "    # ================================================================\n";
        out << "    # Public API (matching MemoryManager)\n";
        out << "    # ================================================================\n\n";

        out << "    def build_memory(self, retrieved_data, *args, **kwargs):\n";
        out << "        \"\"\"\n";
        out << "        Build memory structure from retrieved data.\n";
        out << "        Corresponds to MemoryManager::build_memory().\n";
        out << "        \"\"\"\n";
        if (build_step) {
            out << "        self._fpga_setup = self._run_"
                << to_snake_case(build_step->class_name)
                << "(retrieved_data, *args, **kwargs)\n";
        } else {
            out << "        pass  # no build_memory step configured\n";
        }
        out << "        self._memory_built = True\n";
        out << "        return self._fpga_setup\n\n";

        // ================================================================
        // Public API: manage_memory_and_apply
        // ================================================================
        out << "    def manage_memory_and_apply(self, retrieved_data, memory, query, input_data, **kwargs):\n";
        out << "        \"\"\"\n";
        out << "        Run query through memory and apply to produce output.\n";
        out << "        Corresponds to MemoryManager::manage_memory_and_apply().\n\n";
        out << "        Steps:\n";
        if (use_fused)
            out << "          1. fused compute_score + retrieval (FPGA)\n";
        else {
            out << "          1. compute_score:     (memory, query) -> scores\n";
            out << "          2. memory_retrieval:   scores -> indices\n";
        }
        out << "          " << (use_fused ? "2" : "3")
            << ". apply_memory:       (retrieved_data, indices, input) -> output\n";
        out << "        \"\"\"\n";
        out << "        results = {}\n\n";

        if (use_fused && fused_step) {
            out << "        # Fused compute_score + memory_retrieval\n";
            out << "        indices = self._run_"
                << to_snake_case(fused_step->class_name)
                << "(self._fpga_setup, query, **kwargs)\n";
            out << "        results['fused_retrieve'] = indices\n\n";
        } else {
            if (score_step) {
                out << "        # compute_score\n";
                out << "        scores = self._run_"
                    << to_snake_case(score_step->class_name)
                    << "(memory, query, **kwargs)\n";
                out << "        results['compute_score'] = scores\n\n";
            }
            if (retrieval_step) {
                out << "        # memory_retrieval\n";
                out << "        indices = self._run_"
                    << to_snake_case(retrieval_step->class_name)
                    << "(scores, **kwargs)\n";
                out << "        results['memory_retrieval'] = indices\n\n";
            }
        }

        if (apply_step) {
            out << "        # apply_memory\n";
            out << "        output = self._run_"
                << to_snake_case(apply_step->class_name)
                << "(query, retrieved_data, indices, input_data, **kwargs)\n";
            out << "        results['apply_memory'] = output\n\n";
        }

        out << "        return results\n\n";

        // ================================================================
        // Public API: build_and_apply_memory
        // ================================================================
        out << "    def build_and_apply_memory(self, retrieved_data, query, input_data, **kwargs):\n";
        out << "        \"\"\"\n";
        out << "        Full pipeline: build_memory + manage_memory_and_apply.\n";
        out << "        Corresponds to MemoryManager::build_and_apply_memory().\n";
        out << "        \"\"\"\n";
        out << "        if not self._memory_built:\n";
        out << "            self.build_memory(retrieved_data, **kwargs)\n\n";
        out << "        return self.manage_memory_and_apply(\n";
        out << "            retrieved_data, self._fpga_setup, query, input_data, **kwargs)\n\n";

        // ================================================================
        // Extra: fused_retrieve (if present in deploy header)
        // ================================================================
        if (deploy.has_fused_retrieval && fused_step) {
            out << "    def fused_retrieve(self, memory, query, **kwargs):\n";
            out << "        \"\"\"\n";
            out << "        Run fused compute_score + memory_retrieval.\n";
            out << "        Corresponds to " << deploy.deploy_class << "::fused_retrieve().\n";
            out << "        \"\"\"\n";
            out << "        return self._run_"
                << to_snake_case(fused_step->class_name)
                << "(self._fpga_setup, query, **kwargs)\n\n";
        }

        // ---- summary ----
        out << "    def print_latency_summary(self):\n";
        out << "        \"\"\"Print timing summary for all steps.\"\"\"\n";
        out << "        self.latency.summary()\n";

        out.close();
        std::clog << "[PythonDispatchPass] Generated: " << filepath << std::endl;
    }

    // -----------------------------------------------------------------
    // Pure C++ mode
    // -----------------------------------------------------------------

    void emit_pure_cpp_pipeline(
        const AnalysisResult& /* result */,
        const std::string& output_dir
    ) const {
        std::string filepath = output_dir + "/" + output_filename_;
        std::ofstream out(filepath);

        out << "#!/usr/bin/env python3\n";
        out << "\"\"\"\n";
        out << "Auto-generated HeteroMM pipeline (pure C++ mode).\n";
        out << "Generated by PythonDispatchPass.\n\n";
        out << "All steps are compiled as C++ and exported via a single Pybind11 module.\n";
        out << "Provides the same three public API methods as MemoryManager.\n";
        out << "\"\"\"\n\n";

        out << "import os\n";
        out << "import logging\n";
        out << "import sys\n";
        out << "import time\n\n";

        out << "# Ensure Python modules from backend/lib/python are importable\n";
        out << "_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))\n";
        out << "_LIB_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "
            << "'../../../backend/lib/python'))\n";
        out << "if _LIB_DIR not in sys.path:\n";
        out << "    sys.path.insert(0, _LIB_DIR)\n\n";

        out << "logging.basicConfig(level=logging.INFO,\n";
        out << "                    format='%(asctime)s - %(levelname)s - %(message)s')\n";
        out << "logger = logging.getLogger(__name__)\n\n";

        out << "import heteromm_pipeline as _cpp\n\n";

        out << "\ndef build_memory(*args, **kwargs):\n";
        out << "    \"\"\"Build memory index (C++). Matches MemoryManager::build_memory().\"\"\"\n";
        out << "    return _cpp.build_memory(*args, **kwargs)\n\n";

        out << "\ndef manage_memory_and_apply(*args, **kwargs):\n";
        out << "    \"\"\"Run query + apply (C++). Matches MemoryManager::manage_memory_and_apply().\"\"\"\n";
        out << "    return _cpp.manage_memory_and_apply(*args, **kwargs)\n\n";

        out << "\ndef build_and_apply_memory(*args, **kwargs):\n";
        out << "    \"\"\"Full pipeline (C++). Matches MemoryManager::build_and_apply_memory().\"\"\"\n";
        out << "    return _cpp.build_and_apply_memory(*args, **kwargs)\n";

        out.close();
        std::clog << "[PythonDispatchPass] Generated: " << filepath << std::endl;
    }
};

}  // namespace passes
}  // namespace heteromm

#endif  // HETEROMM_DEV_PASSES_PYTHON_DISPATCH_PASS_H_
