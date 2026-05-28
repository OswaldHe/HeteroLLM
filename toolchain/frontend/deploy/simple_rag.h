/**
 * @file simple_rag.h
 * @brief RAG Memory Manager for BM25-based retrieval-augmented generation
 * 
 * Connects the RAG pipeline steps:
 * 1. BM25DatasetBuilder (build_memory, CPU): loads corpus + initializes FPGA
 * 2. FusedBM25Retrieval (fused compute_score + retrieval, FPGA): BM25 scoring + top-K
 * 3. RAGApplyMemory (apply_memory, GPU): builds RAG prompt from retrieved docs
 * 
 * Follows the same pattern as paged_attention.h.
 * 
 * This mirrors the flow in rag_test/rag_pipeline.py:
 * - fpga_retriever_setup() → BM25DatasetBuilder::run_cpu_kernel
 * - fpga_retriver_launch() → FusedBM25Retrieval::run_fpga_kernel
 * - _build_rag_prompt()    → RAGApplyMemory::run_gpu_kernel
 */

#ifndef HETEROMM_DEPLOY_SIMPLE_RAG_H_
#define HETEROMM_DEPLOY_SIMPLE_RAG_H_

#include "memory_manager.h"
#include "../dev/steps/fused_steps/template.h"
#include <string>
#include <memory>
#include <chrono>

namespace heteromm {
namespace deploy {

/**
 * @brief Type alias for the RAG memory manager base
 * 
 * Data flow:
 * - RetrievedData: TextDBData - the tokenized corpus
 * - Memory: BM25IndexMemory - BM25 index structure
 * - Query: BM25Query - query token frequencies
 * - Score: VectorScore<float> - BM25 scores (used in unfused path)
 * - Index: TopKIndex - top-K document indices
 * - Input: TextInputOutputData<int> - query tokens
 * - Output: TextInputOutputData<int> - RAG prompt tokens
 */
using SimpleRAGManager = MemoryManager<
    data_type::TextDBData,                   // RetrievedData: tokenized corpus
    data_type::BM25IndexMemory,              // Memory: BM25 index
    data_type::BM25Query,                    // Query: token frequency map
    data_type::VectorScore<float>,           // Score: BM25 scores
    data_type::TopKIndex,                    // Index: top-K doc indices
    data_type::TextInputOutputData<int>,     // Input: query tokens
    data_type::TextInputOutputData<int>      // Output: RAG prompt tokens
>;

/**
 * @brief Concrete RAG memory manager with BM25 + FPGA step implementations
 * 
 * This class provides a ready-to-use RAG pipeline implementation.
 * It follows the same pattern as PagedAttention:
 * - Inherits from MemoryManager (via SimpleRAGManager typedef)
 * - Overrides factory methods to create step instances
 * - Provides accessors and setters for pipeline-specific parameters
 * 
 * The standard 4-step flow (build → score → retrieve → apply) can be
 * used directly. Additionally, a fused_retrieve() helper is available
 * to run the combined BM25 scoring + top-K on FPGA in a single pass.
 * 
 * Usage:
 * @code
 *   SimpleRAG rag("schedule.json", "../indexer_bm25.xclbin", "./export");
 *   
 *   TextDBData corpus = ...;
 *   BM25IndexMemory index;
 *   rag.build_memory(corpus, index);
 *   
 *   BM25Query query = ...;
 *   TextInputOutputData<int> input_tokens = ...;
 *   TextInputOutputData<int> output_tokens;
 *   rag.manage_memory_and_apply(corpus, index, query, input_tokens, output_tokens);
 * @endcode
 */
class SimpleRAG : public SimpleRAGManager {
public:
    /**
     * @brief Construct a RAG memory manager
     * 
     * @param schedule_path Path to the schedule JSON file
     * @param bitstream_path Path to FPGA bitstream (.xclbin)
     * @param export_dir Directory with exported BM25 data files
     * @param python_module_path Path to Python modules directory
     * @param model_name HuggingFace model name for tokenizer
     */
    SimpleRAG(
        const std::string& schedule_path,
        const std::string& bitstream_path = "../indexer_bm25.xclbin",
        const std::string& export_dir = "./export",
        const std::string& python_module_path = ".",
        const std::string& model_name = "meta-llama/Llama-3.2-1B-Instruct"
    ) : SimpleRAGManager(schedule_path),
        bitstream_path_(bitstream_path),
        export_dir_(export_dir),
        python_module_path_(python_module_path),
        model_name_(model_name) {}

    /**
     * @brief Default constructor
     */
    explicit SimpleRAG(const std::string& schedule_path = "")
        : SimpleRAGManager(schedule_path) {}

    ~SimpleRAG() override = default;

    // ===== Parameter accessors =====

    void set_bitstream_path(const std::string& path) {
        bitstream_path_ = path;
        build_memory_step_.reset();
    }

    void set_export_dir(const std::string& dir) {
        export_dir_ = dir;
        build_memory_step_.reset();
    }

    void set_python_module_path(const std::string& path) {
        python_module_path_ = path;
        build_memory_step_.reset();
        apply_memory_step_.reset();
    }

    void set_model_name(const std::string& name) {
        model_name_ = name;
        apply_memory_step_.reset();
    }

    const std::string& bitstream_path() const { return bitstream_path_; }
    const std::string& export_dir() const { return export_dir_; }
    const std::string& python_module_path() const { return python_module_path_; }
    const std::string& model_name() const { return model_name_; }

    int ret_data_size(
        const data_type::TextDBData& retrieved_data
    ) {
        return retrieved_data.get_num_documents();
    }

    int memory_size(
        const data_type::BM25IndexMemory& memory
    ) {
        return memory.get_size();
    }

    // ===== Fused retrieval helper =====

    /**
     * @brief Run fused compute_score + memory_retrieval on FPGA
     * 
     * This is an additional helper (not overriding the base class).
     * It replaces the separate compute_score → memory_retrieval flow
     * with a single fused FPGA kernel call.
     * 
     * @param memory Pre-built BM25 index memory
     * @param query BM25 query (token frequencies)
     * @param indices Output top-K document indices
     * @return Execution result with timing info
     */
    ExecutionResult fused_retrieve(
        const data_type::BM25IndexMemory& memory,
        const data_type::BM25Query& query,
        data_type::TopKIndex& indices
    ) {
        if (!fused_retrieval_step_) {
            void* fpga_setup = nullptr;
            if (dataset_builder_) {
                fpga_setup = dataset_builder_->get_fpga_setup_object();
            }
            fused_retrieval_step_ = std::make_shared<step::FusedBM25Retrieval>(
                fpga_setup, python_module_path_
            );
            fused_retrieval_step_->set_kernel(step::KernelType::FPGA);
        }

        auto start = std::chrono::high_resolution_clock::now();
        int status = fused_retrieval_step_->execute(memory, query, indices, false, true);
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (status != 0) {
            return ExecutionResult::Failure("Fused retrieval execution failed");
        }

        ExecutionResult result = ExecutionResult::Success(time_ms);
        result.compute_score_time_ms = time_ms;
        result.retrieval_time_ms = 0.0;
        return result;
    }

protected:
    // ===== Factory methods — create step handlers =====

    /**
     * @brief Create BM25DatasetBuilder step
     * Builds BM25 index and initializes FPGA via Python
     */
    std::shared_ptr<BuildMemoryStepT> create_build_memory_step() override {
        auto builder = std::make_shared<step::BM25DatasetBuilder>(
            bitstream_path_, export_dir_, python_module_path_
        );
        dataset_builder_ = builder;
        return builder;
    }

    /**
     * @brief Create compute score step (not used in fused path)
     * Returns nullptr since we use fused retrieval.
     * If the standard 4-step flow is desired, override and provide a BM25
     * software scoring implementation.
     */
    std::shared_ptr<ComputeScoreStepT> create_compute_score_step() override {
        return nullptr;
    }

    /**
     * @brief Create memory retrieval step (not used in fused path)
     */
    std::shared_ptr<MemoryRetrievalStepT> create_memory_retrieval_step() override {
        return nullptr;
    }

    /**
     * @brief Create RAGApplyMemory step
     * Builds RAG prompt from retrieved documents via Python
     */
    std::shared_ptr<ApplyMemoryStepT> create_apply_memory_step() override {
        return std::make_shared<step::RAGApplyMemory>(
            python_module_path_, model_name_
        );
    }

private:
    std::string bitstream_path_;
    std::string export_dir_;
    std::string python_module_path_;
    std::string model_name_;

    // Cached step instances for fused retrieval
    std::shared_ptr<step::BM25DatasetBuilder> dataset_builder_;
    std::shared_ptr<step::FusedBM25Retrieval> fused_retrieval_step_;
};

/**
 * @brief Builder for creating configured RAG memory managers
 * 
 * Usage:
 * @code
 *   auto rag = SimpleRAGBuilder()
 *       .with_schedule_path("schedule.json")
 *       .with_bitstream("../indexer_bm25.xclbin")
 *       .with_export_dir("./export")
 *       .with_model("meta-llama/Llama-3.2-1B-Instruct")
 *       .build();
 * @endcode
 */
class SimpleRAGBuilder {
public:
    SimpleRAGBuilder() = default;

    SimpleRAGBuilder& with_schedule_path(const std::string& path) {
        schedule_path_ = path;
        return *this;
    }

    SimpleRAGBuilder& with_bitstream(const std::string& path) {
        bitstream_path_ = path;
        return *this;
    }

    SimpleRAGBuilder& with_export_dir(const std::string& dir) {
        export_dir_ = dir;
        return *this;
    }

    SimpleRAGBuilder& with_python_module_path(const std::string& path) {
        python_module_path_ = path;
        return *this;
    }

    SimpleRAGBuilder& with_model(const std::string& name) {
        model_name_ = name;
        return *this;
    }

    std::shared_ptr<SimpleRAG> build() {
        return std::make_shared<SimpleRAG>(
            schedule_path_,
            bitstream_path_,
            export_dir_,
            python_module_path_,
            model_name_
        );
    }

private:
    std::string schedule_path_;
    std::string bitstream_path_ = "../indexer_bm25.xclbin";
    std::string export_dir_ = "./export";
    std::string python_module_path_ = ".";
    std::string model_name_ = "meta-llama/Llama-3.2-1B-Instruct";
};

/**
 * @brief Helper to create a RAG memory manager builder
 */
inline SimpleRAGBuilder create_simple_rag() {
    return SimpleRAGBuilder();
}

}  // namespace deploy
}  // namespace heteromm

#endif  // HETEROMM_DEPLOY_SIMPLE_RAG_H_
