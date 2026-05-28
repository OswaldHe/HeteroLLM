/**
 * @file memory_manager.h
 * @brief MemoryManager base class for pipeline orchestration
 * 
 * Provides the main interface for building memory, managing retrieval,
 * and applying memory in a unified workflow.
 */

#ifndef HETEROMM_DEPLOY_MEMORY_MANAGER_H_
#define HETEROMM_DEPLOY_MEMORY_MANAGER_H_

// Include step definitions from dev folder
#include "../dev/steps/util.h"
#include "../dev/steps/build_memory.h"
#include "../dev/steps/compute_score.h"
#include "../dev/steps/memory_retrieval.h"
#include "../dev/steps/apply_memory.h"

// Include type definitions from dev folder
#include "../dev/types/base_types.h"
#include "../dev/types/memory.h"
#include "../dev/types/query.h"
#include "../dev/types/score.h"
#include "../dev/types/retrieved_index.h"
#include "../dev/types/retrieved_data.h"
#include "../dev/types/target_data.h"

#include "utils.h"

#include <memory>
#include <stdexcept>
#include <chrono>
#include <iostream>

namespace heteromm {
namespace deploy {

/**
 * @brief Configuration for kernel types across pipeline steps
 */
struct PipelineKernelConfig {
    step::KernelType build_memory = step::KernelType::CPU;
    step::KernelType compute_score = step::KernelType::CPU;
    step::KernelType memory_retrieval = step::KernelType::CPU;
    step::KernelType apply_memory = step::KernelType::CPU;
    
    PipelineKernelConfig() = default;
    
    PipelineKernelConfig(step::KernelType build, step::KernelType compute,
                         step::KernelType retrieval, step::KernelType apply)
        : build_memory(build), compute_score(compute),
          memory_retrieval(retrieval), apply_memory(apply) {}
};

/**
 * @brief Execution result from memory manager operations
 */
struct ExecutionResult {
    bool success = false;
    std::string error_message;
    double total_time_ms = 0.0;
    
    // Timing breakdown
    double build_memory_time_ms = 0.0;
    double compute_score_time_ms = 0.0;
    double retrieval_time_ms = 0.0;
    double apply_memory_time_ms = 0.0;
    
    static ExecutionResult Success(double time_ms = 0.0) {
        ExecutionResult r;
        r.success = true;
        r.total_time_ms = time_ms;
        return r;
    }
    
    static ExecutionResult Failure(const std::string& msg) {
        ExecutionResult r;
        r.success = false;
        r.error_message = msg;
        return r;
    }
};

/**
 * @brief Memory Manager base class
 * 
 * This class orchestrates the full retrieval-augmented pipeline:
 * 1. build_memory: RetrievedData -> Memory
 * 2. compute_score: (Memory, Query) -> Score
 * 3. memory_retrieval: Score -> RetrievedIndex
 * 4. apply_memory: (RetrievedData, RetrievedIndex, TargetData) -> TargetData
 * 
 * Three main entry points are provided:
 * - build_memory(): Just build the memory structure
 * - manage_memory_and_apply(): Given built memory, run query->output
 * - build_and_apply_memory(): Full pipeline from raw data to output
 * 
 * Users should create new classes inheriting from this base to:
 * - Provide custom step handlers
 * - Configure kernel types for each step
 * - Add logging, profiling, or other customizations
 * 
 * The three main functions are for use, NOT for reimplementation.
 * Customization should happen via step handlers and factory methods.
 * 
 * @tparam RetDataT Type derived from data_type::RetrievedData
 * @tparam MemoryT Type derived from data_type::Memory
 * @tparam QueryT Type derived from data_type::Query
 * @tparam ScoreT Type derived from data_type::Score
 * @tparam IndexT Type derived from data_type::RetrievedIndex
 * @tparam InputT Type derived from data_type::TargetData (input to apply_memory)
 * @tparam OutputT Type derived from data_type::TargetData (output of apply_memory)
 */
template<typename RetDataT, typename MemoryT, typename QueryT,
         typename ScoreT, typename IndexT, typename InputT, typename OutputT>
class MemoryManager {
public:
    // Type aliases for convenience
    using RetrievedData = RetDataT;
    using Memory = MemoryT;
    using Query = QueryT;
    using Score = ScoreT;
    using Index = IndexT;
    using Input = InputT;
    using Output = OutputT;
    
    // Step types using dev folder step classes
    using BuildMemoryStepT = step::BuildMemory<RetDataT, MemoryT>;
    using ComputeScoreStepT = step::ComputeScore<MemoryT, QueryT, ScoreT>;
    using MemoryRetrievalStepT = step::MemoryRetrieval<ScoreT, IndexT>;
    using ApplyMemoryStepT = step::ApplyMemory<RetDataT, IndexT, InputT, OutputT>;
    
    /**
     * @brief Default constructor with CPU kernels for all steps
     * @param schedule_path Path to the schedule JSON file (optional)
     */
    explicit MemoryManager(const std::string& schedule_path = "") 
        : kernel_config_(){
        if (!schedule_path.empty()) {
            schedule_config_ = ScheduleConfig::load_from_file(schedule_path);
        }
    }
    
    /**
     * @brief Constructor with kernel configuration
     * @param schedule_path Path to the schedule JSON file
     * @param config Kernel configuration for each pipeline step
     */
    MemoryManager(const std::string& schedule_path, const PipelineKernelConfig& config) 
        : kernel_config_(config){
        if (!schedule_path.empty()) {
            schedule_config_ = ScheduleConfig::load_from_file(schedule_path);
        }
    }
    
    virtual ~MemoryManager() = default;
    
    // ===== Kernel configuration setters =====
    
    /**
     * @brief Set the kernel configuration for all steps
     * @param config Kernel configuration
     */
    void set_kernel_config(const PipelineKernelConfig& config) {
        kernel_config_ = config;
    }
    
    /**
     * @brief Set the kernel type for build_memory step
     * @param type Kernel type (CPU, GPU, FPGA)
     */
    void set_build_memory_kernel(step::KernelType type) {
        kernel_config_.build_memory = type;
    }
    
    /**
     * @brief Set the kernel type for compute_score step
     * @param type Kernel type (CPU, GPU, FPGA)
     */
    void set_compute_score_kernel(step::KernelType type) {
        kernel_config_.compute_score = type;
    }
    
    /**
     * @brief Set the kernel type for memory_retrieval step
     * @param type Kernel type (CPU, GPU, FPGA)
     */
    void set_memory_retrieval_kernel(step::KernelType type) {
        kernel_config_.memory_retrieval = type;
    }
    
    /**
     * @brief Set the kernel type for apply_memory step
     * @param type Kernel type (CPU, GPU, FPGA)
     */
    void set_apply_memory_kernel(step::KernelType type) {
        kernel_config_.apply_memory = type;
    }
    
    /**
     * @brief Get the current kernel configuration
     * @return Current kernel configuration
     */
    const PipelineKernelConfig& get_kernel_config() const {
        return kernel_config_;
    }
    
    /**
     * @brief Get the schedule configuration
     * @return Schedule configuration
     */
    const ScheduleConfig& get_schedule_config() const {
        return schedule_config_;
    }
    
    // ===== Step handler setters (for customization) =====
    
    /**
     * @brief Set the build memory step handler
     */
    void set_build_memory_step(std::shared_ptr<BuildMemoryStepT> step) {
        build_memory_step_ = std::move(step);
    }
    
    /**
     * @brief Set the compute score step handler
     */
    void set_compute_score_step(std::shared_ptr<ComputeScoreStepT> step) {
        compute_score_step_ = std::move(step);
    }
    
    /**
     * @brief Set the memory retrieval step handler
     */
    void set_memory_retrieval_step(std::shared_ptr<MemoryRetrievalStepT> step) {
        memory_retrieval_step_ = std::move(step);
    }
    
    /**
     * @brief Set the apply memory step handler
     */
    void set_apply_memory_step(std::shared_ptr<ApplyMemoryStepT> step) {
        apply_memory_step_ = std::move(step);
    }
    
    // ===== Factory methods (override to provide custom handlers) =====
    
    /**
     * @brief Create a build memory step handler
     * Override this to provide a custom implementation.
     */
    virtual std::shared_ptr<BuildMemoryStepT> create_build_memory_step() {
        return nullptr;  // Must be overridden or set manually
    }
    
    /**
     * @brief Create a compute score step handler
     */
    virtual std::shared_ptr<ComputeScoreStepT> create_compute_score_step() {
        return nullptr;
    }
    
    /**
     * @brief Create a memory retrieval step handler
     */
    virtual std::shared_ptr<MemoryRetrievalStepT> create_memory_retrieval_step() {
        return nullptr;
    }
    
    /**
     * @brief Create an apply memory step handler
     */
    virtual std::shared_ptr<ApplyMemoryStepT> create_apply_memory_step() {
        return nullptr;
    }

    virtual int ret_data_size(RetDataT& retrieved_data) = 0;
    virtual int memory_size(MemoryT& memory) = 0;
    
    // ===== Main API functions (NOT for reimplementation) =====
    
    /**
     * @brief Build memory structure from retrieved data
     * 
     * This function handles only the build_memory step.
     * Use this when memory building is a separate, potentially
     * offline process.
     * 
     * Note: Kernel type should be configured via set_build_memory_kernel()
     * or set_kernel_config() before calling this function.
     * 
     * @param retrieved_data Input retrieved data
     * @param memory Output memory structure
     * @param verbose Enable verbose logging
     * @return Execution result
     */
    ExecutionResult build_memory(
            const RetDataT& retrieved_data,
            MemoryT& memory,
            bool verbose = false) {

        int ret_data_size_val = ret_data_size(retrieved_data);
        int memory_size_val = memory_size(memory);
        auto config = schedule_config_.get_build_memory_config(ret_data_size_val, memory_size_val);
        set_build_memory_kernel(step::string_to_kernel_type(config));
        
        auto step = get_or_create_build_memory_step();
        if (!step) {
            return ExecutionResult::Failure("BuildMemory step not configured");
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        int status = step->execute(retrieved_data, memory, false, verbose);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        if (status != 0) {
            return ExecutionResult::Failure("BuildMemory execution failed");
        }
        
        ExecutionResult result = ExecutionResult::Success(time_ms);
        result.build_memory_time_ms = time_ms;
        return result;
    }
    
    /**
     * @brief Manage memory and apply to produce output
     * 
     * This function handles:
     * - compute_score: (memory, query) -> scores
     * - memory_retrieval: scores -> indices
     * - apply_memory: (retrieved_data, indices, input) -> output
     * 
     * Use this when memory is already built.
     * 
     * Note: Kernel types should be configured via set_*_kernel()
     * or set_kernel_config() before calling this function.
     * 
     * @param retrieved_data Original retrieved data (for apply_memory)
     * @param memory Pre-built memory structure
     * @param query Query to search for
     * @param input Additional input data
     * @param output Output result
     * @param verbose Enable verbose logging
     * @return Execution result
     */
    ExecutionResult manage_memory_and_apply(
            const RetDataT& retrieved_data,
            const MemoryT& memory,
            const QueryT& query,
            const InputT& input,
            OutputT& output,
            bool verbose = false) {
        
        ExecutionResult result;
        result.success = true;

        // get config
        int ret_data_size_val = ret_data_size(retrieved_data);
        int memory_size_val = memory_size(memory);
        auto config = schedule_config_.get_manage_memory_and_apply_config(ret_data_size_val, memory_size_val);
        set_compute_score_kernel(step::string_to_kernel_type(config[0]));
        set_memory_retrieval_kernel(step::string_to_kernel_type(config[1]));
        set_apply_memory_kernel(step::string_to_kernel_type(config[2]));
        
        // Step 1: Compute scores
        auto score_step = get_or_create_compute_score_step();
        if (!score_step) {
            return ExecutionResult::Failure("ComputeScore step not configured");
        }
        
        ScoreT scores;
        auto score_start = std::chrono::high_resolution_clock::now();
        int score_status = score_step->execute(memory, query, scores, false, verbose);
        auto score_end = std::chrono::high_resolution_clock::now();
        
        if (score_status != 0) {
            return ExecutionResult::Failure("ComputeScore execution failed");
        }
        result.compute_score_time_ms = std::chrono::duration<double, std::milli>(
            score_end - score_start).count();
        
        // Step 2: Memory retrieval
        auto retrieval_step = get_or_create_memory_retrieval_step();
        if (!retrieval_step) {
            return ExecutionResult::Failure("MemoryRetrieval step not configured");
        }
        
        IndexT indices;
        auto retr_start = std::chrono::high_resolution_clock::now();
        int retr_status = retrieval_step->execute(scores, indices, false, verbose);
        auto retr_end = std::chrono::high_resolution_clock::now();
        
        if (retr_status != 0) {
            return ExecutionResult::Failure("MemoryRetrieval execution failed");
        }
        result.retrieval_time_ms = std::chrono::duration<double, std::milli>(
            retr_end - retr_start).count();
        
        // Step 3: Apply memory
        auto apply_step = get_or_create_apply_memory_step();
        if (!apply_step) {
            return ExecutionResult::Failure("ApplyMemory step not configured");
        }
        
        auto apply_start = std::chrono::high_resolution_clock::now();
        int apply_status = apply_step->execute(retrieved_data, indices, input, output, false, verbose);
        auto apply_end = std::chrono::high_resolution_clock::now();
        
        if (apply_status != 0) {
            return ExecutionResult::Failure("ApplyMemory execution failed");
        }
        result.apply_memory_time_ms = std::chrono::duration<double, std::milli>(
            apply_end - apply_start).count();
        
        result.total_time_ms = result.compute_score_time_ms + 
                               result.retrieval_time_ms + 
                               result.apply_memory_time_ms;
        
        return result;
    }
    
    /**
     * @brief Build memory and apply in one operation
     * 
     * This function handles the complete pipeline:
     * - build_memory: retrieved_data -> memory
     * - compute_score: (memory, query) -> scores
     * - memory_retrieval: scores -> indices
     * - apply_memory: (retrieved_data, indices, input) -> output
     * 
     * Note: Kernel types should be configured via set_*_kernel()
     * or set_kernel_config() before calling this function.
     * 
     * @param retrieved_data Input retrieved data
     * @param query Query to search for
     * @param input Additional input data
     * @param output Output result
     * @param verbose Enable verbose logging
     * @return Execution result
     */
    ExecutionResult build_and_apply_memory(
            const RetDataT& retrieved_data,
            const QueryT& query,
            const InputT& input,
            OutputT& output,
            bool verbose = false) {

        int ret_data_size_val = ret_data_size(retrieved_data);
        int memory_size_val = memory_size(memory);
        auto config = schedule_config_.get_build_memory_and_apply_config(ret_data_size_val, memory_size_val);
        set_build_memory_kernel(step::string_to_kernel_type(config[0]));
        set_compute_score_kernel(step::string_to_kernel_type(config[1]));
        set_memory_retrieval_kernel(step::string_to_kernel_type(config[2]));
        set_apply_memory_kernel(step::string_to_kernel_type(config[3]));
        
        // Build memory
        MemoryT memory;
        auto build_result = build_memory(retrieved_data, memory, verbose);
        if (!build_result.success) {
            return build_result;
        }
        
        // Run remaining steps
        auto apply_result = manage_memory_and_apply(
            retrieved_data, memory, query, input, output, verbose);
        
        if (!apply_result.success) {
            return apply_result;
        }
        
        // Combine results
        ExecutionResult result = apply_result;
        result.build_memory_time_ms = build_result.build_memory_time_ms;
        result.total_time_ms = build_result.total_time_ms + apply_result.total_time_ms;
        
        return result;
    }
    
    /**
     * @brief Run functional tests on all configured steps
     * 
     * Tests each step using its test kernel with ground truth data.
     * 
     * @param retrieved_data Test input data
     * @param expected_memory Expected memory output
     * @param query Test query
     * @param expected_scores Expected scores output
     * @param expected_indices Expected indices output
     * @param input Test input for apply_memory
     * @param expected_output Expected final output
     * @param verbose Enable verbose logging
     * @return True if all tests pass
     */
    bool run_functional_tests(
            const RetDataT& retrieved_data,
            MemoryT& expected_memory,
            const QueryT& query,
            ScoreT& expected_scores,
            IndexT& expected_indices,
            const InputT& input,
            OutputT& expected_output,
            bool verbose = false) {
        
        bool all_passed = true;
        
        // Test build_memory
        auto build_step = get_or_create_build_memory_step();
        if (build_step) {
            int status = build_step->execute(retrieved_data, expected_memory, true, verbose);
            if (status != 0) {
                if (verbose) std::clog << "[MemoryManager] BuildMemory test FAILED" << std::endl;
                all_passed = false;
            } else if (verbose) {
                std::clog << "[MemoryManager] BuildMemory test PASSED" << std::endl;
            }
        }
        
        // Test compute_score
        auto score_step = get_or_create_compute_score_step();
        if (score_step) {
            int status = score_step->execute(expected_memory, query, expected_scores, true, verbose);
            if (status != 0) {
                if (verbose) std::clog << "[MemoryManager] ComputeScore test FAILED" << std::endl;
                all_passed = false;
            } else if (verbose) {
                std::clog << "[MemoryManager] ComputeScore test PASSED" << std::endl;
            }
        }
        
        // Test memory_retrieval
        auto retrieval_step = get_or_create_memory_retrieval_step();
        if (retrieval_step) {
            int status = retrieval_step->execute(expected_scores, expected_indices, true, verbose);
            if (status != 0) {
                if (verbose) std::clog << "[MemoryManager] MemoryRetrieval test FAILED" << std::endl;
                all_passed = false;
            } else if (verbose) {
                std::clog << "[MemoryManager] MemoryRetrieval test PASSED" << std::endl;
            }
        }
        
        // Test apply_memory
        auto apply_step = get_or_create_apply_memory_step();
        if (apply_step) {
            int status = apply_step->execute(retrieved_data, expected_indices, input, 
                                              expected_output, true, verbose);
            if (status != 0) {
                if (verbose) std::clog << "[MemoryManager] ApplyMemory test FAILED" << std::endl;
                all_passed = false;
            } else if (verbose) {
                std::clog << "[MemoryManager] ApplyMemory test PASSED" << std::endl;
            }
        }
        
        return all_passed;
    }

protected:
    // Step handlers
    std::shared_ptr<BuildMemoryStepT> build_memory_step_;
    std::shared_ptr<ComputeScoreStepT> compute_score_step_;
    std::shared_ptr<MemoryRetrievalStepT> memory_retrieval_step_;
    std::shared_ptr<ApplyMemoryStepT> apply_memory_step_;
    
    // Kernel configuration
    PipelineKernelConfig kernel_config_;
    
    // Schedule configuration
    ScheduleConfig schedule_config_;
    
    // Helper to get or create step handlers and apply kernel configuration
    std::shared_ptr<BuildMemoryStepT> get_or_create_build_memory_step() {
        if (!build_memory_step_) {
            build_memory_step_ = create_build_memory_step();
        }
        if (build_memory_step_) {
            build_memory_step_->set_current_kernel(kernel_config_.build_memory);
        }
        return build_memory_step_;
    }
    
    std::shared_ptr<ComputeScoreStepT> get_or_create_compute_score_step() {
        if (!compute_score_step_) {
            compute_score_step_ = create_compute_score_step();
        }
        if (compute_score_step_) {
            compute_score_step_->set_current_kernel(kernel_config_.compute_score);
        }
        return compute_score_step_;
    }
    
    std::shared_ptr<MemoryRetrievalStepT> get_or_create_memory_retrieval_step() {
        if (!memory_retrieval_step_) {
            memory_retrieval_step_ = create_memory_retrieval_step();
        }
        if (memory_retrieval_step_) {
            memory_retrieval_step_->set_current_kernel(kernel_config_.memory_retrieval);
        }
        return memory_retrieval_step_;
    }
    
    std::shared_ptr<ApplyMemoryStepT> get_or_create_apply_memory_step() {
        if (!apply_memory_step_) {
            apply_memory_step_ = create_apply_memory_step();
        }
        if (apply_memory_step_) {
            apply_memory_step_->set_current_kernel(kernel_config_.apply_memory);
        }
        return apply_memory_step_;
    }
};

}  // namespace deploy
}  // namespace heteromm

#endif  // HETEROMM_DEPLOY_MEMORY_MANAGER_H_
