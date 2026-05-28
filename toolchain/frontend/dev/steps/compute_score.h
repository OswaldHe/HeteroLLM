/**
 * @file compute_score.h
 * @brief ComputeScore step - computes similarity scores between query and memory
 * 
 * This step handles computing similarity or relevance scores between
 * query vectors and memory entries.
 */

#ifndef HETEROMM_DEV_STEPS_COMPUTE_SCORE_H_
#define HETEROMM_DEV_STEPS_COMPUTE_SCORE_H_

#include "util.h"
#include "../types/memory.h"
#include "../types/query.h"
#include "../types/score.h"
#include <type_traits>
#include <memory>
#include <tuple>
#include <assert.h>
#include <iostream>

namespace heteromm {
namespace step {

/**
 * @brief Abstract base class for the ComputeScore step
 * 
 * Data flow: (Memory, Query) -> ComputeScore -> Score
 * 
 * This step computes similarity scores between query vectors and
 * entries in the memory structure. The scores are used to identify
 * the most relevant memory entries.
 * 
 * Usage:
 * @code
 *   class DotProductScore : public ComputeScoreStep<MyIndex, MyQuery, MyScore> {
 *       StepStatus run_cpu_kernel(const MyIndex& mem, const MyQuery& q, MyScore& s) override {
 *           // Compute dot products
 *       }
 *   };
 * @endcode
 */
template<typename MemoryType, typename QueryType, typename ScoreType>
class ComputeScore {

static_assert(std::is_base_of<data_type::Memory<typename MemoryType::content_type>, MemoryType>::value,
              "MemoryType must derive from data_type::Memory");
static_assert(std::is_base_of<data_type::Query<typename QueryType::content_type>, QueryType>::value,
              "QueryType must derive from data_type::Query");
static_assert(std::is_base_of<data_type::Score<typename ScoreType::content_type>, ScoreType>::value,
              "ScoreType must derive from data_type::Score");

public:
    ComputeScore() = default;
    virtual ~ComputeScore() = default;

    int execute(
        const MemoryType& memory,
        const QueryType& query,
        ScoreType& score, // for testing, pass the ground truth
        bool run_functional_test = false,
        bool verbose = false
    ) {
        if(run_functional_test) {
            if(verbose) {
                std::clog << "[ComputeScore] Running functional test kernel." << std::endl;
                std::clog << "  Memory Type: " << memory.type_name() << std::endl;
                std::clog << "  Query Type: " << query.type_name() << std::endl;
                std::clog << "  Score Type: " << score.type_name() << std::endl;
            }
            // deep copy score to check
            ScoreType original_score = score;

            run_test_kernel(memory, query, score);

            if(verbose) {
                std::clog << "[ComputeScore] Functional test kernel completed." << std::endl;
            }
            if(score.is_equal(original_score)) {
                if(verbose) {
                    std::clog << "[ComputeScore] Functional test passed: computed score matches original." << std::endl;
                }
            } else {
                if(verbose) {
                    std::clog << "[ComputeScore] Functional test failed: computed score does not match original." << std::endl;
                }
                return 1; 
            }
            
            return 0;
        }

        // consider static schedule first
        switch (current_kernel_) {
            case KernelType::CPU:
                if(verbose) {
                    std::clog << "[ComputeScore] Running CPU kernel." << std::endl;
                }
                run_cpu_kernel(memory, query, score);
                if(verbose) {
                    std::clog << "[ComputeScore] CPU kernel completed." << std::endl;
                }
                break;
            case KernelType::GPU:
                if(verbose) {
                    std::clog << "[ComputeScore] Running GPU kernel." << std::endl;
                }
                run_gpu_kernel(memory, query, score);
                if(verbose) {
                    std::clog << "[ComputeScore] GPU kernel completed." << std::endl;
                }
                break;
            case KernelType::FPGA:
                if(verbose) {
                    std::clog << "[ComputeScore] Running FPGA kernel." << std::endl;
                }
                run_fpga_kernel(memory, query, score);
                if(verbose) {
                    std::clog << "[ComputeScore] FPGA kernel completed." << std::endl;
                }
                break;
            default:
                break;
            }
        // dynamic schedule: call backend
        return 0;
    }

    KernelType current_kernel() const {
        return current_kernel_;
    }

    void set_current_kernel(KernelType type) {
        current_kernel_ = type;
    }

protected:
    virtual void run_test_kernel(
        const MemoryType& memory,
        const QueryType& query,
        ScoreType& score
    ) = 0;

    virtual void run_gpu_kernel(
        const MemoryType& memory,
        const QueryType& query,
        ScoreType& score
    ) = 0;

    virtual void run_fpga_kernel(
        const MemoryType& memory,
        const QueryType& query,
        ScoreType& score
    ) = 0;

    virtual void run_cpu_kernel(
        const MemoryType& memory,
        const QueryType& query,
        ScoreType& score
    ) = 0;

private:
    KernelType current_kernel_ = KernelType::CPU;

};

/**
 * @brief Inner product (dot product) computation between query vectors and memory
 * 
 * Computes dot product similarity scores between a VectorQuery and FlatIndexMemory.
 */
class InnerProductCompute : public ComputeScore<
    data_type::FlatIndexMemory<float>,
    data_type::VectorQuery<float>,
    data_type::VectorScore<float>
> {
public:
    InnerProductCompute() = default;
    ~InnerProductCompute() override = default;

protected:
    void run_test_kernel(
        const data_type::FlatIndexMemory<float>& memory,
        const data_type::VectorQuery<float>& query,
        data_type::VectorScore<float>& score
    ) override;

    void run_cpu_kernel(
        const data_type::FlatIndexMemory<float>& memory,
        const data_type::VectorQuery<float>& query,
        data_type::VectorScore<float>& score
    ) override;

    void run_gpu_kernel(
        const data_type::FlatIndexMemory<float>& memory,
        const data_type::VectorQuery<float>& query,
        data_type::VectorScore<float>& score
    ) override;

    void run_fpga_kernel(
        const data_type::FlatIndexMemory<float>& memory,
        const data_type::VectorQuery<float>& query,
        data_type::VectorScore<float>& score
    ) override;
};

}  // namespace step
}  // namespace heteromm

#endif  // HETEROMM_DEV_STEPS_COMPUTE_SCORE_H_
