/**
 * @file memory_retrieval.h
 * @brief MemoryRetrieval step - selects top-k entries based on scores
 * 
 * This step handles selecting the most relevant memory entries
 * based on computed similarity scores.
 */

#ifndef HETEROMM_DEV_STEPS_MEMORY_RETRIEVAL_H_
#define HETEROMM_DEV_STEPS_MEMORY_RETRIEVAL_H_

#include "util.h"
#include "../types/score.h"
#include "../types/retrieved_index.h"
#include <type_traits>
#include <memory>
#include <iostream>

namespace heteromm {
namespace step {

/**
 * @brief Abstract base class for the MemoryRetrieval step
 * 
 * Data flow: Score -> MemoryRetrieval -> RetrievedIndex
 * 
 * This step takes the computed scores and selects the indices
 * of the top-k most relevant memory entries.
 * 
 * Usage:
 * @code
 *   class TopKRetrieval : public MemoryRetrieval<MyScore, MyIndices> {
 *       void run_cpu_kernel(const MyScore& s, MyIndices& idx) override {
 *           // Select top-k entries
 *       }
 *   };
 * @endcode
 */
template<typename ScoreType, typename IndexType>
class MemoryRetrieval {

static_assert(std::is_base_of<data_type::Score<typename ScoreType::content_type>, ScoreType>::value,
              "ScoreType must derive from data_type::Score");
static_assert(std::is_base_of<data_type::RetrievedIndex<typename IndexType::content_type>, IndexType>::value,
              "IndexType must derive from data_type::RetrievedIndex");

public:
    MemoryRetrieval() = default;
    virtual ~MemoryRetrieval() = default;

    int execute(
        const ScoreType& score,
        IndexType& index, // for testing, pass the ground truth
        bool run_functional_test = false,
        bool verbose = false
    ) {
        if(run_functional_test) {
            if(verbose) {
                std::clog << "[MemoryRetrieval] Running functional test kernel." << std::endl;
                std::clog << "  Score Type: " << score.type_name() << std::endl;
                std::clog << "  Index Type: " << index.type_name() << std::endl;
            }
            // deep copy index to check
            IndexType original_index = index;

            run_test_kernel(score, index);

            if(verbose) {
                std::clog << "[MemoryRetrieval] Functional test kernel completed." << std::endl;
            }
            if(index.is_equal(original_index)) {
                if(verbose) {
                    std::clog << "[MemoryRetrieval] Functional test passed: computed index matches original." << std::endl;
                }
            } else {
                if(verbose) {
                    std::clog << "[MemoryRetrieval] Functional test failed: computed index does not match original." << std::endl;
                }
                return 1; 
            }
            
            return 0;
        }

        // consider static schedule first
        switch (current_kernel_) {
            case KernelType::CPU:
                if(verbose) {
                    std::clog << "[MemoryRetrieval] Running CPU kernel." << std::endl;
                }
                run_cpu_kernel(score, index);
                if(verbose) {
                    std::clog << "[MemoryRetrieval] CPU kernel completed." << std::endl;
                }
                break;
            case KernelType::GPU:
                if(verbose) {
                    std::clog << "[MemoryRetrieval] Running GPU kernel." << std::endl;
                }
                run_gpu_kernel(score, index);
                if(verbose) {
                    std::clog << "[MemoryRetrieval] GPU kernel completed." << std::endl;
                }
                break;
            case KernelType::FPGA:
                if(verbose) {
                    std::clog << "[MemoryRetrieval] Running FPGA kernel." << std::endl;
                }
                run_fpga_kernel(score, index);
                if(verbose) {
                    std::clog << "[MemoryRetrieval] FPGA kernel completed." << std::endl;
                }
                break;
            default:
                break;
        }
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
        const ScoreType& score,
        IndexType& index
    ) = 0;

    virtual void run_gpu_kernel(
        const ScoreType& score,
        IndexType& index
    ) = 0;

    virtual void run_fpga_kernel(
        const ScoreType& score,
        IndexType& index
    ) = 0;

    virtual void run_cpu_kernel(
        const ScoreType& score,
        IndexType& index
    ) = 0;

private:
    KernelType current_kernel_ = KernelType::CPU;

};

/**
 * @brief Top-K retrieval - selects the k entries with highest scores
 * 
 * Returns indices of the k highest scoring entries.
 */
class TopKRetrieval : public MemoryRetrieval<
    data_type::VectorScore<float>,
    data_type::TopKIndex
> {
public:
    explicit TopKRetrieval(size_t k) : k_(k) {}
    ~TopKRetrieval() override = default;

    size_t k() const { return k_; }
    void set_k(size_t k) { k_ = k; }

protected:
    void run_test_kernel(
        const data_type::VectorScore<float>& score,
        data_type::TopKIndex& index
    ) override;

    void run_cpu_kernel(
        const data_type::VectorScore<float>& score,
        data_type::TopKIndex& index
    ) override;

    void run_gpu_kernel(
        const data_type::VectorScore<float>& score,
        data_type::TopKIndex& index
    ) override;

    void run_fpga_kernel(
        const data_type::VectorScore<float>& score,
        data_type::TopKIndex& index
    ) override;

private:
    size_t k_;
};

/**
 * @brief Threshold retrieval - selects entries with scores above a threshold
 * 
 * Returns a bitmap indicating which entries exceed the threshold.
 */
class ThresholdRetrieval : public MemoryRetrieval<
    data_type::VectorScore<float>,
    data_type::ThresholdBitmapIndex
> {
public:
    explicit ThresholdRetrieval(float threshold) : threshold_(threshold) {}
    ~ThresholdRetrieval() override = default;

    float threshold() const { return threshold_; }
    void set_threshold(float threshold) { threshold_ = threshold; }

protected:
    void run_test_kernel(
        const data_type::VectorScore<float>& score,
        data_type::ThresholdBitmapIndex& index
    ) override;

    void run_cpu_kernel(
        const data_type::VectorScore<float>& score,
        data_type::ThresholdBitmapIndex& index
    ) override;

    void run_gpu_kernel(
        const data_type::VectorScore<float>& score,
        data_type::ThresholdBitmapIndex& index
    ) override;

    void run_fpga_kernel(
        const data_type::VectorScore<float>& score,
        data_type::ThresholdBitmapIndex& index
    ) override;

private:
    float threshold_;
};

}  // namespace step
}  // namespace heteromm

#endif  // HETEROMM_DEV_STEPS_MEMORY_RETRIEVAL_H_
