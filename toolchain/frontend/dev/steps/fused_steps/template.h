#ifndef HETEROMM_DEV_FUSED_STEPS_TEMPLATE_H_
#define HETEROMM_DEV_FUSED_STEPS_TEMPLATE_H_

#include "../util.h"
#include "../../types/memory.h"
#include "../../types/query.h"
#include "../../types/score.h"
#include "../../types/retrieved_index.h"
#include "../../types/retrieved_data.h"
#include "../../types/target_data.h"
#include <type_traits>
#include <memory>
#include <string>
#include <tuple>
#include <assert.h>
#include <iostream>

namespace heteromm {
namespace step {

template<typename MemoryType, typename QueryType, typename IndexType>
class FusedComputeScoreAndRetrieval {
static_assert(std::is_base_of<data_type::Memory<typename MemoryType::content_type>, MemoryType>::value,
              "MemoryType must derive from data_type::Memory");
static_assert(std::is_base_of<data_type::Query<typename QueryType::content_type>, QueryType>::value,
              "QueryType must derive from data_type::Query");
static_assert(std::is_base_of<data_type::RetrievedIndex<typename IndexType::content_type>, IndexType>::value,
              "IndexType must derive from data_type::RetrievedIndex");

public:
    FusedComputeScoreAndRetrieval() = default;
    virtual ~FusedComputeScoreAndRetrieval() = default;

    int execute(
        const MemoryType& memory,
        const QueryType& query,
        IndexType& index, // for testing, pass the ground truth
        bool run_functional_test = false,
        bool verbose = false
    ) {
        if(run_functional_test) {
            if(verbose) {
                std::clog << "[FusedComputeScoreAndRetrieval] Running functional test kernel." << std::endl;
                std::clog << "  Memory Type: " << memory.type_name() << std::endl;
                std::clog << "  Query Type: " << query.type_name() << std::endl;
                std::clog << "  Index Type: " << index.type_name() << std::endl;
            }

            IndexType original_index = index;

            run_test_kernel(memory, query, index);
            
            if(!index.is_equal(original_index)) {
                std::clog << "[FusedComputeScoreAndRetrieval] Functional test failed: output index does not match expected." << std::endl;
                return 1;
            } else {
                if(verbose) {
                    std::clog << "[FusedComputeScoreAndRetrieval] Functional test passed." << std::endl;
                }
            }
            return 0;
        }

        // consider static schedule first
        switch (current_kernel_) {
            case KernelType::CPU:
                if(verbose) {
                    std::clog << "[FusedComputeScoreAndRetrieval] Running CPU kernel." << std::endl;
                }
                run_cpu_kernel(memory, query, index);
                if(verbose) {
                    std::clog << "[FusedComputeScoreAndRetrieval] CPU kernel completed." << std::endl;
                }
                break;
            case KernelType::GPU:
                if(verbose) {
                    std::clog << "[FusedComputeScoreAndRetrieval] Running GPU kernel." << std::endl;
                }
                run_gpu_kernel(memory, query, index);
                if(verbose) {
                    std::clog << "[FusedComputeScoreAndRetrieval] GPU kernel completed." << std::endl;
                }
                break;
            case KernelType::FPGA:
                if(verbose) {
                    std::clog << "[FusedComputeScoreAndRetrieval] Running FPGA kernel." << std::endl;
                }
                run_fpga_kernel(memory, query, index);
                if(verbose) {
                    std::clog << "[FusedComputeScoreAndRetrieval] FPGA kernel completed." << std::endl;
                }
                break;
            default:
                break;
        }
        return 0;
    }

    void set_kernel(KernelType kernel) {
        current_kernel_ = kernel;
    }

    KernelType get_kernel() const {
        return current_kernel_;
    }

protected:
    virtual void run_test_kernel(
        const MemoryType& memory,
        const QueryType& query,
        IndexType& index
    ) = 0;  

    virtual void run_cpu_kernel(
        const MemoryType& memory,
        const QueryType& query,
        IndexType& index
    ) = 0;

    virtual void run_gpu_kernel(
        const MemoryType& memory,
        const QueryType& query,
        IndexType& index
    ) = 0;

    virtual void run_fpga_kernel(
        const MemoryType& memory,
        const QueryType& query,
        IndexType& index
    ) = 0;

private:
    KernelType current_kernel_ = KernelType::CPU;

};

/**
 * @brief Fused BM25 score computation and top-K retrieval on FPGA
 * 
 * This step takes the pre-built BM25 index memory and a BM25 query,
 * then runs the fused FPGA kernel that computes BM25 scores for all
 * documents and returns the top-K document indices in a single pass.
 * 
 * The FPGA kernel (indexer_top) performs:
 * - Document frequency lookup
 * - BM25 score computation across all documents (4-channel parallel)
 * - Top-K selection via merge-sort network
 * 
 * The run_fpga_kernel implementation calls launch_bm25.fpga_retriver_launch
 * via the embedded Python interpreter.
 * 
 * Data flow: (BM25IndexMemory, BM25Query) -> FusedBM25Retrieval -> TopKIndex
 */
class FusedBM25Retrieval : public FusedComputeScoreAndRetrieval<
    data_type::BM25IndexMemory,
    data_type::BM25Query,
    data_type::TopKIndex
> {
public:
    FusedBM25Retrieval(
        void* fpga_setup_object = nullptr,
        const std::string& python_module_path = "."
    ) : fpga_setup_object_(fpga_setup_object),
        python_module_path_(python_module_path) {}

    ~FusedBM25Retrieval() override;

    void set_fpga_setup(void* fpga_setup) {
        fpga_setup_object_ = fpga_setup;
    }

protected:
    void run_fpga_kernel(
        const data_type::BM25IndexMemory& memory,
        const data_type::BM25Query& query,
        data_type::TopKIndex& index
    ) override;

    void run_cpu_kernel(
        const data_type::BM25IndexMemory& memory,
        const data_type::BM25Query& query,
        data_type::TopKIndex& index
    ) override;

    void run_gpu_kernel(
        const data_type::BM25IndexMemory& memory,
        const data_type::BM25Query& query,
        data_type::TopKIndex& index
    ) override;

    void run_test_kernel(
        const data_type::BM25IndexMemory& memory,
        const data_type::BM25Query& query,
        data_type::TopKIndex& index
    ) override;

private:
    void* fpga_setup_object_;
    std::string python_module_path_;
};

}
}

#endif // HETEROMM_DEV_FUSED_STEPS_TEMPLATE_H_