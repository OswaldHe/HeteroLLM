/**
 * @file build_memory.h
 * @brief BuildMemory step - converts RawData to Memory
 * 
 * This step handles the construction of memory structures from raw input data.
 * Examples include building vector indices, creating hash tables, or
 * quantizing embeddings.
 */

#ifndef HETEROMM_DEV_STEPS_BUILD_MEMORY_H_
#define HETEROMM_DEV_STEPS_BUILD_MEMORY_H_

#include "util.h"
#include "../types/retrieved_data.h"
#include "../types/memory.h"
#include <type_traits>
#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include <assert.h>
#include <iostream>

namespace heteromm {
namespace step {

/**
 * @brief Abstract base class for the BuildMemory step
 * 
 * Data flow: RawData -> BuildMemory -> Memory
 * 
 * This step processes raw input data and constructs an indexed
 * memory structure suitable for efficient retrieval operations.
 * 
 * Usage:
 * @code
 *   class MyIndexBuilder : public BuildMemoryStep<MyRawData, MyIndex> {
 *       StepStatus run_cpu_kernel(const MyRawData& raw, MyIndex& memory) override {
 *           // Build index on CPU/GPU
 *       }
 *   };
 * @endcode
 */
template<typename RetDataType, typename MemoryType>
class BuildMemory {
static_assert(std::is_base_of<data_type::RetrievedData<typename RetDataType::content_type>, RetDataType>::value,
              "RetDataType must derive from data_type::RetrievedData");
static_assert(std::is_base_of<data_type::Memory<typename MemoryType::content_type>, MemoryType>::value,
              "MemoryType must derive from data_type::Memory");

public:
    BuildMemory() = default;
    virtual ~BuildMemory() = default;

    int execute(
        const RetDataType& raw_data,
        MemoryType& memory, // for testing, pass the ground truth
        bool run_functional_test = false,
        bool verbose = false
    ) {
        if(run_functional_test) {
            if(verbose) {
                std::clog << "[BuildMemory] Running functional test kernel." << std::endl;
                std::clog << "  RawData Type: " << raw_data.type_name() << std::endl;
                std::clog << "  Memory Type: " << memory.type_name() << std::endl;
            }

            MemoryType original_memory = memory;

            run_test_kernel(raw_data, memory);
            
            if(!memory.is_equal(original_memory)) {
                std::clog << "[BuildMemory] Functional test failed: output memory does not match expected." << std::endl;
                return 1;
            } else {
                if(verbose) {
                    std::clog << "[BuildMemory] Functional test passed." << std::endl;
                }
            }
            return 0;
        }

        // consider static schedule first
        switch (current_kernel_) {
            case KernelType::CPU:
                if(verbose) {
                    std::clog << "[BuildMemory] Running CPU kernel." << std::endl;
                }
                run_cpu_kernel(raw_data, memory);
                if(verbose) {
                    std::clog << "[BuildMemory] CPU kernel completed." << std::endl;
                }
                break;
            case KernelType::GPU:
                if(verbose) {
                    std::clog << "[BuildMemory] Running GPU kernel." << std::endl;
                }
                run_gpu_kernel(raw_data, memory);
                if(verbose) {
                    std::clog << "[BuildMemory] GPU kernel completed." << std::endl;
                }
                break;
            case KernelType::FPGA:
                if(verbose) {
                    std::clog << "[BuildMemory] Running FPGA kernel." << std::endl;
                }
                run_fpga_kernel(raw_data, memory);
                if(verbose) {
                    std::clog << "[BuildMemory] FPGA kernel completed." << std::endl;
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
        const RetDataType& raw_data,
        MemoryType& memory
    ) = 0;

    virtual void run_cpu_kernel(
        const RetDataType& raw_data,
        MemoryType& memory
    ) = 0;

    virtual void run_gpu_kernel(
        const RetDataType& raw_data,
        MemoryType& memory
    ) = 0;

    virtual void run_fpga_kernel(
        const RetDataType& raw_data,
        MemoryType& memory
    ) = 0;

private:
    KernelType current_kernel_ = KernelType::CPU;      
};

class PagedKVIndexBuilder : public BuildMemory<data_type::KVCacheData<float>, data_type::FlatIndexMemory<float>> {
public:
    PagedKVIndexBuilder() = default;
    PagedKVIndexBuilder(const std::vector<std::vector<float>>& weight, int page_size)
        : weight_(weight), page_size_(page_size) {}
    ~PagedKVIndexBuilder() override = default;
protected:
    void run_test_kernel(
        const data_type::KVCacheData<float>& raw_data,
        data_type::FlatIndexMemory<float>& memory
    ) override;

    void run_cpu_kernel(
        const data_type::KVCacheData<float>& raw_data,
        data_type::FlatIndexMemory<float>& memory
    ) override;

    void run_gpu_kernel(
        const data_type::KVCacheData<float>& raw_data,
        data_type::FlatIndexMemory<float>& memory
    ) override;

    void run_fpga_kernel(
        const data_type::KVCacheData<float>& raw_data,
        data_type::FlatIndexMemory<float>& memory
    ) override;

private:
    std::vector<std::vector<float>> weight_;
    int page_size_;
};

/**
 * @brief BM25 Dataset Builder - builds BM25 index memory from text corpus
 * 
 * This step loads the corpus, tokenizes it, builds a BM25 index,
 * and prepares FPGA buffers for the fused retrieval step.
 * 
 * The CPU kernel delegates to Python functions:
 * - bm25_loader_xrt.load_document_frequency_mmap: loads document frequencies
 * - bm25_loader_xrt.load_term_frequencies_mmap: loads term frequencies
 * - bm25_loader_xrt.pack_documents_for_hw: packs documents for FPGA
 * - launch_bm25.fpga_retriever_setup: initializes FPGA device and buffers
 * 
 * Data flow: TextDBData -> BM25DatasetBuilder -> BM25IndexMemory
 */
class BM25DatasetBuilder : public BuildMemory<data_type::TextDBData, data_type::BM25IndexMemory> {
public:
    BM25DatasetBuilder(
        const std::string& bitstream_path = "../indexer_bm25.xclbin",
        const std::string& export_dir = "./export",
        const std::string& python_module_path = "."
    ) : bitstream_path_(bitstream_path),
        export_dir_(export_dir),
        python_module_path_(python_module_path),
        py_initialized_(false),
        fpga_setup_object_(nullptr) {}

    ~BM25DatasetBuilder() override;

    /**
     * @brief Get the stored FPGA setup Python object
     * Used by the fused retrieval step to launch the FPGA kernel.
     */
    void* get_fpga_setup_object() const { return fpga_setup_object_; }

protected:
    void run_cpu_kernel(
        const data_type::TextDBData& raw_data,
        data_type::BM25IndexMemory& memory
    ) override;

    void run_gpu_kernel(
        const data_type::TextDBData& raw_data,
        data_type::BM25IndexMemory& memory
    ) override;

    void run_fpga_kernel(
        const data_type::TextDBData& raw_data,
        data_type::BM25IndexMemory& memory
    ) override;

    void run_test_kernel(
        const data_type::TextDBData& raw_data,
        data_type::BM25IndexMemory& memory
    ) override;

private:
    void ensure_python_initialized();

    std::string bitstream_path_;
    std::string export_dir_;
    std::string python_module_path_;
    bool py_initialized_;
    void* fpga_setup_object_;
};

}  // namespace step
}  // namespace heteromm

#endif  // HETEROMM_DEV_STEPS_BUILD_MEMORY_H_
