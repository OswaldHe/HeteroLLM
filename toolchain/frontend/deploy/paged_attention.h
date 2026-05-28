/**
 * @file paged_attention.h
 * @brief Paged attention memory manager for block sparse attention
 * 
 * Provides a memory manager that connects:
 * PagedKVIndexBuilder -> InnerProductCompute -> TopKRetrieval -> BlockSparseAttention
 * 
 * This implements block-sparse attention where:
 * 1. KV cache is partitioned into pages and indexed
 * 2. Query is matched against page indices using inner product
 * 3. Top-K relevant pages are selected
 * 4. Sparse attention is computed only on selected pages
 */

#ifndef HETEROMM_DEPLOY_PAGED_ATTENTION_H_
#define HETEROMM_DEPLOY_PAGED_ATTENTION_H_

#include "memory_manager.h"
#include <cmath>

namespace heteromm {
namespace deploy {

/**
 * @brief Type alias for the paged attention memory manager
 * 
 * Data flow:
 * - RawData (RetrievedData): KVCacheData<float> - the KV cache
 * - Memory: FlatIndexMemory<float> - paged index of keys
 * - Query: VectorQuery<float> - the attention query vector
 * - Score: VectorScore<float> - similarity scores per page
 * - Index: TopKIndex - indices of top-K relevant pages
 * - Input: VectorInputOutputData<float> - query for attention computation
 * - Output: VectorInputOutputData<float> - attention output
 */
using PagedAttentionManager = MemoryManager<
    data_type::KVCacheData<float>,           // RetrievedData: KV cache
    data_type::FlatIndexMemory<float>,       // Memory: Paged key index
    data_type::VectorQuery<float>,           // Query: Attention query
    data_type::VectorScore<float>,           // Score: Page scores
    data_type::TopKIndex,                    // Index: Top-K page indices
    data_type::VectorInputOutputData<float>, // Input: Query for attention
    data_type::VectorInputOutputData<float>  // Output: Attention result
>;

/**
 * @brief Concrete paged attention manager with default step implementations
 * 
 * This class provides a ready-to-use paged attention implementation with
 * configurable parameters for page size, projection weights, and top-K.
 * 
 * Usage:
 * @code
 *   // Create manager with page_size=16, top_k=8
 *   auto weight = create_projection_weight(head_dim, proj_dim);
 *   PagedAttention attn(weight, 16, 8);
 *   
 *   // Build index from KV cache
 *   KVCacheData<float> kv_cache = ...;
 *   FlatIndexMemory<float> index;
 *   attn.build_memory(kv_cache, index);
 *   
 *   // Run attention
 *   VectorQuery<float> query = ...;
 *   VectorInputOutputData<float> input = ...;
 *   VectorInputOutputData<float> output;
 *   attn.manage_memory_and_apply(kv_cache, index, query, input, output);
 * @endcode
 */
class PagedAttention : public PagedAttentionManager {
public:
    /**
     * @brief Construct a paged attention manager
     * 
     * @param schedule_path Path to the schedule JSON file
     * @param projection_weight Weight matrix for projecting pooled keys [proj_dim x head_dim]
     * @param page_size Number of tokens per page for average pooling
     * @param top_k Number of top pages to select for sparse attention
     */
    PagedAttention(
        const std::string& schedule_path,
        const std::vector<std::vector<float>>& projection_weight,
        size_t page_size,
        size_t top_k
    ) : PagedAttentionManager(schedule_path),
        projection_weight_(projection_weight),
        page_size_(page_size),
        top_k_(top_k) {}
    
    /**
     * @brief Default constructor with no initialization
     * Must call set_parameters before use.
     * @param schedule_path Path to the schedule JSON file (optional)
     */
    explicit PagedAttention(const std::string& schedule_path = "") 
        : PagedAttentionManager(schedule_path) {}
    
    ~PagedAttention() override = default;
    
    // ===== Parameter accessors =====
    
    void set_projection_weight(const std::vector<std::vector<float>>& weight) {
        projection_weight_ = weight;
        // Reset step to pick up new weight
        build_memory_step_.reset();
    }
    
    void set_page_size(size_t page_size) {
        page_size_ = page_size;
        // Reset steps that depend on page_size
        build_memory_step_.reset();
        apply_memory_step_.reset();
    }
    
    void set_top_k(size_t top_k) {
        top_k_ = top_k;
        // Reset retrieval step
        memory_retrieval_step_.reset();
    }
    
    size_t page_size() const { return page_size_; }
    size_t top_k() const { return top_k_; }
    const std::vector<std::vector<float>>& projection_weight() const { 
        return projection_weight_; 
    }

    int ret_data_size(
        const data_type::KVCacheData<float>& retrieved_data
    ) {
        return retrieved_data.get_context_length();
    }

    int memory_size(
        const data_type::FlatIndexMemory<float>& memory
    ) {
        return memory.get_num_entries();
    }

protected:
    // ===== Factory methods - create step handlers =====
    
    /**
     * @brief Create PagedKVIndexBuilder step
     * Builds paged index from KV cache using average pooling + projection
     */
    std::shared_ptr<BuildMemoryStepT> create_build_memory_step() override {
        return std::make_shared<step::PagedKVIndexBuilder>(
            projection_weight_, 
            static_cast<int>(page_size_)
        );
    }
    
    /**
     * @brief Create InnerProductCompute step
     * Computes inner product between query and page indices
     */
    std::shared_ptr<ComputeScoreStepT> create_compute_score_step() override {
        return std::make_shared<step::InnerProductCompute>();
    }
    
    /**
     * @brief Create TopKRetrieval step
     * Selects top-K pages with highest scores
     */
    std::shared_ptr<MemoryRetrievalStepT> create_memory_retrieval_step() override {
        return std::make_shared<step::TopKRetrieval>(top_k_);
    }
    
    /**
     * @brief Create BlockSparseAttention step
     * Computes attention only on selected pages
     */
    std::shared_ptr<ApplyMemoryStepT> create_apply_memory_step() override {
        return std::make_shared<step::BlockSparseAttention>(page_size_);
    }

private:
    std::vector<std::vector<float>> projection_weight_;
    size_t page_size_ = 16;
    size_t top_k_ = 8;
};

/**
 * @brief Builder for creating configured paged attention managers
 * 
 * Provides a fluent interface for configuring a paged attention manager.
 * 
 * Usage:
 * @code
 *   auto attn = PagedAttentionBuilder()
 *       .with_schedule_path("schedule.json")
 *       .with_projection_weight(weight)
 *       .with_page_size(32)
 *       .with_top_k(16)
 *       .build();
 * @endcode
 */
class PagedAttentionBuilder {
public:
    PagedAttentionBuilder() = default;
    
    /**
     * @brief Set the schedule configuration file path
     */
    PagedAttentionBuilder& with_schedule_path(const std::string& path) {
        schedule_path_ = path;
        return *this;
    }
    
    /**
     * @brief Set the projection weight matrix
     */
    PagedAttentionBuilder& with_projection_weight(
        const std::vector<std::vector<float>>& weight
    ) {
        projection_weight_ = weight;
        return *this;
    }
    
    /**
     * @brief Set the page size (tokens per page)
     */
    PagedAttentionBuilder& with_page_size(size_t page_size) {
        page_size_ = page_size;
        return *this;
    }
    
    /**
     * @brief Set the number of top pages to select
     */
    PagedAttentionBuilder& with_top_k(size_t top_k) {
        top_k_ = top_k;
        return *this;
    }
    
    /**
     * @brief Build and return the configured manager
     */
    std::shared_ptr<PagedAttention> build() {
        return std::make_shared<PagedAttention>(
            schedule_path_,
            projection_weight_,
            page_size_,
            top_k_
        );
    }

private:
    std::string schedule_path_;
    std::vector<std::vector<float>> projection_weight_;
    size_t page_size_ = 16;
    size_t top_k_ = 8;
};

/**
 * @brief Create a paged attention builder
 */
inline PagedAttentionBuilder create_paged_attention() {
    return PagedAttentionBuilder();
}

/**
 * @brief Helper to create a random projection weight matrix
 * 
 * @param head_dim Input dimension (head dimension)
 * @param proj_dim Output dimension (projection dimension)
 * @param seed Random seed for reproducibility
 * @return Weight matrix [proj_dim x head_dim]
 */
inline std::vector<std::vector<float>> create_random_projection_weight(
    size_t head_dim,
    size_t proj_dim,
    unsigned int seed = 42
) {
    std::vector<std::vector<float>> weight(proj_dim, std::vector<float>(head_dim));
    
    // Simple LCG random number generator for reproducibility
    unsigned int state = seed;
    auto next_random = [&state]() -> float {
        state = state * 1103515245 + 12345;
        return static_cast<float>((state >> 16) & 0x7FFF) / 32767.0f - 0.5f;
    };
    
    // Xavier/Glorot initialization scale
    float scale = std::sqrt(2.0f / (head_dim + proj_dim));
    
    for (size_t i = 0; i < proj_dim; ++i) {
        for (size_t j = 0; j < head_dim; ++j) {
            weight[i][j] = next_random() * scale;
        }
    }
    
    return weight;
}

}  // namespace deploy
}  // namespace heteromm

#endif  // HETEROMM_DEPLOY_PAGED_ATTENTION_H_
