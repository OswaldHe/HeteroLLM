/**
 * @file retrieved_index.h
 * @brief RetrievedIndex type definition - output of memory_retrieval, input to apply_memory
 * 
 * RetrievedIndex represents the indices of memory entries selected
 * based on their scores, typically the top-k most relevant entries.
 */

#ifndef HETEROMM_DEV_TYPES_RETRIEVED_INDEX_H_
#define HETEROMM_DEV_TYPES_RETRIEVED_INDEX_H_

#include "base_types.h"
#include <vector>
#include <cstring>

namespace heteromm {
namespace data_type {

/**
 * @brief Base class for retrieved index data
 * 
 * RetrievedIndex contains the indices of memory entries that were
 * selected during the retrieval step. These indices are used to
 * gather the actual data for the apply_memory step.
 * 
 * Usage:
 *   class MyTopKIndices : public RetrievedIndex { ... };
 */
template<typename T>
class RetrievedIndex : public BaseType<T> {
public:
    RetrievedIndex() = default;
    virtual ~RetrievedIndex() = default;
    
    std::string type_name() const override { return "RetrievedIndex"; }

    virtual bool is_equal(const RetrievedIndex<T>& other) = 0;
};

/**
 * @brief Top-K indices with scores implementation
 */
class TopKIndex : public RetrievedIndex<std::vector<int>> {
public:
    using content_type = std::vector<int>;

    TopKIndex() = default;
    TopKIndex(const std::vector<int>& indices) {
        this->data = indices;
        k_ = indices.size();
    }
    
    std::string type_name() const override { return "TopKIndex"; }

    size_t get_size() const override {
        return k_;
    }

    bool is_equal(const RetrievedIndex<std::vector<int>>& other) override {
        const auto& other_data = other.export_data();
        if (this->data.size() != other_data.size()) {
            return false;
        }
        for (size_t i = 0; i < this->data.size(); ++i) {
            if (this->data[i] != other_data[i]) {
                return false;
            }
        }
        return true;
    }

private:
    size_t k_ = 0;
};


/**
 * @brief bitmap of selected indices. used in SeerAttention threshold selection
 */
class ThresholdBitmapIndex : public RetrievedIndex<std::vector<unsigned long long>> {
public:
    using content_type = std::vector<unsigned long long>;

    ThresholdBitmapIndex() = default;
    ThresholdBitmapIndex(const std::vector<unsigned long long>& bitmap, size_t total_len) {
        this->data = bitmap;
        total_length_ = total_len;
        selected_size_ = 0;
        selected_indices_ = {};
        // Calculate selected indices and size
        for (size_t i = 0; i < total_length_; ++i) {
            size_t word_index = i / 64;
            size_t bit_index = i % 64;
            if (word_index < bitmap.size()) {
                if (bitmap[word_index] & (1ULL << bit_index)) {
                    selected_indices_.push_back(i);
                    selected_size_++;
                }
            }
        }
    }
    
    std::string type_name() const override { return "ThresholdBitmapIndex"; }

    size_t get_size() const override {
        return selected_size_;
    }

    const std::vector<int>& get_selected_indices() const {
        return selected_indices_;
    }

    size_t get_total_length() const {
        return total_length_;
    }

    bool is_equal(const RetrievedIndex<std::vector<unsigned long long>>& other) override {
        const auto& other_data = other.export_data();
        if (this->data.size() != other_data.size()) {
            return false;
        }
        for (size_t i = 0; i < this->data.size(); ++i) {
            if (this->data[i] != other_data[i]) {
                return false;
            }
        }
        return true;
    }

private:
    size_t selected_size_ = 0;
    size_t total_length_ = 0;
    std::vector<int> selected_indices_;
};

}  // namespace data_type
}  // namespace heteromm

#endif  // HETEROMM_DEV_TYPES_RETRIEVED_INDEX_H_
