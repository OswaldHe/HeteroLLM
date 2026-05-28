/**
 * @file memory.h
 * @brief Memory type definition - output of build_memory, input to compute_score
 * 
 * Memory represents the processed/indexed data structure that enables
 * efficient similarity search or retrieval operations.
 */

#ifndef HETEROMM_DEV_TYPES_MEMORY_H_
#define HETEROMM_DEV_TYPES_MEMORY_H_

#include "base_types.h"
#include <vector>
#include <cstring>
#include <unordered_map>
#include <stdexcept>

namespace heteromm {
namespace data_type {

/**
 * @brief Base class for memory structures
 * 
 * Memory is the indexed or processed form of RawData that enables
 * efficient retrieval. Examples include vector indices, hash tables,
 * or quantized representations.
 * 
 * Usage:
 *   class MyVectorIndex : public Memory { ... };
 */
template<typename T>
class Memory : public BaseType<T> {
public:
    Memory() = default;
    virtual ~Memory() = default;
    virtual bool is_equal(const Memory<T>& other) = 0;
    
    std::string type_name() const override { return "Memory"; }
    
};

/**
 * @brief flat index memory: fixed size vector index, 2D array
 */
// we can use different scalar types for indices
template<typename S>
class FlatIndexMemory : public Memory<std::vector<std::vector<S>>> {

public:
    using content_type = std::vector<std::vector<S>>;
    
    FlatIndexMemory() = default;

    FlatIndexMemory(const std::vector<std::vector<S>>& data)
        : num_entries_(data.size()), memory_size_(data.empty() ? 0 : data[0].size()) {
        this->data = data;
    }
    
    std::string type_name() const override { return "FlatIndexMemory"; }

    bool is_equal(const Memory<std::vector<std::vector<S>>>& other) override {
        const FlatIndexMemory<S>* other_mem = dynamic_cast<const FlatIndexMemory<S>*>(&other);
        if (!other_mem) {
            return false;
        }
        if (this->num_entries_ != other_mem->num_entries_ ||
            this->memory_size_ != other_mem->memory_size_) {
            return false;
        }
        for (size_t i = 0; i < this->num_entries_; ++i) {
            for (size_t j = 0; j < this->memory_size_; ++j) {
                if (this->data[i][j] != other_mem->data[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

    size_t get_size() const override {
        return num_entries_ * memory_size_;
    }

    size_t get_num_entries() const {
        return num_entries_;
    }

    size_t get_memory_size() const {
        return memory_size_;
    }

    void insert_memory(const std::vector<S>& index_vector) {
        if (index_vector.size() != memory_size_) {
            throw std::runtime_error("Index vector size does not match memory size");
        }
        this->data.push_back(index_vector);
        num_entries_++;
    }

    void pop_last_memory() {
        if (num_entries_ == 0) {
            throw std::runtime_error("No entries to pop");
        }
        this->data.pop_back();
        num_entries_--;
    }

    void delete_memory(size_t idx) {
        if (idx >= num_entries_) {
            throw std::runtime_error("Index out of bounds");
        }
        this->data.erase(this->data.begin() + idx);
        num_entries_--;
    }

    std::vector<S> get_memory(size_t idx) const {
        if (idx >= num_entries_) {
            throw std::runtime_error("Index out of bounds");
        }
        return this->data[idx];
    }

private:
    size_t num_entries_;
    size_t memory_size_;
};

/**
 * @brief min-max index memory: used in omniserve, paged attention with min and max vector for each block
 */
template<typename S>
class MinMaxIndexMemory : public Memory<std::vector<std::vector<S>>> {
public:
    MinMaxIndexMemory() = default;

    MinMaxIndexMemory(const std::vector<std::vector<S>>& data, size_t phy_page_size)
        : num_entries_(data.size()), memory_size_(data.empty() ? 0 : data[0].size()), phy_page_size_(phy_page_size) {
        this->data = data;
    }
    
    std::string type_name() const override { return "MinMaxIndexMemory"; }

    bool is_equal(const Memory<std::vector<std::vector<S>>>& other) override {
        const MinMaxIndexMemory<S>* other_mem = dynamic_cast<const MinMaxIndexMemory<S>*>(&other);
        if (!other_mem) {
            return false;
        }
        if (this->num_entries_ != other_mem->num_entries_ ||
            this->memory_size_ != other_mem->memory_size_) {
            return false;
        }
        for (size_t i = 0; i < this->num_entries_; ++i) {
            for (size_t j = 0; j < this->memory_size_; ++j) {
                if (this->data[i][j] != other_mem->data[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

    size_t get_size() const override {
        return num_entries_ * memory_size_;
    }

    size_t get_num_entries() const {
        return num_entries_ / 2;
    }

    size_t get_num_pages() const {
        return num_entries_ / (2 * phy_page_size_);
    }

    size_t get_memory_size() const {
        return memory_size_;
    }

    void insert_memory(const std::vector<S>& min_vector, const std::vector<S>& max_vector) {
        if (min_vector.size() != memory_size_ || max_vector.size() != memory_size_) {
            throw std::runtime_error("Index vector size does not match memory size");
        }
        this->data.push_back(min_vector);
        this->data.push_back(max_vector);
        num_entries_ += 2;
    }

    void pop_last_memory() {
        if (num_entries_ == 0) {
            throw std::runtime_error("No entries to pop");
        }
        this->data.pop_back();
        this->data.pop_back();
        num_entries_ -= 2;
    }

    void delete_memory(size_t idx) {
        if (idx >= num_entries_) {
            throw std::runtime_error("Index out of bounds");
        }
        // delete both min and max vectors
        this->data.erase(this->data.begin() + idx * 2);
        this->data.erase(this->data.begin() + idx * 2);
        num_entries_ -= 2;
    }

    std::vector<std::vector<S>> get_memory(size_t idx) const {
        if (idx >= num_entries_ / 2) {
            throw std::runtime_error("Index out of bounds");
        }
        return {this->data[idx * 2], this->data[idx * 2 + 1]};
    }

    std::vector<std::vector<S>> get_page(size_t page_idx) const {
        if (page_idx >= num_entries_ / (2 * phy_page_size_)) {
            throw std::runtime_error("Page index out of bounds");
        }
        size_t start_idx = page_idx * phy_page_size_ * 2;
        std::vector<std::vector<S>> page_data;
        for (size_t i = 0; i < phy_page_size_; i++) {
            page_data.push_back(this->data[start_idx + i * 2]);
            page_data.push_back(this->data[start_idx + i * 2 + 1]);
        }
        return page_data;
    }

private:
    size_t num_entries_;
    size_t memory_size_;
    size_t phy_page_size_;
};

struct BM25IndexData {
    std::unordered_map<int, int> df_map; // token_id -> document frequency
    std::vector<std::unordered_map<int, int>> doc_freqs;
};

/**
 * @brief BM25 index for RAG
 */
class BM25IndexMemory : public Memory<BM25IndexData> {
public:
    BM25IndexMemory() = default;

    BM25IndexMemory(const BM25IndexData& data): num_documents_(data.doc_freqs.size()), num_tokens_(0) {
        this->data = data;
        for(const auto& doc: data.doc_freqs) {
            for(const auto& [token_id, freq]: doc) {
                num_tokens_ += freq;
            }
        }
    }

    size_t get_size() const override {
        return num_tokens_;
    }

    bool is_equal(const Memory<BM25IndexData>& other) override {
        const BM25IndexMemory* other_mem = dynamic_cast<const BM25IndexMemory*>(&other);
        if (!other_mem) {
            return false;
        }
        if (this->num_documents_ != other_mem->num_documents_ ||
            this->num_tokens_ != other_mem->num_tokens_) {
            return false;
        }
        // compare doc_freqs
        if (this->data.doc_freqs.size() != other_mem->data.doc_freqs.size()) {
            return false;
        }
        for (size_t i = 0; i < this->data.doc_freqs.size(); ++i) {
            if (this->data.doc_freqs[i] != other_mem->data.doc_freqs[i]) {
                return false;
            }
        }

        for (const auto& [token_id, df] : this->data.df_map) {
            auto it = other_mem->data.df_map.find(token_id);
            if (it == other_mem->data.df_map.end() || it->second != df) {
                return false;
            }
        }
        return true;
    }

    // we will not delete document for RAG

    void insert_document(const std::unordered_map<int, int>& doc_freq) {
        this->data.doc_freqs.push_back(doc_freq);
        for (const auto& [token_id, freq] : doc_freq) {
            this->data.df_map[token_id] += 1;
            num_tokens_ += freq;
        }
        num_documents_++;
    }
    
    std::string type_name() const override { return "BM25IndexMemory"; }

private:
    size_t num_tokens_;
    size_t num_documents_;
};

/**
 * @brief HMT embedding memory: used in HMT model, 2D array
 */
class HMTEmbeddingMemory : public Memory<std::vector<std::vector<float>>> {
public:
    HMTEmbeddingMemory() = default;

    explicit HMTEmbeddingMemory(
        const std::vector<std::vector<float>>& data,
        size_t segment_length,
        size_t num_entries_per_segment)
        : num_entries_(data.size()),
          memory_size_(data.empty() ? 0 : data[0].size()),
          segment_length_(segment_length),
          num_entries_per_segment_(num_entries_per_segment) {
        this->data = data;
    }
    
    std::string type_name() const override { return "HMTEmbeddingMemory"; }

    bool is_equal(const Memory<std::vector<std::vector<float>>>& other) override {
        const HMTEmbeddingMemory* other_mem = dynamic_cast<const HMTEmbeddingMemory*>(&other);
        if (!other_mem) {
            return false;
        }
        if (this->num_entries_ != other_mem->num_entries_ ||
            this->memory_size_ != other_mem->memory_size_ ||
            this->segment_length_ != other_mem->segment_length_ ||
            this->num_entries_per_segment_ != other_mem->num_entries_per_segment_) {
            return false;
        }
        for (size_t i = 0; i < this->num_entries_; ++i) {
            for (size_t j = 0; j < this->memory_size_; ++j) {
                if (this->data[i][j] != other_mem->data[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

    size_t get_size() const override {
        return num_entries_ * memory_size_;
    }

    size_t get_num_entries() const {
        return num_entries_;
    }

    size_t get_memory_size() const {
        return memory_size_;
    }

    size_t get_segment_length() const {
        return segment_length_;
    }

    size_t get_equiv_context_length() const {
        return num_entries_ * segment_length_ / num_entries_per_segment_;
    }

    void insert_memory(const std::vector<std::vector<float>>& index_vector) {
        if (index_vector.size() != num_entries_per_segment_) {
            throw std::runtime_error("Index vector size does not match entries per segment");
        }
        for (const auto& vec : index_vector) {
            if (vec.size() != memory_size_) {
                throw std::runtime_error("Index vector dimension does not match memory size");
            }
            this->data.push_back(vec);
            num_entries_++;
        }
    }

    void pop_last_memory() {
        if (num_entries_ < num_entries_per_segment_) {
            throw std::runtime_error("Not enough entries to pop a full segment");
        }
        for (size_t i = 0; i < num_entries_per_segment_; i++) {
            this->data.pop_back();
            num_entries_--;
        }
    }

    void delete_memory(size_t idx) {
        if (idx >= num_entries_ / num_entries_per_segment_) {
            throw std::runtime_error("Index out of bounds");
        }
        // delete the entire segment
        size_t start_idx = idx * num_entries_per_segment_;
        this->data.erase(this->data.begin() + start_idx, this->data.begin() + start_idx + num_entries_per_segment_);
        num_entries_ -= num_entries_per_segment_;
    }

    std::vector<float> get_memory(size_t idx) const {
        // get num_entries_per_segment_ vectors concatenated
        if (idx >= num_entries_ / num_entries_per_segment_) {
            throw std::runtime_error("Index out of bounds");
        }
        std::vector<float> result;
        size_t start_idx = idx * num_entries_per_segment_;
        for (size_t i = 0; i < num_entries_per_segment_; i++) {
            const auto& vec = this->data[start_idx + i];
            result.insert(result.end(), vec.begin(), vec.end());
        }
        return result;
    }

private:
    size_t num_entries_;
    size_t memory_size_;
    size_t segment_length_;
    size_t num_entries_per_segment_;
};

struct LaCTMLPData {
    // 3 weights, 1 and 3 has low rank matrices, no bias
    std::vector<std::vector<float>> weight1_u;
    std::vector<std::vector<float>> weight1_v;
    std::vector<std::vector<float>> weight2;
    std::vector<std::vector<float>> weight3_u;
    std::vector<std::vector<float>> weight3_v;
};

/**
 * @brief LaCT MLP memory: parametric memory in LaCT TTT layer, memory is updated by single step BP
 */
class LaCTMLPMemory : public Memory<LaCTMLPData> { 
public:
    LaCTMLPMemory() = default;
    LaCTMLPMemory(const LaCTMLPData& data) {
        this->data = data;
    }

    std::string type_name() const override { return "LaCTMLPMemory"; }

    bool is_equal(const Memory<LaCTMLPData>& other) override {
        const LaCTMLPMemory* other_mem = dynamic_cast<const LaCTMLPMemory*>(&other);
        if (!other_mem) {
            return false;
        }
        // compare all weights
        if (this->data.weight1_u != other_mem->data.weight1_u ||
            this->data.weight1_v != other_mem->data.weight1_v ||
            this->data.weight2 != other_mem->data.weight2 ||
            this->data.weight3_u != other_mem->data.weight3_u ||
            this->data.weight3_v != other_mem->data.weight3_v) {
            return false;
        }
        return true;
    }

    size_t get_size() const override {
        size_t total_size = 0;
        total_size += this->data.weight1_u.size() * this->data.weight1_u[0].size();
        total_size += this->data.weight1_v.size() * this->data.weight1_v[0].size();
        total_size += this->data.weight2.size() * this->data.weight2[0].size();
        total_size += this->data.weight3_u.size() * this->data.weight3_u[0].size();
        total_size += this->data.weight3_v.size() * this->data.weight3_v[0].size();
        return total_size;
    }
};

/**
 * @brief text memory: token ids stored as memory, used in memagent
 */
class TextMemory : public Memory<std::vector<int>> {
public:
    TextMemory() = default;
    TextMemory(const std::vector<int>& token_ids) {
        this->data = token_ids;
    }

    bool is_equal(const Memory<std::vector<int>>& other) override {
        const TextMemory* other_mem = dynamic_cast<const TextMemory*>(&other);
        if (!other_mem) {
            return false;
        }
        if (this->data.size() != other_mem->data.size()) {
            return false;
        }
        for (size_t i = 0; i < this->data.size(); ++i) {
            if (this->data[i] != other_mem->data[i]) {
                return false;
            }
        }
        return true;
    }
    
    std::string type_name() const override { return "TextMemory"; }
    size_t get_size() const override {
        return this->data.size();
    }
};

}  // namespace data_type
}  // namespace heteromm

#endif  // HETEROMM_DEV_TYPES_MEMORY_H_
