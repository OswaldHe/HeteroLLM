/**
 * @file raw_data.h
 * @brief RawData type definition - input to build_memory step
 * 
 * RawData represents the initial input data (e.g., documents, embeddings)
 * that will be processed to build memory structures.
 */

#ifndef HETEROMM_DEV_TYPES_RAW_DATA_H_
#define HETEROMM_DEV_TYPES_RAW_DATA_H_

#include "base_types.h"
#include "memory.h"
#include <vector>
#include <cstring>

namespace heteromm {
namespace data_type {

/**
 * @brief Base class for raw input data
 * 
 * This represents the initial data that enters the pipeline,
 * such as document embeddings, text tokens, or other structured data.
 * 
 * Usage:
 *   class MyDocumentData : public RawData { ... };
 */
template<typename T>
class RetrievedData : public BaseType<T> {
public:
    RetrievedData() = default;
    virtual ~RetrievedData() = default;
    
    std::string type_name() const override { return "RetrievedData"; }
};

template<typename S>
struct KVCache {
    std::vector<std::vector<S>> keys;
    std::vector<std::vector<S>> values;
};

/**
 * @brief KV cache data, used in any sparse attention
 */
template<typename S>
class KVCacheData : public RetrievedData<KVCache<S>> {
public:
    using content_type = KVCache<S>;
    KVCacheData() = default;
    KVCacheData(const KVCache<S>& data) : context_length_(data.keys.size()),
                                     head_dim_(data.keys.empty() ? 0 : data.keys[0].size()) {
        this->data = data;
    }

    std::string type_name() const override { return "KVCacheData"; }
    size_t get_size() const override {
        return context_length_ * head_dim_ * 2;  // keys + values
    }

    size_t get_context_length() const {
        return context_length_;
    }

    size_t get_head_dim() const {
        return head_dim_;
    }

private:
    size_t context_length_;
    size_t head_dim_;
};

/**
 * @brief text database data: used in RAG and MemAgent
 */
class TextDBData : public RetrievedData<std::vector<std::vector<int>>> {

public:
    using content_type = std::vector<std::vector<int>>;
    TextDBData() = default;
    TextDBData(const std::vector<std::vector<int>>& documents) : num_docs_(documents.size()) {
        this->data = documents;
        total_tokens_ = 0;
        for (const auto& doc : documents) {
            total_tokens_ += doc.size();   
        }
    }

    std::string type_name() const override { return "TextDBData"; }
    size_t get_size() const override {
        return total_tokens_;
    }
    size_t get_num_documents() const {
        return num_docs_;
    }

private:
    size_t num_docs_;
    size_t total_tokens_;
};

/**
 * @brief Parametrized data: used in LaCT TTT, same as memory
 */
class ParametrizedData : public RetrievedData<LaCTMLPData> { 
public:
    using content_type = LaCTMLPData;
    ParametrizedData() = default;
    ParametrizedData(const LaCTMLPData& data) {
        this->data = data;
    }

    std::string type_name() const override { return "ParametrizedData"; }
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



}  // namespace data_type
}  // namespace heteromm

#endif  // HETEROMM_DEV_TYPES_RAW_DATA_H_
