/**
 * @file query.h
 * @brief Query type definition - input to compute_score step
 * 
 * Query represents the search query or prompt that will be used
 * to retrieve relevant information from memory.
 */

#ifndef HETEROMM_DEV_TYPES_QUERY_H_
#define HETEROMM_DEV_TYPES_QUERY_H_

#include "base_types.h"
#include <vector>
#include <cstring>
#include <unordered_map>
#include <stdexcept>

namespace heteromm {
namespace data_type {

/**
 * @brief Base class for query data
 * 
 * Query represents the input for similarity search or retrieval.
 * This could be query embeddings, token sequences, or other
 * representations depending on the retrieval method.
 * 
 * Usage:
 *   class MyQueryEmbedding : public Query { ... };
 */

using BM25QueryType = std::unordered_map<int, int>;  // token_id -> frequency

template<typename T>
class Query : public BaseType<T> {
public:
    Query() = default;
    virtual ~Query() = default;
    
    std::string type_name() const override { return "Query"; }
};

/**
 * @brief vector query: used in vector similarity search, 1D array
 */
template<typename S>
class VectorQuery : public Query<std::vector<S>> {

public:
    using content_type = std::vector<S>;
    VectorQuery() = default;
    VectorQuery(const std::vector<S>& vec) : vector_size_(vec.size()) {
        this->data = vec;
    }
    
    std::string type_name() const override { return "VectorQuery"; }

    size_t get_size() const override {
        return vector_size_;
    }

private:
    size_t vector_size_;

};

/**
 * @brief multi-head query: used in multi-head attention in DSA
 */
template<typename S>
class MultiHeadQuery : public Query<std::vector<std::vector<S>>> {
public:
    MultiHeadQuery() = default;
    MultiHeadQuery(const std::vector<std::vector<S>>& query, const std::vector<float>& weights)
        : num_heads_(query.size()), head_dim_(query.empty() ? 0 : query[0].size()), weights_(weights) {
        this->data = query;
        if (query.size() != weights.size()) {
            // to add additional data format checking, add checker in constructors.
            throw std::invalid_argument("Number of heads and weights size must match.");
        }
    }

    std::string type_name() const override { return "MultiHeadQuery"; }
    size_t get_size() const override {
        return num_heads_ * head_dim_;
    }

    std::vector<float> get_weights() const {
        return weights_;
    }

private:
    size_t num_heads_;
    size_t head_dim_;
    std::vector<float> weights_;
};

/**
 * @brief BM25 query: used in BM25 in RAG
 */
class BM25Query : public Query<BM25QueryType> {
public:
    BM25Query() = default;
    BM25Query(const BM25QueryType& token_freq_map) {
        this->data = token_freq_map;
        num_tokens_ = 0;
        for (const auto& pair : token_freq_map) {
            num_tokens_ += pair.second;
        }
    }
    
    std::string type_name() const override { return "BM25Query"; }

    size_t get_size() const override {
        return num_tokens_;
    }
private:
    size_t num_tokens_;
};

}  // namespace data_type
}  // namespace heteromm

#endif  // HETEROMM_DEV_TYPES_QUERY_H_
