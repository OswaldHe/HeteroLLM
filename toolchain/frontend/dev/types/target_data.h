/**
 * @file output_data.h
 * @brief OutputData type definition - final output of apply_memory step
 * 
 * OutputData represents the final result after applying retrieved
 * memory to the input, such as augmented embeddings or generated tokens.
 */

#ifndef HETEROMM_DEV_TYPES_OUTPUT_DATA_H_
#define HETEROMM_DEV_TYPES_OUTPUT_DATA_H_

#include "base_types.h"
#include <vector>
#include <cstring>
#include <any>

namespace heteromm {
namespace data_type {

/**
 * @brief Base class for output data
 * 
 * Include input and output data. Stationary, does not changed by the retrieval process
 * 
 * Usage:
 *   class MyAugmentedOutput : public OutputData { ... };
 */
template<typename T>
class TargetData : public BaseType<T> {
public:
    TargetData() = default;
    virtual ~TargetData() = default;
    virtual bool is_equal(const TargetData<T>& other) = 0;
    
    std::string type_name() const override { return "TargetData"; }
};

template<typename S>
class VectorInputOutputData : public TargetData<std::vector<S>> {
public:
    using content_type = std::vector<S>;
    VectorInputOutputData() = default;
    VectorInputOutputData(const std::vector<S>& vec) : vector_size_(vec.size()) {
        this->data = vec;
    }

    std::string type_name() const override { return "VectorInputOutputData"; }
    bool is_equal(const TargetData<std::vector<S>>& other) override {
        const VectorInputOutputData<S>* other_data = dynamic_cast<const VectorInputOutputData<S>*>(&other);
        if (!other_data) {
            return false;
        }
        if (this->vector_size_ != other_data->vector_size_) {
            return false;
        }
        for (size_t i = 0; i < this->vector_size_; ++i) {
            if (this->data[i] != other_data->data[i]) {
                return false;
            }
        }
        return true;
    }
    size_t get_size() const override {
        return vector_size_;
    }
private:
    size_t vector_size_;
};

template<typename S>
class MatrixInputOutputData : public TargetData<std::vector<std::vector<S>>> {
public:
    using content_type = std::vector<std::vector<S>>;
    MatrixInputOutputData() = default;
    MatrixInputOutputData(const std::vector<std::vector<S>>& mat)
        : num_rows_(mat.size()), num_cols_(mat.empty() ? 0 : mat[0].size()) {
        this->data = mat;
    }

    bool is_equal(const TargetData<std::vector<std::vector<S>>>& other) override {
        const MatrixInputOutputData<S>* other_data = dynamic_cast<const MatrixInputOutputData<S>*>(&other);
        if (!other_data) {
            return false;
        }
        if (this->num_rows_ != other_data->num_rows_ ||
            this->num_cols_ != other_data->num_cols_) {
            return false;
        }
        for (size_t i = 0; i < this->num_rows_; ++i) {
            for (size_t j = 0; j < this->num_cols_; ++j) {
                if (this->data[i][j] != other_data->data[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

    std::string type_name() const override { return "MatrixInputOutputData"; }
    size_t get_size() const override {
        return num_rows_ * num_cols_;
    }
private:
    size_t num_rows_;
    size_t num_cols_;
};

template<typename S>
class TextInputOutputData : public TargetData<std::vector<int>> {
public:
    using content_type = std::vector<int>;
    TextInputOutputData() = default;
    TextInputOutputData(const std::vector<int>& token_ids) {
        this->data = token_ids;
    }

    bool is_equal(const TargetData<std::vector<int>>& other) override {
        const TextInputOutputData<S>* other_data = dynamic_cast<const TextInputOutputData<S>*>(&other);
        if (!other_data) {
            return false;
        }
        if (this->data.size() != other_data->data.size()) {
            return false;
        }
        for (size_t i = 0; i < this->data.size(); ++i) {
            if (this->data[i] != other_data->data[i]) {
                return false;
            }
        }
        return true;
    }

    std::string type_name() const override { return "TextInputOutputData"; }
    size_t get_size() const override {
        return this->data.size();
    }
};

}  // namespace data_type
}  // namespace heteromm

#endif  // HETEROMM_DEV_TYPES_OUTPUT_DATA_H_
