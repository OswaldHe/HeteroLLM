/**
 * @file score.h
 * @brief Score type definition - output of compute_score, input to memory_retrieval
 * 
 * Score represents similarity scores or relevance scores computed
 * between queries and memory entries.
 */

#ifndef HETEROMM_DEV_TYPES_SCORE_H_
#define HETEROMM_DEV_TYPES_SCORE_H_

#include "base_types.h"
#include <vector>
#include <cstring>
#include <limits>

namespace heteromm {
namespace data_type {

/**
 * @brief Base class for score data
 * 
 * Score represents the similarity or relevance scores between
 * queries and memory entries. These scores are used by the
 * memory_retrieval step to select the most relevant entries.
 * 
 * Usage:
 *   class MyDotProductScore : public Score { ... };
 */
template<typename T>
class Score : public BaseType<T> {
public:
    Score() = default;
    virtual ~Score() = default;
    virtual bool is_equal(const Score<T>& other) = 0;
    
    std::string type_name() const override { return "Score"; }
};

template<typename S>
class VectorScore : public Score<std::vector<S>> {
public:
    using content_type = std::vector<S>;
    VectorScore() = default;
    VectorScore(const std::vector<S>& scores)
        : num_scores_(scores.size()) {
        this->data = scores;
    }

    std::string type_name() const override { return "VectorScore"; }

    size_t get_size() const override {
        return num_scores_;
    }

    bool is_equal(const Score<std::vector<S>>& other) override {
        const VectorScore<S>* other_vec_score = dynamic_cast<const VectorScore<S>*>(&other);
        if (!other_vec_score) {
            return false;
        }
        if (this->data.size() != other_vec_score->data.size()) {
            return false;
        }
        for (size_t i = 0; i < this->data.size(); ++i) {
            if (this->data[i] != other_vec_score->data[i]) {
                return false;
            }
        }
        return true;
    }

private:
    size_t num_scores_;
};


template<typename S>
class MatrixScore : public Score<std::vector<std::vector<S>>> {
public:
    MatrixScore() = default;
    MatrixScore(const std::vector<std::vector<S>>& scores)
        : num_queries_(scores.size()),
          num_scores_(scores.empty() ? 0 : scores[0].size()) {
        this->data = scores;
    }

    std::string type_name() const override { return "MatrixScore"; }

    size_t get_size() const override {
        return num_queries_ * num_scores_;
    }

    size_t get_num_queries() const {
        return num_queries_;
    }

    size_t get_num_scores() const {
        return num_scores_;
    }

    bool is_equal(const Score<std::vector<std::vector<S>>>& other) override {
        const MatrixScore<S>* other_mat_score = dynamic_cast<const MatrixScore<S>*>(&other);
        if (!other_mat_score) {
            return false;
        }
        if (this->data.size() != other_mat_score->data.size()) {
            return false;
        }
        for (size_t i = 0; i < this->data.size(); ++i) {
            if (this->data[i].size() != other_mat_score->data[i].size()) {
                return false;
            }
            for (size_t j = 0; j < this->data[i].size(); ++j) {
                if (this->data[i][j] != other_mat_score->data[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

private:
    size_t num_queries_;
    size_t num_scores_;
};

/**
 * @brief Loss score: the L(*) output for LaCT TTT layer -> -fw(k)^Tv
 */
class LossScore : public Score<std::vector<std::vector<float>>> {
public:
    LossScore() = default;
    LossScore(const std::vector<std::vector<float>>& losses){
        this->data = losses;
    }

    std::string type_name() const override { return "LossScore"; }

    size_t get_size() const override {
        return this->data.size() * (this->data.empty() ? 0 : this->data[0].size());
    }

    bool is_equal(const Score<std::vector<std::vector<float>>>& other) override {
        const LossScore* other_loss_score = dynamic_cast<const LossScore*>(&other);
        if (!other_loss_score) {
            return false;
        }
        if (this->data.size() != other_loss_score->data.size()) {
            return false;
        }
        for (size_t i = 0; i < this->data.size(); ++i) {
            if (this->data[i].size() != other_loss_score->data[i].size()) {
                return false;
            }
            for (size_t j = 0; j < this->data[i].size(); ++j) {
                if (this->data[i][j] != other_loss_score->data[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }
};

}  // namespace data_type
}  // namespace heteromm

#endif  // HETEROMM_DEV_TYPES_SCORE_H_
