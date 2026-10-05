#include "autograd/function.h"
#include "tensor/tensor.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace {

Tensor::value_type sigmoid_stable(Tensor::value_type x) {
    if (x >= 0.0f) {
        const Tensor::value_type exp_neg = std::exp(-x);
        return 1.0f / (1.0f + exp_neg);
    }

    const Tensor::value_type exp_pos = std::exp(x);
    return exp_pos / (1.0f + exp_pos);
}

// Resolve a possibly negative (Python-style) dimension into a valid axis index.
Tensor::size_type normalize_dim(int dim, Tensor::size_type ndim) {
    int normalized_dim = dim;
    if (normalized_dim < 0) {
        normalized_dim += static_cast<int>(ndim);
    }

    if (normalized_dim < 0 || normalized_dim >= static_cast<int>(ndim)) {
        throw std::invalid_argument("Log-softmax dimension out of range.");
    }

    return static_cast<Tensor::size_type>(normalized_dim);
}

} // namespace

Tensor Tensor::relu() const {
    Storage result_data(data_.size());

    for (size_type i = 0; i < data_.size(); ++i) {
        result_data[i] = data_[i] > 0.0f ? data_[i] : 0.0f;
    }
    Tensor result(result_data, shape_);
    if (requires_grad_) {
        result.set_requires_grad(true);
        result.grad_fn_ = std::make_shared<ReLUBackward>(*this);
    }
    return result;
}

Tensor Tensor::sigmoid() const {
    Storage result_data(data_.size());

    for (size_type i = 0; i < data_.size(); ++i) {
        result_data[i] = sigmoid_stable(data_[i]);
    }

    Tensor result(result_data, shape_);
    if (requires_grad_) {
        result.set_requires_grad(true);
        result.grad_fn_ = std::make_shared<SigmoidBackward>(*this, result);
    }
    return result;
}

Tensor Tensor::tanh() const {
    Storage result_data(data_.size());

    for (size_type i = 0; i < data_.size(); ++i) {
        result_data[i] = std::tanh(data_[i]);
    }

    Tensor result(result_data, shape_);
    if (requires_grad_) {
        result.set_requires_grad(true);
        result.grad_fn_ = std::make_shared<TanhBackward>(*this, result);
    }

    return result;
}

Tensor Tensor::gelu() const {
    Storage result_data(data_.size());

    for (size_type i = 0; i < data_.size(); ++i) {
        const value_type x = data_[i];
        const value_type sig = sigmoid_stable(1.702f * x);
        result_data[i] = x * sig;
    }

    Tensor result(result_data, shape_);
    if (requires_grad_) {
        result.set_requires_grad(true);
        result.grad_fn_ = std::make_shared<GELUBackward>(*this);
    }
    return result;
}

Tensor Tensor::softmax(int dim) const {
    if (empty()) {
        return Tensor({}, shape_);
    }

    if (shape_.empty()) {
        return Tensor({1.0f}, {1});
    }

    int normalized_dim = static_cast<int>(dim);
    if (normalized_dim < 0) {
        normalized_dim += static_cast<int>(shape_.size());
    }

    if (normalized_dim < 0 || normalized_dim >= static_cast<int>(shape_.size())) {
        throw std::invalid_argument("Softmax dimension out of range.");
    }

    if (shape_.size() == 1) {
        value_type max_val = data_[0];
        for (size_type i = 1; i < data_.size(); ++i) {
            max_val = std::max(max_val, data_[i]);
        }

        Storage exp_values(data_.size());
        value_type sum_exp = 0.0f;
        for (size_type i = 0; i < data_.size(); ++i) {
            exp_values[i] = std::exp(data_[i] - max_val);
            sum_exp += exp_values[i];
        }

        Storage result_data(data_.size());
        for (size_type i = 0; i < data_.size(); ++i) {
            result_data[i] = exp_values[i] / sum_exp;
        }

        return Tensor(result_data, shape_);
    }

    const size_type axis = static_cast<size_type>(normalized_dim);
    const size_type outer_size = std::accumulate(shape_.begin(), shape_.begin() + axis,
                                                 size_type{1}, std::multiplies<size_type>());
    const size_type inner_size = std::accumulate(shape_.begin() + axis + 1, shape_.end(),
                                                 size_type{1}, std::multiplies<size_type>());
    const size_type dim_size = shape_[axis];

    Storage result_data(data_.size());

    for (size_type outer = 0; outer < outer_size; ++outer) {
        for (size_type inner = 0; inner < inner_size; ++inner) {
            size_type base_offset = outer * (dim_size * inner_size) + inner;
            value_type max_val = data_[base_offset];

            for (size_type d = 1; d < dim_size; ++d) {
                const size_type offset = base_offset + d * inner_size;
                max_val = std::max(max_val, data_[offset]);
            }

            value_type sum_exp = 0.0f;
            for (size_type d = 0; d < dim_size; ++d) {
                const size_type offset = base_offset + d * inner_size;
                const value_type exp_val = std::exp(data_[offset] - max_val);
                result_data[offset] = exp_val;
                sum_exp += exp_val;
            }

            for (size_type d = 0; d < dim_size; ++d) {
                const size_type offset = base_offset + d * inner_size;
                result_data[offset] /= sum_exp;
            }
        }
    }

    Tensor result(result_data, shape_);
    if (requires_grad_) {
        result.set_requires_grad(true);
        result.grad_fn_ = std::make_shared<SoftmaxBackward>(*this, result, dim);
    }
    return result;
}

// Numerically stable log(softmax(x)) along a single dimension.
//
// Naive form:   log_softmax(x_i) = log(exp(x_i) / sum_j exp(x_j))
// exp(x_i) overflows to +inf for large logits and underflows to 0 for very negative ones,
// so log(softmax(x)) is never computed as the composition of two operations.
//
// Log-Sum-Exp form actually used, with m = max_j x_j:
//     log_softmax(x_i) = (x_i - m) - log( sum_j exp(x_j - m) )
// Every exponent is <= 0, so exp() can never overflow, and the largest term is exactly
// exp(0) = 1, so the sum is always >= 1 and its logarithm is always finite.
//
// Complexity: O(n) time (three passes over each lane), O(n) space.
Tensor Tensor::log_softmax(int dim) const {
    if (empty()) {
        return Tensor({}, shape_);
    }

    if (shape_.empty()) {
        return Tensor({0.0f}, {1}); // a single class always has probability 1 -> log(1) = 0
    }

    const size_type axis = normalize_dim(dim, shape_.size());

    const size_type outer_size = std::accumulate(shape_.begin(), shape_.begin() + axis,
                                                 size_type{1}, std::multiplies<size_type>());
    const size_type inner_size = std::accumulate(shape_.begin() + axis + 1, shape_.end(),
                                                 size_type{1}, std::multiplies<size_type>());
    const size_type dim_size = shape_[axis];

    Storage result_data(data_.size());

    for (size_type outer = 0; outer < outer_size; ++outer) {
        for (size_type inner = 0; inner < inner_size; ++inner) {
            const size_type base_offset = outer * (dim_size * inner_size) + inner;

            value_type max_val = data_[base_offset];
            for (size_type d = 1; d < dim_size; ++d) {
                max_val = std::max(max_val, data_[base_offset + d * inner_size]);
            }

            value_type sum_exp = 0.0f;
            for (size_type d = 0; d < dim_size; ++d) {
                sum_exp += std::exp(data_[base_offset + d * inner_size] - max_val);
            }

            // Shifting by max_val is undone here, so the result is exact, not an approximation.
            const value_type log_sum_exp = max_val + std::log(sum_exp);

            for (size_type d = 0; d < dim_size; ++d) {
                const size_type offset = base_offset + d * inner_size;
                result_data[offset] = data_[offset] - log_sum_exp;
            }
        }
    }

    return Tensor(result_data, shape_);
}
