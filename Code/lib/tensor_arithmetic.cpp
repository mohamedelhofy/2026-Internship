<<<<<<< HEAD:Code/lib/tensor_arithmetic.cpp
#include "tensor.h"
=======
#include "autograd/function.h"
#include "tensor/tensor.h"
#include <algorithm>
#include <cmath>
>>>>>>> origin/feature/final-demo:Code/lib/tensor/tensor_arithmetic.cpp
#include <functional>
#include <omp.h>
#include <stdexcept>

// ============================= Autograd Helper =============================
template <typename BackwardOp>
Tensor wire_autograd(const Tensor& lhs, const Tensor& rhs, Tensor result) {
    if (lhs.requires_grad() || rhs.requires_grad()) {
        result.set_requires_grad(true);
        result.grad_fn_ = std::make_shared<BackwardOp>(lhs, rhs);
    }
    return result;
}

// ============================= Tensor Arithmetic Operations =============================
template <typename BinaryOp>
Tensor Tensor::apply_tensor_operation(const Tensor& other, BinaryOp op, bool check_division) const {
    Shape result_shape = broadcast_shape(shape_, other.shape_);
    Storage resultData(compute_size(result_shape));

<<<<<<< HEAD:Code/lib/tensor_arithmetic.cpp
    for (size_type i = 0; i < resultData.size(); ++i) {

        const Shape output_index = unravel_index(i, result_shape);

        const Shape lhs_index = broadcast_index(output_index, shape_);
        const Shape rhs_index = broadcast_index(output_index, other.shape_);

        const size_type lhs_flat = ravel_index(lhs_index, shape_);
        const size_type rhs_flat = ravel_index(rhs_index, other.shape_);

        const value_type lhs = data_[lhs_flat];
        const value_type rhs = other.data_[rhs_flat];

        if (check_division && rhs == 0) {
            throw std::domain_error("Division by zero in tensor-tensor operation.");
=======
    // ========================================================================
    // 1. FAST PATH: No broadcasting needed (Shapes are exactly the same)
    // ========================================================================
    if (shape_ == other.shape_) {
        if (check_division) {
            // Sequential loop for division to safely handle exceptions
            for (size_type i = 0; i < resultData.size(); ++i) {
                if (other.data_[i] == 0) {
                    throw std::invalid_argument("Division by zero in tensor-tensor operation.");
                }
                resultData[i] = op(data_[i], other.data_[i]);
            }
        } else {
// Lightning-fast parallel SIMD execution
#pragma omp parallel for simd schedule(static)
            for (size_type i = 0; i < resultData.size(); ++i) {
                resultData[i] = op(data_[i], other.data_[i]);
            }
>>>>>>> origin/feature/final-demo:Code/lib/tensor/tensor_arithmetic.cpp
        }
        return Tensor(resultData, result_shape);
    }

    // ========================================================================
    // 2. BROADCASTING PATH: Requires index mapping
    // ========================================================================
    if (check_division) {
        // Sequential fallback to safely throw C++ exceptions
        for (size_type i = 0; i < resultData.size(); ++i) {
            const Shape output_index = unravel_index(i, result_shape);
            const Shape rhs_index = broadcast_index(output_index, other.shape_);
            const value_type rhs = other.data_[ravel_index(rhs_index, other.shape_)];

            if (rhs == 0) {
                throw std::invalid_argument("Division by zero in tensor-tensor operation.");
            }

            const Shape lhs_index = broadcast_index(output_index, shape_);
            const value_type lhs = data_[ravel_index(lhs_index, shape_)];
            resultData[i] = op(lhs, rhs);
        }
    } else {
// Parallel execution for broadcasting (Safe since there are no exceptions)
#pragma omp parallel for schedule(static)
        for (size_type i = 0; i < resultData.size(); ++i) {
            const Shape output_index = unravel_index(i, result_shape);

            const Shape lhs_index = broadcast_index(output_index, shape_);
            const value_type lhs = data_[ravel_index(lhs_index, shape_)];

            const Shape rhs_index = broadcast_index(output_index, other.shape_);
            const value_type rhs = other.data_[ravel_index(rhs_index, other.shape_)];

            resultData[i] = op(lhs, rhs);
        }
    }
    return Tensor(resultData, result_shape);
}

template <typename BinaryOp>
Tensor Tensor::apply_scalar_operation(value_type scalar, BinaryOp op, bool check_division) const {
    if (check_division && scalar == 0) {
        throw std::domain_error("Division by zero in tensor-scalar operation.");
    }
    Storage resultData(data_.size());

// Fully safe to parallelize as exceptions are handled beforehand
#pragma omp parallel for simd schedule(static)
    for (size_type i = 0; i < data_.size(); ++i) {
        resultData[i] = op(data_[i], scalar);
    }
    return Tensor(resultData, shape_);
}

// ============================= Addition =============================
Tensor Tensor::operator+(const Tensor& other) const {
    return wire_autograd<AddBackward>(*this, other,
                                      apply_tensor_operation(other, std::plus<value_type>()));
}

Tensor Tensor::operator+(const value_type scalar) const {
    // Utilize the fast scalar operation path, and construct a dummy tensor for the Autograd graph
    return wire_autograd<AddBackward>(
        *this, Tensor({scalar}, {1}),
        apply_scalar_operation(scalar, std::plus<value_type>(), false));
}

Tensor operator+(const Tensor::value_type scalar, const Tensor& tensor) { return tensor + scalar; }

Tensor& Tensor::operator+=(const Tensor& other) {
    *this = *this + other;
    return *this;
}
Tensor& Tensor::operator+=(const value_type scalar) {
    *this = *this + scalar;
    return *this;
}

// ============================= Subtraction =============================
Tensor Tensor::operator-(const Tensor& other) const {
    return wire_autograd<SubBackward>(*this, other,
                                      apply_tensor_operation(other, std::minus<value_type>()));
}

Tensor Tensor::operator-(const value_type scalar) const {
    return wire_autograd<SubBackward>(
        *this, Tensor({scalar}, {1}),
        apply_scalar_operation(scalar, std::minus<value_type>(), false));
}

Tensor operator-(const Tensor::value_type scalar, const Tensor& tensor) {
    // Subtraction is non-commutative, so it requires its own specialized loop
    Tensor::Storage resultData(tensor.data().size());

#pragma omp parallel for simd schedule(static)
    for (Tensor::size_type i = 0; i < tensor.data().size(); ++i) {
        resultData[i] = scalar - tensor.data()[i];
    }

    Tensor result(resultData, tensor.shape());
    return wire_autograd<SubBackward>(Tensor({scalar}, {1}), tensor, result);
}

Tensor& Tensor::operator-=(const Tensor& other) {
    *this = *this - other;
    return *this;
}
Tensor& Tensor::operator-=(const value_type scalar) {
    *this = *this - scalar;
    return *this;
}

// ============================= Multiplication =============================
Tensor Tensor::operator*(const Tensor& other) const {
    return wire_autograd<MulBackward>(*this, other,
                                      apply_tensor_operation(other, std::multiplies<value_type>()));
}

Tensor Tensor::operator*(const value_type scalar) const {
    return wire_autograd<MulBackward>(
        *this, Tensor({scalar}, {1}),
        apply_scalar_operation(scalar, std::multiplies<value_type>(), false));
}

Tensor operator*(const Tensor::value_type scalar, const Tensor& tensor) { return tensor * scalar; }

Tensor& Tensor::operator*=(const Tensor& other) {
    *this = *this * other;
    return *this;
}
Tensor& Tensor::operator*=(const value_type scalar) {
    *this = *this * scalar;
    return *this;
}

// ============================= Division =============================
Tensor Tensor::operator/(const Tensor& other) const {
    return wire_autograd<DivBackward>(
        *this, other, apply_tensor_operation(other, std::divides<value_type>(), true));
}

Tensor Tensor::operator/(const value_type scalar) const {
    return wire_autograd<DivBackward>(
        *this, Tensor({scalar}, {1}),
        apply_scalar_operation(scalar, std::divides<value_type>(), true));
}

Tensor operator/(const Tensor::value_type scalar, const Tensor& tensor) {
    const auto& tensor_data = tensor.data();
    Tensor::Storage resultData(tensor.data().size());

    // Sequential loop due to exception throwing
    for (Tensor::size_type i = 0; i < tensor.data().size(); ++i) {
        if (tensor_data[i] == 0) {
            throw std::domain_error("Division by zero in scalar-tensor division.");
        }
        resultData[i] = scalar / tensor.data()[i];
    }

    Tensor result(resultData, tensor.shape());
    return wire_autograd<DivBackward>(Tensor({scalar}, {1}), tensor, result);
}

Tensor& Tensor::operator/=(const Tensor& other) {
    *this = *this / other;
    return *this;
}
Tensor& Tensor::operator/=(const value_type scalar) {
    *this = *this / scalar;
    return *this;
}

// ============================= Math Operations =============================
Tensor Tensor::log() const {
    Storage resultData(data_.size());

    // Sequential loop due to exception throwing
    for (size_type i = 0; i < data_.size(); ++i) {
        if (data_[i] <= 0.0f) {
            throw std::domain_error("Logarithm is undefined for non-positive tensor values.");
        }
        resultData[i] = std::log(data_[i]);
    }
    return Tensor(resultData, shape_);
}

// Restrict every element to the closed interval [min_value, max_value].
Tensor Tensor::clamp(const value_type min_value, const value_type max_value) const {
    if (min_value > max_value) {
        throw std::invalid_argument("Clamp bounds are inverted: min is greater than max.");
    }

    Storage resultData(data_.size());

// Fully safe for SIMD vectorization
#pragma omp parallel for simd schedule(static)
    for (size_type i = 0; i < data_.size(); ++i) {
        resultData[i] = std::min(std::max(data_[i], min_value), max_value);
    }
    return Tensor(resultData, shape_);
}
