<<<<<<< HEAD:Code/lib/tensor_matrix.cpp
#include "tensor.h"
=======
#include "autograd/function.h"
#include "tensor/tensor.h"
#include <omp.h>
>>>>>>> origin/feature/final-demo:Code/lib/tensor/tensor_matrix.cpp
#include <stdexcept>

Tensor::Shape Tensor::extract_batch_shape(const Shape& shape) {
    if (shape.size() < 2) {
        throw std::invalid_argument(
            "Tensor must have at least 2 dimensions for matrix multiplication.");
    }
    return Shape(shape.begin(), shape.end() - 2);
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (ndim() < 2 || other.ndim() < 2) {
        throw std::invalid_argument(
            "Both tensors must have at least 2 dimensions for matrix multiplication.");
    }

    // Extract batch shapes
    Shape batch_shape_a = extract_batch_shape(shape_);
    Shape batch_shape_b = extract_batch_shape(other.shape_);

    Shape broadcasted_batch_shape = broadcast_shape(batch_shape_a, batch_shape_b);

    if (shape_[shape_.size() - 1] != other.shape_[other.shape_.size() - 2]) {
        throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
    }
    size_type N = shape_[shape_.size() - 2];
    size_type K = shape_[shape_.size() - 1];
    size_type M = other.shape_[other.shape_.size() - 1];

    Shape result_shape = broadcasted_batch_shape;
    result_shape.push_back(N);
    result_shape.push_back(M);

    Storage resultData(compute_size(result_shape));

    for (size_type batch_index = 0; batch_index < compute_size(broadcasted_batch_shape);
         ++batch_index) {
        Shape output_batch_index = unravel_index(batch_index, broadcasted_batch_shape);
        Shape a_batch_index = broadcast_index(output_batch_index, batch_shape_a);
        Shape b_batch_index = broadcast_index(output_batch_index, batch_shape_b);

        // Pre-calculate base offsets for the current batch to avoid calling ravel_index
        // inside the O(N*M*K) loops. We append 0s to get the starting flat index.
        Shape a_start_indices = a_batch_index;
        a_start_indices.push_back(0);
        a_start_indices.push_back(0);

        Shape b_start_indices = b_batch_index;
        b_start_indices.push_back(0);
        b_start_indices.push_back(0);

        Shape result_start_indices = output_batch_index;
        result_start_indices.push_back(0);
        result_start_indices.push_back(0);

        size_type a_base = ravel_index(a_start_indices, shape_);
        size_type b_base = ravel_index(b_start_indices, other.shape_);
        size_type result_base = ravel_index(result_start_indices, result_shape);

        // 1. Cache-Friendly Memory Access: Transpose the B matrix slice for this specific batch
        std::vector<value_type> b_transposed(K * M);
        for (size_type k = 0; k < K; ++k) {
            for (size_type m = 0; m < M; ++m) {
                b_transposed[m * K + k] = other.data_[b_base + k * M + m];
            }
        }

// 2. Parallel Execution & SIMD Vectorization
// Using fast 1D offsets (e.g., n * K + k) instead of ravel_index
#pragma omp parallel for collapse(2) schedule(static)
        for (size_type n = 0; n < N; ++n) {
            for (size_type m = 0; m < M; ++m) {
                value_type sum = 0.0f;

// 3. SIMD Instructions
#pragma omp simd
                for (size_type k = 0; k < K; ++k) {
                    sum += data_[a_base + n * K + k] * b_transposed[m * K + k];
                }

                resultData[result_base + n * M + m] = sum;
            }
        }
    }

    Tensor result(resultData, result_shape);
    if (requires_grad_ || other.requires_grad_) {
        result.set_requires_grad(true);
        result.grad_fn_ = std::make_shared<MatmulBackward>(*this, other);
    }
    return result;
}
