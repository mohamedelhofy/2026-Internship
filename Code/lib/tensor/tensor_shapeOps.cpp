#include "autograd/function.h"
#include "tensor/tensor.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
Tensor Tensor::reShape(const Shape& new_shape) const {
    // must ensure that the data is the same in the new shape
    size_type newSize{1};
    for (auto& dimen : new_shape) {
        newSize *= dimen;
    }

    if (newSize != this->size()) {
        throw std::invalid_argument("CannotReshape , total number of element must be unchanged");
    }

    Tensor reshapedTensor(this->data_, new_shape);
    if (this->requires_grad_) {
        reshapedTensor.set_requires_grad(true);
        reshapedTensor.grad_fn_ = std::make_shared<ReshapeBackward>(*this);
    }
    return reshapedTensor;
}

Tensor Tensor::Transpose(const Shape& axis) const {
    if (ndim() <= 1) {
        return *this;
    }

    Shape f_axes = axis;
    if (f_axes.empty()) {
        f_axes.resize(ndim());
        std::iota(f_axes.begin(), f_axes.end(), 0);
        std::reverse(f_axes.begin(), f_axes.end());
    }

    if (f_axes.size() != ndim()) {
        throw std::invalid_argument("Axes don't match tensor dimensions");
    }

    Shape used(ndim(), 0);
    for (size_type axis_index : f_axes) {
        if (axis_index >= ndim()) {
            throw std::invalid_argument("Axis index out of range for transpose");
        }
        if (used[axis_index]) {
            throw std::invalid_argument("Duplicate axis in transpose permutation");
        }
        used[axis_index] = 1;
    }

    Shape transposedShape(ndim());
    for (size_type i = 0; i < ndim(); ++i) {
        transposedShape[i] = shape_[f_axes[i]];
    }

    Storage transposedData(size());
    for (size_type index = 0; index < size(); ++index) {
        Shape output_index = unravel_index(index, transposedShape);
        Shape original(ndim());
        for (size_type i = 0; i < ndim(); ++i) {
            original[f_axes[i]] = output_index[i];
        }
        size_type src_index = ravel_index(original, shape_);
        transposedData[index] = data_[src_index];
    }

    // return Tensor(transposedData, transposedShape);
    Tensor transposedTensor(transposedData, transposedShape);
    if (this->requires_grad_) {
        transposedTensor.set_requires_grad(true);
        transposedTensor.grad_fn_ = std::make_shared<TransposeBackward>(*this, f_axes);
    }
    return transposedTensor;
}
