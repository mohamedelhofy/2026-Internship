#include "losses/loss.h"
#include <stdexcept>
#include <string>

// ============================================================================
// Base Loss Implementation
// ============================================================================
namespace {

std::string shape_to_string(const Tensor::Shape& shape) {
    std::string text = "(";
    for (Tensor::size_type i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            text += ", ";
        }
        text += std::to_string(shape[i]);
    }
    text += ')';
    return text;
}

} // namespace

Tensor Loss::reduce(const Tensor& values) const {
    switch (reduction_) {
    case Reduction::Sum:
        return Tensor({values.sum()}, Tensor::Shape{1});

    case Reduction::None:
        return values;

    case Reduction::Mean:
    default:
        return Tensor({values.mean()}, Tensor::Shape{1});
    }
}

void Loss::check_same_shape(const Tensor& prediction, const Tensor& target, const char* name) {
    if (prediction.shape() != target.shape()) {
        throw std::invalid_argument(
            std::string(name) + ": prediction shape " + shape_to_string(prediction.shape()) +
            " does not match target shape " + shape_to_string(target.shape()) + '.');
    }
}

void Loss::check_not_empty(const Tensor& tensor, const char* name) {
    if (tensor.empty()) {
        throw std::invalid_argument(std::string(name) + " must not be empty.");
    }
}