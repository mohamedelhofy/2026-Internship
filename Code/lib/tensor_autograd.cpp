#include "autograd/function.h"
#include "tensor/tensor.h"
#include <stdexcept>

void Tensor::set_requires_grad(bool req) {
    requires_grad_ = req;
    if (requires_grad_ && !grad_) {
        grad_ = std::make_shared<Tensor>(shape_);
        grad_->fill(0.0f);
    }
}

void Tensor::zero_grad() {
    if (grad_) {
        std::fill(grad_->data().begin(), grad_->data().end(), 0.0f);
    }
}
void Tensor::backward(const std::optional<Tensor>& gradient) {
    if (!requires_grad_)
        return;

    Tensor current_grad;
    if (gradient.has_value()) {
        current_grad = gradient.value();
    } else {
        // Default gradient for scalar outputs is 1.0
        if (size() == 1) {
            current_grad = Tensor(shape_);
            current_grad.fill(1.0f);
        } else {
            throw std::invalid_argument("backward() requires gradient for non-scalar tensors");
        }
    }

    // Accumulate gradient
    if (!grad_) {
        grad_ = std::make_shared<Tensor>(shape_);
        grad_->fill(0.0f);
    }
    *grad_ = *grad_ + current_grad;

    // Propagate backwards (Recursive Tree Walk)
    if (grad_fn_) {
        std::vector<Tensor> input_grads = grad_fn_->apply(current_grad);
        for (size_t i = 0; i < grad_fn_->saved_tensors.size(); ++i) {
            if (grad_fn_->saved_tensors[i].requires_grad()) {
                grad_fn_->saved_tensors[i].backward(input_grads[i]);
            }
        }
    }
}
