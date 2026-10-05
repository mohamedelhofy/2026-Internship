#include "autograd/loss_function.h"

// ================= MSE Backward =================
MSEBackward::MSEBackward(const Tensor& prediction, const Tensor& target, Reduction reduction)
    : reduction_(reduction) {
    saved_tensors = {prediction, target};
}

std::vector<Tensor> MSEBackward::apply(const Tensor& grad_output) {
    const Tensor& pred = saved_tensors[0];
    const Tensor& target = saved_tensors[1];

    Tensor grad_pred = (pred - target) * 2.0f;

    if (reduction_ == Reduction::Mean) {
        grad_pred = grad_pred / static_cast<Tensor::value_type>(pred.size());
    }

    grad_pred = grad_pred * grad_output;
    Tensor grad_target = grad_pred * -1.0f;

    return {grad_pred, grad_target};
}

// ================= BCE Backward =================
BCEBackward::BCEBackward(const Tensor& probabilities, const Tensor& targets, Reduction reduction,
                         float epsilon)
    : reduction_(reduction), epsilon_(epsilon) {
    saved_tensors = {probabilities, targets};
}

std::vector<Tensor> BCEBackward::apply(const Tensor& grad_output) {
    const Tensor& probs = saved_tensors[0];
    const Tensor& targets = saved_tensors[1];

    Tensor clipped = probs.clamp(epsilon_, 1.0f - epsilon_);

    Tensor numerator = clipped - targets;
    Tensor denominator = clipped * (clipped * -1.0f + 1.0f);
    Tensor grad_pred = numerator / denominator;

    if (reduction_ == Reduction::Mean) {
        grad_pred = grad_pred / static_cast<Tensor::value_type>(probs.size());
    }

    grad_pred = grad_pred * grad_output;
    return {grad_pred, Tensor()};
}

// ================= Cross Entropy Backward =================
CrossEntropyBackward::CrossEntropyBackward(const Tensor& logits,
                                           const std::vector<Tensor::size_type>& targets,
                                           Reduction reduction, bool batched)
    : targets_(targets), reduction_(reduction), batched_(batched) {
    saved_tensors = {logits.softmax(-1)};
}

std::vector<Tensor> CrossEntropyBackward::apply(const Tensor& grad_output) {
    Tensor softmax_probs = saved_tensors[0];
    Tensor::Storage grad_data = softmax_probs.data();

    Tensor::size_type batch_size = batched_ ? softmax_probs.shape()[0] : 1;
    Tensor::size_type num_classes = softmax_probs.shape()[softmax_probs.ndim() - 1];

    for (Tensor::size_type sample = 0; sample < batch_size; ++sample) {
        Tensor::size_type target = targets_[sample];
        Tensor::size_type flat_index = sample * num_classes + target;
        grad_data[flat_index] -= 1.0f;
    }

    Tensor grad_logits(grad_data, softmax_probs.shape());

    if (reduction_ == Reduction::Mean) {
        grad_logits = grad_logits / static_cast<Tensor::value_type>(batch_size);
    }

    grad_logits = grad_logits * grad_output;
    return {grad_logits, Tensor()};
}
