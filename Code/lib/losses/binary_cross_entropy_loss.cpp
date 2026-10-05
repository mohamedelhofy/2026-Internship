#include "losses/binary_cross_entropy_loss.h"
#include "autograd/loss_function.h"
#include <stdexcept>
#include <string>

BinaryCrossEntropyLoss::BinaryCrossEntropyLoss(Reduction reduction, Tensor::value_type epsilon)
    : Loss(reduction), epsilon_(epsilon) {

    // eps >= 0.5 would invert the clipping interval; eps <= 0 would not protect
    // against log(0) at all.
    if (!(epsilon > 0.0f) || !(epsilon < 0.5f)) {
        throw std::invalid_argument(
            "BinaryCrossEntropyLoss epsilon must lie strictly between 0 and 0.5, but got " +
            std::to_string(epsilon) + '.');
    }

    // An epsilon below float32 resolution rounds the upper bound back to 1.0,
    // which would let log(1 - p) reach log(0) despite the clipping.
    if (!(1.0f - epsilon < 1.0f)) {
        throw std::invalid_argument("BinaryCrossEntropyLoss epsilon is too small to be "
                                    "representable in float32: 1 - epsilon rounds back to 1.");
    }
}

Tensor BinaryCrossEntropyLoss::forward(const Tensor& probabilities, const Tensor& targets) const {
    check_not_empty(probabilities, "BinaryCrossEntropyLoss probabilities");
    check_same_shape(probabilities, targets, "BinaryCrossEntropyLoss");
    check_in_unit_interval(probabilities, "BinaryCrossEntropyLoss probabilities");
    check_in_unit_interval(targets, "BinaryCrossEntropyLoss targets");

    // Clip before taking any logarithm: p = 0 or p = 1 is exactly the case that
    // produces log(0) = -inf and poisons the whole reduction.
    const Tensor clipped = probabilities.clamp(epsilon_, 1.0f - epsilon_);

    const Tensor positive_term = targets * clipped.log();
    const Tensor negative_term = (1.0f - targets) * (1.0f - clipped).log();

    // return reduce((positive_term + negative_term) * -1.0f);
    Tensor result = reduce((positive_term + negative_term) * -1.0f);
    if (probabilities.requires_grad() || targets.requires_grad()) {
        result.set_requires_grad(true);
        result.grad_fn_ =
            std::make_shared<BCEBackward>(probabilities, targets, reduction(), epsilon_);
    }
    return result;
}

void BinaryCrossEntropyLoss::check_in_unit_interval(const Tensor& tensor, const char* name) {
    for (Tensor::size_type i = 0; i < tensor.size(); ++i) {
        const Tensor::value_type value = tensor[i];

        if (!(value >= 0.0f) || !(value <= 1.0f)) { // also rejects NaN
            throw std::invalid_argument(std::string(name) +
                                        " must all lie between 0 and 1, but got " +
                                        std::to_string(value) + '.');
        }
    }
}