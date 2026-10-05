#include "losses/mse_loss.h"
#include "autograd/loss_function.h"
// ============================================================================
// Mean Squared Error Loss Implementation
// ============================================================================
Tensor MSELoss::forward(const Tensor& prediction, const Tensor& target) const {
    check_not_empty(prediction, "MSELoss prediction");
    check_same_shape(prediction, target, "MSELoss");

    // Element-wise tensor arithmetic; no manual index loops needed. Squaring
    // through multiplication rather than std::pow keeps it to a single pass.
    const Tensor difference = prediction - target;

    // return reduce(difference * difference);
    Tensor result = reduce(difference * difference);

    // ================= Autograd Wiring =================
    if (prediction.requires_grad() || target.requires_grad()) {
        result.set_requires_grad(true);
        result.grad_fn_ = std::make_shared<MSEBackward>(prediction, target, reduction());
    }

    return result;
}
