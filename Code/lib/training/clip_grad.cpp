#include "training/clip_grad.h"
#include <cmath>

float clip_grad_norm(const std::vector<Tensor*>& parameters, float max_norm) {
    float total_norm = 0.0f;

    // 1. Compute global L2 norm across all gradients
    for (Tensor* param : parameters) {
        if (param && param->grad_) {
            for (float g : param->grad_->data()) {
                total_norm += g * g;
            }
        }
    }
    total_norm = std::sqrt(total_norm);

    // 2. Scale uniformly if it exceeds threshold
    if (total_norm > max_norm) {
        float clip_coef = max_norm / (total_norm + 1e-6f); // 1e-6 to prevent div by zero
        for (Tensor* param : parameters) {
            if (param && param->grad_) {
                for (float& g : param->grad_->data()) {
                    g *= clip_coef;
                }
            }
        }
    }
    return total_norm;
}
