#include "optimizers/sgd.h"

#include <cstddef>

SGD::SGD(const ParamList& params, float lr, float momentum, float weight_decay)
    : Optimizer(params), lr_(lr), momentum_(momentum), weight_decay_(weight_decay) {}

void SGD::ensure_velocities() {
    if (!velocities_.empty()) {
        return;
    }
    velocities_.reserve(params_.size());
    for (Tensor* param : params_) {
        auto velocity = std::make_shared<Tensor>(param->shape());
        velocity->fill(0.0f);
        velocities_.push_back(velocity);
    }
}

void SGD::step() {
    ensure_velocities();

    const bool use_momentum = momentum_ > 0.0f;
    const bool use_decay = weight_decay_ > 0.0f;

    for (std::size_t p = 0; p < params_.size(); ++p) {
        Tensor* param = params_[p];
        if (!param->grad_) {
            continue;
        }

        Tensor& grad = *param->grad_;
        Tensor& theta = *param;
        Tensor& v = *velocities_[p];

        const std::size_t n = theta.size();
        auto* grad_ptr = grad.data().data();
        auto* theta_ptr = theta.data().data();
        auto* vel_ptr = v.data().data();

        if (use_momentum) {
#pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < n; ++i) {
                float g = grad_ptr[i];
                if (use_decay) {
                    g += weight_decay_ * theta_ptr[i];
                }
                vel_ptr[i] = momentum_ * vel_ptr[i] + g;
                theta_ptr[i] -= lr_ * vel_ptr[i];
            }
        } else {
#pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < n; ++i) {
                float g = grad_ptr[i];
                if (use_decay) {
                    g += weight_decay_ * theta_ptr[i];
                }
                theta_ptr[i] -= lr_ * g;
            }
        }
    }

    ++step_count_;
}
