#include "optimizers/adamw.h"

#include <cstddef>

AdamW::AdamW(const ParamList& params, float lr, float beta1, float beta2, float eps,
             float weight_decay)
    : Adam(params, lr, beta1, beta2, eps, weight_decay) {}

void AdamW::step() {
    ensure_state();

    // Run the exact Adam update on PURE gradients (no weight decay folded in).
    const float bias_corr1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_ + 1));
    const float bias_corr2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_ + 1));

    for (std::size_t p = 0; p < params_.size(); ++p) {
        Tensor* param = params_[p];
        if (!param->grad_) {
            continue;
        }

        Tensor& grad = *param->grad_;
        Tensor& theta = *param;
        Tensor& m = *m_buffers_[p];
        Tensor& v = *v_buffers_[p];

        const std::size_t n = theta.size();
        auto* grad_ptr = grad.data().data();
        auto* theta_ptr = theta.data().data();
        auto* m_ptr = m.data().data();
        auto* v_ptr = v.data().data();

#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            const float g = grad_ptr[i];

            m_ptr[i] = beta1_ * m_ptr[i] + (1.0f - beta1_) * g;
            v_ptr[i] = beta2_ * v_ptr[i] + (1.0f - beta2_) * g * g;

            const float m_hat = m_ptr[i] / bias_corr1;
            const float v_hat = v_ptr[i] / bias_corr2;

            theta_ptr[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }

    // Decoupled weight decay: theta *= (1 - alpha * lambda).
    const float decay_factor = 1.0f - lr_ * weight_decay_;
    for (Tensor* param : params_) {
        if (!param->grad_) {
            continue;
        }
        Tensor& theta = *param;
        auto* theta_ptr = theta.data().data();
        const std::size_t n = theta.size();
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            theta_ptr[i] *= decay_factor;
        }
    }

    ++step_count_;
}
