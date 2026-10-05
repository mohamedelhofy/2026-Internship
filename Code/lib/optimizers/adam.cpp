#include "optimizers/adam.h"

#include <cmath>
#include <cstddef>

Adam::Adam(const ParamList& params, float lr, float beta1, float beta2, float eps,
           float weight_decay)
    : Optimizer(params), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps),
      weight_decay_(weight_decay) {}

void Adam::ensure_state() {
    if (!m_buffers_.empty()) {
        return;
    }
    m_buffers_.reserve(params_.size());
    v_buffers_.reserve(params_.size());
    for (Tensor* param : params_) {
        auto m = std::make_shared<Tensor>(param->shape());
        m->fill(0.0f);
        m_buffers_.push_back(m);

        auto v = std::make_shared<Tensor>(param->shape());
        v->fill(0.0f);
        v_buffers_.push_back(v);
    }
}

void Adam::step() {
    ensure_state();

    const bool use_decay = weight_decay_ > 0.0f;
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
            float g = grad_ptr[i];
            if (use_decay) {
                g += weight_decay_ * theta_ptr[i];
            }

            m_ptr[i] = beta1_ * m_ptr[i] + (1.0f - beta1_) * g;
            v_ptr[i] = beta2_ * v_ptr[i] + (1.0f - beta2_) * g * g;

            const float m_hat = m_ptr[i] / bias_corr1;
            const float v_hat = v_ptr[i] / bias_corr2;

            theta_ptr[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }

    ++step_count_;
}
