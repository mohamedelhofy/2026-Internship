#pragma once

#include "optimizers/optimizer.h"

#include <memory>
#include <vector>

class Adam : public Optimizer {
  public:
    Adam(const ParamList& params, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
         float eps = 1e-8f, float weight_decay = 0.0f);

    void step() override;

    // =============== Hyper-parameters ===============
    [[nodiscard]] float lr() const noexcept { return lr_; }
    [[nodiscard]] float beta1() const noexcept { return beta1_; }
    [[nodiscard]] float beta2() const noexcept { return beta2_; }
    [[nodiscard]] float eps() const noexcept { return eps_; }
    [[nodiscard]] float weight_decay() const noexcept { return weight_decay_; }

    void set_lr(float lr) noexcept { lr_ = lr; }

  protected:
    // Shared by AdamW which reuses the exact Adam step on pure gradients.
    void ensure_state();

    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;

    std::vector<std::shared_ptr<Tensor>> m_buffers_;
    std::vector<std::shared_ptr<Tensor>> v_buffers_;
};
