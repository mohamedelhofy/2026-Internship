#pragma once

#include "optimizers/optimizer.h"

// SGD Optimizer
class SGD : public Optimizer {
  public:
    SGD(const ParamList& params, float lr = 0.01f, float momentum = 0.0f,
        float weight_decay = 0.0f);

    void step() override;

    [[nodiscard]] float lr() const noexcept { return lr_; }
    [[nodiscard]] float momentum() const noexcept { return momentum_; }
    [[nodiscard]] float weight_decay() const noexcept { return weight_decay_; }

    void set_lr(float lr) noexcept { lr_ = lr; }

  private:
    void ensure_velocities();

    float lr_;
    float momentum_;
    float weight_decay_;

    // One velocity tensor per parameter, created lazily on the first step.
    std::vector<std::shared_ptr<Tensor>> velocities_;
};
