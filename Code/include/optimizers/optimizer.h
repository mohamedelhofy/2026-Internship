#pragma once

#include "tensor/tensor.h"

#include <cstddef>
#include <vector>

class Optimizer {
  public:
    using ParamRef = Tensor*;
    using ParamList = std::vector<ParamRef>;

    explicit Optimizer(const ParamList& params);

    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;

    virtual ~Optimizer() = default;

    // Apply a single optimization step using the accumulated gradients.
    virtual void step() = 0;

    void zero_grad();

    [[nodiscard]] std::size_t step_count() const noexcept { return step_count_; }

  protected:
    ParamList params_;
    std::size_t step_count_ = 0;
};
