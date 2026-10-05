#pragma once

#include "losses/loss.h"
#include "tensor/tensor.h"

class MSELoss : public Loss {
  public:
    // ================= Constructors =================
    explicit MSELoss(Reduction reduction = Reduction::Mean) : Loss(reduction) {}
    ~MSELoss() override = default;

    // ================= Core Operations =================
    [[nodiscard]] Tensor forward(const Tensor& prediction, const Tensor& target) const override;
};