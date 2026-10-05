#pragma once

#include "losses/loss.h"
#include "tensor/tensor.h"

class BinaryCrossEntropyLoss : public Loss {
  public:
    static constexpr Tensor::value_type default_epsilon = 1e-7f;

    explicit BinaryCrossEntropyLoss(Reduction reduction = Reduction::Mean,
                                    Tensor::value_type epsilon = default_epsilon);
    ~BinaryCrossEntropyLoss() override = default;

    [[nodiscard]] Tensor forward(const Tensor& probabilities, const Tensor& targets) const override;

    [[nodiscard]] Tensor::value_type epsilon() const noexcept { return epsilon_; }

  private:
    // Check that all values in the tensor lie in the unit interval [0, 1].
    static void check_in_unit_interval(const Tensor& tensor, const char* name);

    Tensor::value_type epsilon_;
};