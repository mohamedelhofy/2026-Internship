#pragma once

#include "autograd/function.h"
#include "losses/loss.h"
#include <vector>

// ================= MSE Backward =================
class MSEBackward : public Function {
  public:
    MSEBackward(const Tensor& prediction, const Tensor& target, Reduction reduction);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;

  private:
    Reduction reduction_;
};

// ================= BCE Backward =================
class BCEBackward : public Function {
  public:
    BCEBackward(const Tensor& probabilities, const Tensor& targets, Reduction reduction,
                float epsilon);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;

  private:
    Reduction reduction_;
    float epsilon_;
};

// ================= Cross Entropy Backward =================
class CrossEntropyBackward : public Function {
  public:
    CrossEntropyBackward(const Tensor& logits, const std::vector<Tensor::size_type>& targets,
                         Reduction reduction, bool batched);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;

  private:
    std::vector<Tensor::size_type> targets_;
    Reduction reduction_;
    bool batched_;
};
