#pragma once

#include "tensor/tensor.h"
#include <vector>

class Function {
  public:
    virtual ~Function() = default;

    // Receives the upstream gradient and returns gradients for each input
    [[nodiscard]] virtual std::vector<Tensor> apply(const Tensor& grad_output) = 0;

    // Pin activations in memory for the backward pass
    std::vector<Tensor> saved_tensors;
};

// ================= Arithmetic =================
class AddBackward : public Function {
  public:
    AddBackward(const Tensor& a, const Tensor& b);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;
};

class SubBackward : public Function {
  public:
    SubBackward(const Tensor& a, const Tensor& b);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;
};

class MulBackward : public Function {
  public:
    MulBackward(const Tensor& a, const Tensor& b);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;
};

class DivBackward : public Function {
  public:
    DivBackward(const Tensor& a, const Tensor& b);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;
};

// ================= Matrix Operations =================
class MatmulBackward : public Function {
  public:
    MatmulBackward(const Tensor& a, const Tensor& b);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;
};

class ReshapeBackward : public Function {
  public:
    explicit ReshapeBackward(const Tensor& a);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;
};

class TransposeBackward : public Function {
  public:
    TransposeBackward(const Tensor& a, const Tensor::Shape& axes);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;

  private:
    Tensor::Shape axes_;
};

// ================= Activations =================
class ReLUBackward : public Function {
  public:
    explicit ReLUBackward(const Tensor& input);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;
};

class SigmoidBackward : public Function {
  public:
    SigmoidBackward(const Tensor& input, const Tensor& output);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;

  private:
    Tensor output_;
};

class TanhBackward : public Function {
  public:
    TanhBackward(const Tensor& input, const Tensor& output);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;

  private:
    Tensor output_;
};

class GELUBackward : public Function {
  public:
    explicit GELUBackward(const Tensor& input);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;
};

class SoftmaxBackward : public Function {
  public:
    SoftmaxBackward(const Tensor& input, const Tensor& output, int dim);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;

  private:
    Tensor output_;
    int dim_;
};
