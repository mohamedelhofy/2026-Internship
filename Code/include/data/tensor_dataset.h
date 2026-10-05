#pragma once

#include "data/dataset.h"
#include <stdexcept>

class TensorDataset : public Dataset {
  public:
    explicit TensorDataset(const std::vector<Tensor>& tensors);
    ~TensorDataset() override = default;

    [[nodiscard]] Tensor::size_type size() const override;
    [[nodiscard]] std::vector<Tensor> get_item(Tensor::size_type idx) const override;

  private:
    std::vector<Tensor> tensors_;
    Tensor::size_type num_samples_;
};
