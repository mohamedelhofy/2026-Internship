#pragma once

#include "tensor/tensor.h"
#include <vector>

class Dataset {
  public:
    virtual ~Dataset() = default;

    // Returns the total number of samples
    [[nodiscard]] virtual Tensor::size_type size() const = 0;

    // Returns a single sample at the given index.
    [[nodiscard]] virtual std::vector<Tensor> get_item(Tensor::size_type idx) const = 0;
};
