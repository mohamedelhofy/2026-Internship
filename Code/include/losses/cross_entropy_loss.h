#pragma once

#include "losses/loss.h"
#include "tensor/tensor.h"
#include <vector>

class CrossEntropyLoss : public Loss {
  public:
    using IndexList = std::vector<Tensor::size_type>;

    explicit CrossEntropyLoss(Reduction reduction = Reduction::Mean) : Loss(reduction) {}
    ~CrossEntropyLoss() override = default;

    [[nodiscard]] Tensor forward(const Tensor& logits, const Tensor& targets) const override;

    [[nodiscard]] Tensor forward(const Tensor& logits, const IndexList& targets) const;

    // Keep the base (Tensor, Tensor) call operator visible next to the overload below.
    using Loss::operator();

    [[nodiscard]] Tensor operator()(const Tensor& logits, const IndexList& targets) const {
        return forward(logits, targets);
    }

  private:
    // Validate the rank of the logits and report how many classes they hold.
    static Tensor::size_type check_logits(const Tensor& logits);

    // Validate and convert float-encoded class indices into real indices.
    static IndexList to_index_list(const Tensor& targets, Tensor::size_type num_classes);
};