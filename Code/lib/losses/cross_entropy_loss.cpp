#include "losses/cross_entropy_loss.h"
#include "autograd/loss_function.h"
#include <cmath>
#include <stdexcept>
#include <string>

// ============================================================================
// Cross Entropy Loss Implementation
// ============================================================================
Tensor::size_type CrossEntropyLoss::check_logits(const Tensor& logits) {
    check_not_empty(logits, "CrossEntropyLoss logits");

    if (logits.ndim() != 1 && logits.ndim() != 2) {
        throw std::invalid_argument("CrossEntropyLoss expects logits of shape {num_classes} or "
                                    "{batch_size, num_classes}, but got a " +
                                    std::to_string(logits.ndim()) + "-dimensional tensor.");
    }

    return logits.shape()[logits.ndim() - 1];
}

Tensor CrossEntropyLoss::forward(const Tensor& logits, const IndexList& targets) const {
    const Tensor::size_type num_classes = check_logits(logits);

    // A 1-D input is treated as a batch of one, which keeps the rest of the
    // function free of special cases.
    const bool batched = logits.ndim() == 2;
    const Tensor::size_type batch_size = batched ? logits.shape()[0] : 1;

    if (targets.size() != batch_size) {
        throw std::invalid_argument("CrossEntropyLoss expects one target per sample: got " +
                                    std::to_string(targets.size()) + " targets for " +
                                    std::to_string(batch_size) + " samples.");
    }

    // Softmax is applied here, never by the caller. Normalising the logits
    // beforehand would apply it twice and quietly flatten the distribution.
    const Tensor log_probabilities = logits.log_softmax(-1);

    Tensor::Storage per_sample(batch_size);

    for (Tensor::size_type sample = 0; sample < batch_size; ++sample) {
        const Tensor::size_type target = targets[sample];

        if (target >= num_classes) {
            throw std::out_of_range("CrossEntropyLoss target " + std::to_string(target) +
                                    " is out of range for " + std::to_string(num_classes) +
                                    " classes.");
        }

        // Pick out the log-probability the model assigned to the true class.
        const Tensor::Shape index = batched ? Tensor::Shape{sample, target} : Tensor::Shape{target};

        per_sample[sample] = -log_probabilities(index);
    }

    // return reduce(Tensor(per_sample, Tensor::Shape{batch_size}));
    Tensor result = reduce(Tensor(per_sample, Tensor::Shape{batch_size}));
    if (logits.requires_grad() || !targets.empty()) {
        result.set_requires_grad(true);
        result.grad_fn_ =
            std::make_shared<CrossEntropyBackward>(logits, targets, reduction(), batched);
    }
    return result;
}

Tensor CrossEntropyLoss::forward(const Tensor& logits, const Tensor& targets) const {
    const Tensor::size_type num_classes = check_logits(logits);

    if (targets.ndim() > 1) {
        throw std::invalid_argument("CrossEntropyLoss expects a flat tensor of class indices, but "
                                    "got a " +
                                    std::to_string(targets.ndim()) + "-dimensional tensor.");
    }

    return forward(logits, to_index_list(targets, num_classes));
}

CrossEntropyLoss::IndexList CrossEntropyLoss::to_index_list(const Tensor& targets,
                                                            Tensor::size_type num_classes) {
    IndexList indices;
    indices.reserve(targets.size());

    for (Tensor::size_type i = 0; i < targets.size(); ++i) {
        const Tensor::value_type value = targets[i];

        // Targets are class labels, so anything fractional or negative is a
        // caller error rather than something to round away.
        if (value < 0.0f || value != std::floor(value)) {
            throw std::invalid_argument(
                "CrossEntropyLoss targets must be non-negative whole class indices, but got " +
                std::to_string(value) + '.');
        }

        const auto index = static_cast<Tensor::size_type>(value);

        if (index >= num_classes) {
            throw std::out_of_range("CrossEntropyLoss target " + std::to_string(index) +
                                    " is out of range for " + std::to_string(num_classes) +
                                    " classes.");
        }

        indices.push_back(index);
    }

    return indices;
}