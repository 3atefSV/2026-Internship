#include "losses/cross_entropy_loss.h"
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
// Helper function to format Tensor shape as a readable string like [3, 2, 4]
std::string format_shape(const Tensor::Shape& shape) {
    std::string s = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        s += std::to_string(shape[i]);
        if (i + 1 < shape.size()) {
            s += ", ";
        }
    }
    s += "]";
    return s;
}
}

// ============================================================================
// Cross Entropy Loss Implementation
// ============================================================================
Tensor::size_type CrossEntropyLoss::check_logits(const Tensor& logits) {
    check_not_empty(logits, "CrossEntropyLoss logits");

    if (logits.ndim() != 1 && logits.ndim() != 2) {
        throw std::invalid_argument("CrossEntropyLoss expects logits of shape {num_classes} or "
                                    "{batch_size, num_classes}, but got a " +
                                    std::to_string(logits.ndim()) +
                                    "-dimensional tensor of shape " +format_shape(logits.shape())
                                    + '.');
    }

    return logits.shape()[logits.ndim() - 1];
}

Tensor CrossEntropyLoss::forward(const Tensor& logits, const Tensor& targets) const {
    const Tensor::size_type num_classes = check_logits(logits);

    if (targets.ndim() > 1) {
        throw std::invalid_argument("CrossEntropyLoss expects a flat tensor of class indices, but "
                                    "got a " +
                                    std::to_string(targets.ndim()) + "-dimensional tensor.");
    }

    const bool batched = logits.ndim() == 2;
    const Tensor::size_type batch_size = batched ? logits.shape()[0] : 1;

    if (targets.size() != batch_size) {
        throw std::invalid_argument("CrossEntropyLoss expects one target per sample: got " +
                                    std::to_string(targets.size()) + " targets for " +
                                    std::to_string(batch_size) + " samples.");
    }

    const Tensor log_probabilities = logits.log_softmax(-1);

    Tensor::Storage per_sample(batch_size);

    for (Tensor::size_type sample = 0; sample < batch_size; ++sample) {
        const Tensor::value_type value = targets[sample];

        if (value < 0.0f || value != std::floor(value)) {
            throw std::invalid_argument(
                "CrossEntropyLoss targets must be non-negative whole class indices, but got " +
                std::to_string(value) + '.');
        }

        const auto target = static_cast<Tensor::size_type>(value);

        if (target >= num_classes) {
            throw std::out_of_range("CrossEntropyLoss target " + std::to_string(target) +
                                    " is out of range for " + std::to_string(num_classes) +
                                    " classes.");
        }

        // Direct flat-index access avoids std::vector allocation on every iteration
        const Tensor::size_type flat_index = sample * num_classes + target;
        per_sample[sample] = -log_probabilities[flat_index];
    }

    return reduce(Tensor(per_sample, Tensor::Shape{batch_size}));
}

Tensor CrossEntropyLoss::forward(const Tensor& logits, const IndexList& targets) const {
    const Tensor::size_type num_classes = check_logits(logits);

    const bool batched = logits.ndim() == 2;
    const Tensor::size_type batch_size = batched ? logits.shape()[0] : 1;

    if (targets.size() != batch_size) {
        throw std::invalid_argument("CrossEntropyLoss expects one target per sample: got " +
                                    std::to_string(targets.size()) + " targets for " +
                                    std::to_string(batch_size) + " samples.");
    }

    const Tensor log_probabilities = logits.log_softmax(-1);

    Tensor::Storage per_sample(batch_size);

    for (Tensor::size_type sample = 0; sample < batch_size; ++sample) {
        const Tensor::size_type target = targets[sample];

        if (target >= num_classes) {
            throw std::out_of_range("CrossEntropyLoss target " + std::to_string(target) +
                                    " is out of range for " + std::to_string(num_classes) +
                                    " classes.");
        }

        // Direct flat-index access avoids std::vector allocation on every iteration
        const Tensor::size_type flat_index = sample * num_classes + target;
        per_sample[sample] = -log_probabilities[flat_index];
    }

    return reduce(Tensor(per_sample, Tensor::Shape{batch_size}));
}
