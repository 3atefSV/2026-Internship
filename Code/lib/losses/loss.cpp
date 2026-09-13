#include "losses/loss.h"
#include <stdexcept>
#include <string>
#include <string_view>

// ============================================================================
// Base Loss Implementation
// ============================================================================
namespace {

std::string shape_to_string(const Tensor::Shape& shape) {
    std::string text = "(";
    bool first = true;
    for (const auto dim : shape) {
        if (!first) {
            text += ", ";
        }
        text += std::to_string(dim);
        first = false;
    }
    text += ')';
    return text;
}

} // namespace

Tensor Loss::reduce(const Tensor& values) const {
    switch (reduction_) {
    case Reduction::Sum:
        return Tensor({values.sum()}, Tensor::Shape{1});

    case Reduction::None:
        return values;

    case Reduction::Mean:
        return Tensor({values.mean()}, Tensor::Shape{1});

    default:
        throw std::invalid_argument("Unknown reduction mode.");
    }
}

void Loss::check_same_shape(const Tensor& prediction, const Tensor& target, std::string_view name) {
    if (prediction.shape() != target.shape()) {
        throw std::invalid_argument(
            std::string(name) + ": prediction shape " + shape_to_string(prediction.shape()) +
            " does not match target shape " + shape_to_string(target.shape()) + '.');
    }
}

void Loss::check_not_empty(const Tensor& tensor, std::string_view name) {
    if (tensor.empty()) {
        throw std::invalid_argument(std::string(name) + " must not be empty.");
    }
}
