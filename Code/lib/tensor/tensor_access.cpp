#include "tensor/tensor.h"

#include <algorithm>
#include <stdexcept>

Tensor::value_type& Tensor::operator()(const Shape& indices) {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Expected " + std::to_string(shape_.size()) +
                                    " indices, but got " + std::to_string(indices.size()) + '.');
    }

    auto [idx_it, shape_it] =
        std::mismatch(indices.begin(), indices.end(), shape_.begin(),
                      [](size_type index, size_type extent) { return index < extent; });

    if (idx_it != indices.end()) {
        const auto dim = std::distance(indices.begin(), idx_it);

        throw std::out_of_range("Index " + std::to_string(*idx_it) +
                                " is out of bounds for dimension " + std::to_string(dim) +
                                " (size = " + std::to_string(*shape_it) + ").");
    }

    return data_[ravel_index(indices, shape_)];
}

const Tensor::value_type& Tensor::operator()(const Shape& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Expected " + std::to_string(shape_.size()) +
                                    " indices, but got " + std::to_string(indices.size()) + '.');
    }

    auto [idx_it, shape_it] =
        std::mismatch(indices.begin(), indices.end(), shape_.begin(),
                      [](size_type index, size_type extent) { return index < extent; });

    if (idx_it != indices.end()) {
        const auto dim = std::distance(indices.begin(), idx_it);

        throw std::out_of_range("Index " + std::to_string(*idx_it) +
                                " is out of bounds for dimension " + std::to_string(dim) +
                                " (size = " + std::to_string(*shape_it) + ").");
    }

    return data_[ravel_index(indices, shape_)];
}
