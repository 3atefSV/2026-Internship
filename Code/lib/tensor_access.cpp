#include "tensor.h"

Tensor::value_type& Tensor::operator()(const Shape& indices) {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices must match tensor dimensions.");
    }

    for (size_type i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds.");
        }
    }

    return data_[ravel_index(indices, shape_)];
}

const Tensor::value_type& Tensor::operator()(const Shape& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices must match tensor dimensions.");
    }

    for (size_type i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds.");
        }
    }

    return data_[ravel_index(indices, shape_)];
}
