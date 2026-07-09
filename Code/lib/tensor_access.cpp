#include "tensor.h"

Tensor::value_type& Tensor::operator()(Tensor::size_type row, Tensor::size_type col) {
    if (shape_.size() != 2) {
        throw std::invalid_argument("Tensor is not 2D.");
    }
    if (row >= shape_[0] || col >= shape_[1]) {
        throw std::out_of_range("Index out of bounds.");
    }
    return data_[row * shape_[1] + col];
}

const Tensor::value_type& Tensor::operator()(Tensor::size_type row, Tensor::size_type col) const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("Tensor is not 2D.");
    }
    if (row >= shape_[0] || col >= shape_[1]) {
        throw std::out_of_range("Index out of bounds.");
    }
    return data_[row * shape_[1] + col];
}
