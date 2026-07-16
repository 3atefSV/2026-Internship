#include "tensor/tensor.h"
#include <iostream>

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor(shape=[";
    for (size_t i = 0; i < tensor.shape().size(); ++i) {
        os << tensor.shape()[i];
        if (i < tensor.shape().size() - 1) {
            os << ", ";
        }
    }
    os << "], data=[";
    for (size_t i = 0; i < tensor.data().size(); ++i) {
        os << tensor.data()[i];
        if (i < tensor.data().size() - 1) {
            os << ", ";
        }
    }
    os << "])";
    return os;
}

bool Tensor::operator==(const Tensor& other) const noexcept {
    return shape_ == other.shape_ && data_ == other.data_;
}
