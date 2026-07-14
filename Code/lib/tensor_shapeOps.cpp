#include "tensor.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

Tensor Tensor::reShape(const Shape& new_shape) const {
    // must ensure that the data is the same in the new shape
    size_type newSize{1};
    for (auto& dimen : new_shape) {
        newSize *= dimen;
    }

    if (newSize != this->size()) {
        throw std::invalid_argument("CannotReshape , total number of element must be unchanged");
    }
    return Tensor(this->data_, new_shape);
}

Tensor Tensor::Transpose(const Shape& axis) const {

    if (this->ndim() <= 1) {
        return *this;
    }
    if (this->ndim() != 2) {
        throw std::invalid_argument("Works on 2D ");
    }
    Shape f_axes = axis;

    if (f_axes.empty()) {
        f_axes.resize(this->ndim());                //{0,0,0}
        std::iota(f_axes.begin(), f_axes.end(), 0); //{2,1,0}
        std::reverse(f_axes.begin(), f_axes.end()); //{0,1,2}

    } else {
        if (f_axes.size() != this->ndim()) {
            throw std::invalid_argument("Axes,Don't Match");
        }
    }

    Shape TransposedShape(this->ndim());
    for (size_t i{0}; i < this->ndim(); ++i) {
        TransposedShape[i] = shape_[f_axes[i]];
    }

    Storage transposedData(this->size());

    for (size_type index{0}; index < this->size(); ++index) {

        Shape transposed_index = unravel_index(index, TransposedShape);
        Shape original(this->ndim());

        for (size_type i = 0; i < this->ndim(); ++i) {
            original[f_axes[i]] = transposed_index[i];
        }

        size_type original_index = ravel_index(this->shape_, original);
        transposedData[index] = data_[original_index];
    }
    return Tensor(transposedData, TransposedShape);
}