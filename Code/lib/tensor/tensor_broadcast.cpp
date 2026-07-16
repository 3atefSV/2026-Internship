#include "tensor/tensor.h"
#include <stdexcept>

Tensor::Shape Tensor::broadcast_shape(const Shape& lhs, const Shape& rhs) {

    const size_type ndim = std::max(lhs.size(), rhs.size());
    Shape result_shape(ndim);
    for (size_type i = 0; i < ndim; ++i) {
        size_type lhs_dim = 1;
        size_type rhs_dim = 1;
        if (i >= ndim - lhs.size()) {
            lhs_dim = lhs[i - (ndim - lhs.size())];
        }
        if (i >= ndim - rhs.size()) {
            rhs_dim = rhs[i - (ndim - rhs.size())];
        }
        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            throw std::invalid_argument("Shapes are not broadcastable.");
        }
        result_shape[i] = std::max(lhs_dim, rhs_dim);
    }
    return result_shape;
}

Tensor::size_type Tensor::compute_size(const Shape& shape) {
    size_type size = 1;
    for (size_type dim : shape) {
        size *= dim;
    }
    return size;
}

Tensor::Shape Tensor::unravel_index(size_type index, const Shape& shape) {
    Shape indices(shape.size());
    for (size_type i = shape.size(); i-- > 0;) {
        indices[i] = index % shape[i];
        index /= shape[i];
    }
    return indices;
}
Tensor::size_type Tensor::ravel_index(const Shape& indices, const Shape& shape) {
    size_type flat = 0;
    size_type stride = 1;
    for (size_type i = shape.size(); i-- > 0;) {
        flat += indices[i] * stride;
        stride *= shape[i];
    }
    return flat;
}

Tensor::Shape Tensor::broadcast_index(const Shape& output_index, const Shape& input_shape) {
    Shape index(input_shape.size());

    const size_type offset = output_index.size() - input_shape.size();

    for (size_type i = 0; i < input_shape.size(); ++i) {
        if (input_shape[i] == 1) {
            index[i] = 0;
        } else {
            index[i] = output_index[i + offset];
        }
    }

    return index;
}
