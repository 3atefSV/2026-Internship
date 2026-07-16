#include "tensor/tensor.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

Tensor::value_type Tensor::sum() const noexcept {
    if (empty())
        return 0.0f;
    return std::accumulate(data_.begin(), data_.end(), 0.0f);
}

Tensor::value_type Tensor::mean() const {
    if (empty()) {
        throw std::runtime_error("Cannot calculate mean of an empty Tensor.");
    }
    return sum() / static_cast<value_type>(size());
}

Tensor::value_type Tensor::max() const {
    if (empty()) {
        throw std::runtime_error("Cannot calculate max of an empty Tensor.");
    }
    return *std::max_element(data_.begin(), data_.end());
}
// Helper function , not in tensor.h

Tensor::Shape Tensor::reduced_shape(const Shape& shape, size_type axis) {
    if (axis >= shape.size()) {
        throw std::out_of_range("Axis out of range for reduction.");
    }
    Shape result_shape(shape);
    result_shape.erase(result_shape.begin() + axis);
    return result_shape;
}

Tensor::Storage Tensor::reduce_axis_data(const Tensor& tensor, size_type axis, bool compute_mean) {
    const Shape result_shape = reduced_shape(tensor.shape(), axis);
    Storage result_data(compute_size(result_shape));

    for (size_type flat_index = 0; flat_index < result_data.size(); ++flat_index) {
        Shape output_index = unravel_index(flat_index, result_shape);
        value_type accumulator = 0.0f;

        for (size_type axis_index = 0; axis_index < tensor.shape()[axis]; ++axis_index) {
            Shape full_index;
            full_index.reserve(tensor.shape().size());
            size_type output_pos = 0;

            for (size_type dim = 0; dim < tensor.shape().size(); ++dim) {
                if (dim == axis) {
                    full_index.push_back(axis_index);
                } else {
                    full_index.push_back(output_index[output_pos++]);
                }
            }

            accumulator += tensor.data()[ravel_index(full_index, tensor.shape())];
        }

        if (compute_mean && tensor.shape()[axis] > 0) {
            accumulator /= static_cast<value_type>(tensor.shape()[axis]);
        }

        result_data[flat_index] = accumulator;
    }

    return result_data;
}

Tensor Tensor::sum(size_type axis) const {
    return Tensor(reduce_axis_data(*this, axis, false), reduced_shape(shape_, axis));
}

Tensor Tensor::mean(size_type axis) const {
    return Tensor(reduce_axis_data(*this, axis, true), reduced_shape(shape_, axis));
}
Tensor::value_type Tensor::min() const {
    if (empty()) {
        throw std::runtime_error("Cannot calculate min of an empty Tensor.");
    }
    return *std::min_element(data_.begin(), data_.end());
}
