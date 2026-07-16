#include "tensor.h"
#include <stdexcept>

Tensor::Shape Tensor::extract_batch_shape(const Shape& shape) const {
    if (shape.size() < 2) {
        throw std::invalid_argument(
            "Tensor must have at least 2 dimensions for matrix multiplication.");
    }
    return Shape(shape.begin(), shape.end() - 2);
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (ndim() < 2 || other.ndim() < 2) {
        throw std::invalid_argument(
            "Both tensors must have at least 2 dimensions for matrix multiplication.");
    }

    // Extract batch shapes
    Shape batch_shape_a = extract_batch_shape(shape_);
    Shape batch_shape_b = extract_batch_shape(other.shape_);

    Shape broadcasted_batch_shape = broadcast_shape(batch_shape_a, batch_shape_b);

    if (shape_[shape_.size() - 1] != other.shape_[other.shape_.size() - 2]) {
        throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
    }
    size_type N = shape_[shape_.size() - 2];
    size_type K = shape_[shape_.size() - 1];
    size_type M = other.shape_[other.shape_.size() - 1];

    Shape result_shape = broadcasted_batch_shape;
    result_shape.push_back(N);
    result_shape.push_back(M);

    Storage resultData(compute_size(result_shape));

    for (size_type batch_index = 0; batch_index < compute_size(broadcasted_batch_shape);
         ++batch_index) {
        Shape output_batch_index = unravel_index(batch_index, broadcasted_batch_shape);
        Shape a_batch_index = broadcast_index(output_batch_index, batch_shape_a);
        Shape b_batch_index = broadcast_index(output_batch_index, batch_shape_b);

        for (size_type n = 0; n < N; ++n) {
            for (size_type m = 0; m < M; ++m) {
                value_type sum = 0;
                for (size_type k = 0; k < K; ++k) {
                    Shape a_indices = a_batch_index;
                    a_indices.push_back(n);
                    a_indices.push_back(k);

                    Shape b_indices = b_batch_index;
                    b_indices.push_back(k);
                    b_indices.push_back(m);

                    sum += data_[ravel_index(a_indices, shape_)] *
                           other.data_[ravel_index(b_indices, other.shape_)];
                }
                Shape result_indices = output_batch_index;
                result_indices.push_back(n);
                result_indices.push_back(m);
                resultData[ravel_index(result_indices, result_shape)] = sum;
            }
        }
    }

    return Tensor(resultData, result_shape);
}
