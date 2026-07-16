#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace {

Tensor::value_type sigmoid_stable(Tensor::value_type x) {
    if (x >= 0.0f) {
        const Tensor::value_type exp_neg = std::exp(-x);
        return 1.0f / (1.0f + exp_neg);
    }

    const Tensor::value_type exp_pos = std::exp(x);
    return exp_pos / (1.0f + exp_pos);
}

} // namespace

Tensor Tensor::relu() const {
    Storage result_data(data_.size());

    for (size_type i = 0; i < data_.size(); ++i) {
        result_data[i] = data_[i] > 0.0f ? data_[i] : 0.0f;
    }

    return Tensor(result_data, shape_);
}

Tensor Tensor::sigmoid() const {
    Storage result_data(data_.size());

    for (size_type i = 0; i < data_.size(); ++i) {
        result_data[i] = sigmoid_stable(data_[i]);
    }

    return Tensor(result_data, shape_);
}

Tensor Tensor::tanh() const {
    Storage result_data(data_.size());

    for (size_type i = 0; i < data_.size(); ++i) {
        result_data[i] = std::tanh(data_[i]);
    }

    return Tensor(result_data, shape_);
}

Tensor Tensor::gelu() const {
    Storage result_data(data_.size());

    for (size_type i = 0; i < data_.size(); ++i) {
        const value_type x = data_[i];
        const value_type sig = sigmoid_stable(1.702f * x);
        result_data[i] = x * sig;
    }

    return Tensor(result_data, shape_);
}

Tensor Tensor::softmax(int dim) const {
    if (empty()) {
        return Tensor({}, shape_);
    }

    if (shape_.empty()) {
        return Tensor({1.0f}, {1});
    }

    int normalized_dim = static_cast<int>(dim);
    if (normalized_dim < 0) {
        normalized_dim += static_cast<int>(shape_.size());
    }

    if (normalized_dim < 0 || normalized_dim >= static_cast<int>(shape_.size())) {
        throw std::invalid_argument("Softmax dimension out of range.");
    }

    if (shape_.size() == 1) {
        value_type max_val = data_[0];
        for (size_type i = 1; i < data_.size(); ++i) {
            max_val = std::max(max_val, data_[i]);
        }

        Storage exp_values(data_.size());
        value_type sum_exp = 0.0f;
        for (size_type i = 0; i < data_.size(); ++i) {
            exp_values[i] = std::exp(data_[i] - max_val);
            sum_exp += exp_values[i];
        }

        Storage result_data(data_.size());
        for (size_type i = 0; i < data_.size(); ++i) {
            result_data[i] = exp_values[i] / sum_exp;
        }

        return Tensor(result_data, shape_);
    }

    const size_type axis = static_cast<size_type>(normalized_dim);
    const size_type outer_size = std::accumulate(shape_.begin(), shape_.begin() + axis,
                                                 size_type{1}, std::multiplies<size_type>());
    const size_type inner_size = std::accumulate(shape_.begin() + axis + 1, shape_.end(),
                                                 size_type{1}, std::multiplies<size_type>());
    const size_type dim_size = shape_[axis];

    Storage result_data(data_.size());

    for (size_type outer = 0; outer < outer_size; ++outer) {
        for (size_type inner = 0; inner < inner_size; ++inner) {
            size_type base_offset = outer * (dim_size * inner_size) + inner;
            value_type max_val = data_[base_offset];

            for (size_type d = 1; d < dim_size; ++d) {
                const size_type offset = base_offset + d * inner_size;
                max_val = std::max(max_val, data_[offset]);
            }

            value_type sum_exp = 0.0f;
            for (size_type d = 0; d < dim_size; ++d) {
                const size_type offset = base_offset + d * inner_size;
                const value_type exp_val = std::exp(data_[offset] - max_val);
                result_data[offset] = exp_val;
                sum_exp += exp_val;
            }

            for (size_type d = 0; d < dim_size; ++d) {
                const size_type offset = base_offset + d * inner_size;
                result_data[offset] /= sum_exp;
            }
        }
    }

    return Tensor(result_data, shape_);
}
