#include "tensor.h"
// ============================= Tensor Arithmetic Operations
// =============================

template <typename BinaryOp>
Tensor Tensor::apply_tensor_operation(const Tensor& other, BinaryOp op,
                                      const std::string& operation_name,
                                      bool check_division) const {
    // add broadcasting support
    Shape result_shape = broadcast_shape(shape_, other.shape_);
    Storage resultData(compute_size(result_shape));

    for (size_type i = 0; i < resultData.size(); ++i) {
        const Shape output_index = unravel_index(i, result_shape);
        const Shape lhs_index = broadcast_index(output_index, shape_);
        const Shape rhs_index = broadcast_index(output_index, other.shape_);
        if (check_division && other.data_[ravel_index(rhs_index, other.shape_)] == 0) {
            throw std::invalid_argument("Division by zero in tensor-tensor operation.");
        }
        resultData[i] = op(data_[ravel_index(lhs_index, shape_)],
                           other.data_[ravel_index(rhs_index, other.shape_)]);
    }
    return Tensor(resultData, result_shape);
}

template <typename BinaryOp>
Tensor Tensor::apply_scalar_operation(value_type scalar, BinaryOp op, bool check_division) const {
    if (check_division && scalar == 0) {
        throw std::invalid_argument("Division by zero in tensor-scalar operation.");
    }
    Storage resultData(data_.size());
    for (size_type i = 0; i < data_.size(); ++i) {
        resultData[i] = op(data_[i], scalar);
    }
    return Tensor(resultData, shape_);
}
// ============================= Addition =============================
Tensor Tensor::operator+(const Tensor& other) const {
    return apply_tensor_operation(other, std::plus<value_type>(), "addition");
}

Tensor Tensor::operator+(const value_type scalar) const {
    return apply_scalar_operation(scalar, std::plus<value_type>(), false);
}

Tensor operator+(const Tensor::value_type scalar, const Tensor& tensor) {
    return tensor + scalar; // reuse the member function
}
Tensor& Tensor::operator+=(const Tensor& other) {
    *this = *this + other; // reuse the operator+
    return *this;
}
Tensor& Tensor::operator+=(const value_type scalar) {
    *this = *this + scalar; // reuse the member function
    return *this;
}

// ============================= Subtraction =============================
Tensor Tensor::operator-(const Tensor& other) const {
    return apply_tensor_operation(other, std::minus<value_type>(), "subtraction");
}

Tensor Tensor::operator-(const value_type scalar) const {
    return apply_scalar_operation(scalar, std::minus<value_type>(), false);
}

Tensor operator-(const Tensor::value_type scalar, const Tensor& tensor) {
    Tensor::Storage resultData(tensor.data().size());
    for (Tensor::size_type i = 0; i < tensor.data().size(); ++i) {
        resultData[i] = scalar - tensor.data()[i];
    }
    return Tensor(resultData, tensor.shape());
}
Tensor& Tensor::operator-=(const Tensor& other) {

    *this = *this - other; // reuse the operator-
    return *this;
}
Tensor& Tensor::operator-=(const value_type scalar) {
    *this = *this - scalar; // reuse the member function
    return *this;
}

// ============================= Multiplication =============================
Tensor Tensor::operator*(const Tensor& other) const {
    return apply_tensor_operation(other, std::multiplies<value_type>(), "multiplication");
}

Tensor Tensor::operator*(const value_type scalar) const {
    return apply_scalar_operation(scalar, std::multiplies<value_type>(), false);
}
Tensor operator*(const Tensor::value_type scalar, const Tensor& tensor) {
    return tensor * scalar; // reuse the member function
}
Tensor& Tensor::operator*=(const Tensor& other) {
    *this = *this * other; // reuse the operator*
    return *this;
}

Tensor& Tensor::operator*=(const value_type scalar) {
    *this = *this * scalar; // reuse the member function
    return *this;
}

// ============================= Division =============================
Tensor Tensor::operator/(const Tensor& other) const {

    return apply_tensor_operation(other, std::divides<value_type>(), "division",
                                  true); // check for division by zero
}

Tensor Tensor::operator/(const value_type scalar) const {
    return apply_scalar_operation(scalar, std::divides<value_type>(),
                                  true); // check for division by zero
}

Tensor operator/(const Tensor::value_type scalar, const Tensor& tensor) {
    Tensor::Storage resultData(tensor.data().size());
    for (Tensor::size_type i = 0; i < tensor.data().size(); ++i) {
        if (tensor.data()[i] == 0) {
            throw std::invalid_argument("Division by zero in scalar-tensor division.");
        }
        resultData[i] = scalar / tensor.data()[i];
    }
    return Tensor(resultData, tensor.shape());
}

Tensor& Tensor::operator/=(const Tensor& other) {
    *this = *this / other; // reuse the operator/
    return *this;
}
Tensor& Tensor::operator/=(const value_type scalar) {
    *this = *this / scalar; // reuse the member function
    return *this;
}