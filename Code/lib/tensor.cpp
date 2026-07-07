#include "tensor.h"
#include <stdexcept>


Tensor::Tensor(const Storage &data, const Shape &shape)
    : data_(data), shape_(shape) {}

const Tensor::Shape &Tensor::getShape() const { return shape_; }

const Tensor::Storage &Tensor::getData() const { return data_; }

Tensor::size_type Tensor::getSize() const { return data_.size(); }

Tensor::size_type Tensor::getNumDimensions() const { return shape_.size(); }

std::string Tensor::dtype() const {
  return "float"; // currently only float is supported
}

bool Tensor::isEmpty() const { return data_.empty(); }

Tensor Tensor::operator+(const Tensor &other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("Shapes of tensors do not match for addition.");
  }

  Tensor::Storage resultData(data_.size());
  for (Tensor::size_type i = 0; i < data_.size(); ++i) {
    resultData[i] = data_[i] + other.data_[i];
  }

  return Tensor(resultData, shape_);
}
Tensor Tensor::operator-(const Tensor &other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument(
        "Shapes of tensors do not match for subtraction.");
  }

  Tensor::Storage resultData(data_.size());
  for (Tensor::size_type i = 0; i < data_.size(); ++i) {
    resultData[i] = data_[i] - other.data_[i];
  }

  return Tensor(resultData, shape_);
}

Tensor Tensor::operator*(const Tensor &other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument(
        "Shapes of tensors do not match for multiplication.");
  }
  Tensor::Storage resultData(data_.size());
  for (Tensor::size_type i = 0; i < data_.size(); ++i) {
    resultData[i] = data_[i] * other.data_[i];
  }
  return Tensor(resultData, shape_);
}

Tensor Tensor::operator/(const Tensor &other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("Shapes of tensors do not match for division.");
  }
  Tensor::Storage resultData(data_.size());
  for (Tensor::size_type i = 0; i < data_.size(); ++i) {
    if (other.data_[i] == 0) {
      throw std::invalid_argument("Division by zero in tensor division.");
    }
    resultData[i] = data_[i] / other.data_[i];
  }
  return Tensor(resultData, shape_);
}

Tensor Tensor::operator+(const value_type scalar) const {
  Tensor::Storage resultData(data_.size());
  for (Tensor::size_type i = 0; i < data_.size(); ++i) {
    resultData[i] = data_[i] + scalar;
  }
  return Tensor(resultData, shape_);
}

Tensor Tensor::operator-(const value_type scalar) const {
  Tensor::Storage resultData(data_.size());
  for (Tensor::size_type i = 0; i < data_.size(); ++i) {
    resultData[i] = data_[i] - scalar;
  }
  return Tensor(resultData, shape_);
}

Tensor Tensor::operator*(const value_type scalar) const {
  Tensor::Storage resultData(data_.size());
  for (Tensor::size_type i = 0; i < data_.size(); ++i) {
    resultData[i] = data_[i] * scalar;
  }
  return Tensor(resultData, shape_);
}

Tensor Tensor::operator/(const value_type scalar) const {
  if (scalar == 0) {
    throw std::invalid_argument("Division by zero in tensor-scalar division.");
  }
  Tensor::Storage resultData(data_.size());
  for (Tensor::size_type i = 0; i < data_.size(); ++i) {
    resultData[i] = data_[i] / scalar;
  }
  return Tensor(resultData, shape_);
}

Tensor::value_type &Tensor::operator()(Tensor::size_type row, Tensor::size_type col) {
  if (shape_.size() != 2) {
    throw std::invalid_argument("Tensor is not 2D.");
  }
  if (row >= shape_[0] || col >= shape_[1]) {
    throw std::out_of_range("Index out of bounds.");
  }
  return data_[row * shape_[1] + col];
}

const Tensor::value_type &Tensor::operator()(Tensor::size_type row, Tensor::size_type col) const {
  if (shape_.size() != 2) {
    throw std::invalid_argument("Tensor is not 2D.");
  }
  if (row >= shape_[0] || col >= shape_[1]) {
    throw std::out_of_range("Index out of bounds.");
  }
  return data_[row * shape_[1] + col];
}

Tensor Tensor::matmul(const Tensor &other) const {
  if (shape_.size() != 2 || other.shape_.size() != 2) {
    throw std::invalid_argument(
        "Matrix multiplication supports only 2D tensors.");
  }

  Tensor::size_type rowsA = shape_[0];
  Tensor::size_type colsA = shape_[1];

  Tensor::size_type rowsB = other.shape_[0];
  Tensor::size_type colsB = other.shape_[1];

  if (colsA != rowsB) {
    throw std::invalid_argument("Matrix dimensions are incompatible.");
  }

  Tensor::Storage result(rowsA * colsB, 0.0f);

  for (Tensor::size_type i = 0; i < rowsA; ++i) {
    for (Tensor::size_type j = 0; j < colsB; ++j) {
      Tensor::value_type sum = 0.0f;

      for (Tensor::size_type k = 0; k < colsA; ++k) {
        sum += (*this)(i, k) * other(k, j);
      }

      result[i * colsB + j] = sum;
    }
  }

  return Tensor(result, {rowsA, colsB});
}