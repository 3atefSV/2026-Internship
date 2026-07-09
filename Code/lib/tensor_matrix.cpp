#include "tensor.h"

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