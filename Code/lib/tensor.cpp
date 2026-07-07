#include "tensor.h"
#include <stdexcept>


Tensor::Tensor(const std::vector<float> &data, const std::vector<size_t> &shape)
    : data(data), shape(shape) {}

const std::vector<size_t> &Tensor::getShape() const { return shape; }

const std::vector<float> &Tensor::getData() const { return data; }

size_t Tensor::getSize() const { return data.size(); }

size_t Tensor::getNumDimensions() const { return shape.size(); }

std::string Tensor::dtype() const {
  return "float"; // currently only float is supported
}

bool Tensor::isEmpty() const { return data.empty(); }

Tensor Tensor::operator+(const Tensor &other) const {
  if (shape != other.shape) {
    throw std::invalid_argument("Shapes of tensors do not match for addition.");
  }

  std ::vector<float> resultData(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    resultData[i] = data[i] + other.data[i];
  }

  return Tensor(resultData, shape);
}
Tensor Tensor::operator-(const Tensor &other) const {
  if (shape != other.shape) {
    throw std::invalid_argument(
        "Shapes of tensors do not match for subtraction.");
  }

  std::vector<float> resultData(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    resultData[i] = data[i] - other.data[i];
  }

  return Tensor(resultData, shape);
}

Tensor Tensor::operator*(const Tensor &other) const {
  if (shape != other.shape) {
    throw std::invalid_argument(
        "Shapes of tensors do not match for multiplication.");
  }
  std::vector<float> resultData(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    resultData[i] = data[i] * other.data[i];
  }
  return Tensor(resultData, shape);
}

Tensor Tensor::operator/(const Tensor &other) const {
  if (shape != other.shape) {
    throw std::invalid_argument("Shapes of tensors do not match for division.");
  }
  std::vector<float> resultData(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    if (other.data[i] == 0) {
      throw std::invalid_argument("Division by zero in tensor division.");
    }
    resultData[i] = data[i] / other.data[i];
  }
  return Tensor(resultData, shape);
}

Tensor Tensor::operator+(const float scalar) const {
  std::vector<float> resultData(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    resultData[i] = data[i] + scalar;
  }
  return Tensor(resultData, shape);
}

Tensor Tensor::operator-(const float scalar) const {
  std::vector<float> resultData(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    resultData[i] = data[i] - scalar;
  }
  return Tensor(resultData, shape);
}

Tensor Tensor::operator*(const float scalar) const {
  std::vector<float> resultData(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    resultData[i] = data[i] * scalar;
  }
  return Tensor(resultData, shape);
}

Tensor Tensor::operator/(const float scalar) const {
  if (scalar == 0) {
    throw std::invalid_argument("Division by zero in tensor-scalar division.");
  }
  std::vector<float> resultData(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    resultData[i] = data[i] / scalar;
  }
  return Tensor(resultData, shape);
}

float &Tensor::operator()(size_t row, size_t col) {
  if (shape.size() != 2) {
    throw std::invalid_argument("Tensor is not 2D.");
  }
  if (row >= shape[0] || col >= shape[1]) {
    throw std::out_of_range("Index out of bounds.");
  }
  return data[row * shape[1] + col];
}

const float &Tensor::operator()(size_t row, size_t col) const {
  if (shape.size() != 2) {
    throw std::invalid_argument("Tensor is not 2D.");
  }
  if (row >= shape[0] || col >= shape[1]) {
    throw std::out_of_range("Index out of bounds.");
  }
  return data[row * shape[1] + col];
}

Tensor Tensor::matmul(const Tensor &other) const {
  if (shape.size() != 2 || other.shape.size() != 2) {
    throw std::invalid_argument(
        "Matrix multiplication supports only 2D tensors.");
  }

  size_t rowsA = shape[0];
  size_t colsA = shape[1];

  size_t rowsB = other.shape[0];
  size_t colsB = other.shape[1];

  if (colsA != rowsB) {
    throw std::invalid_argument("Matrix dimensions are incompatible.");
  }

  std::vector<float> result(rowsA * colsB, 0.0f);

  for (size_t i = 0; i < rowsA; ++i) {
    for (size_t j = 0; j < colsB; ++j) {
      float sum = 0.0f;

      for (size_t k = 0; k < colsA; ++k) {
        sum += (*this)(i, k) * other(k, j);
      }

      result[i * colsB + j] = sum;
    }
  }

  return Tensor(result, {rowsA, colsB});
}