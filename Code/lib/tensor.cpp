#include "tensor.h"


Tensor::Tensor(const Storage &data, const Shape &shape)
    : data_(data), shape_(shape) {}

const Tensor::Shape &Tensor::shape() const noexcept { return shape_; }

const Tensor::Storage &Tensor::data() const noexcept { return data_; }

Tensor::size_type Tensor::size() const noexcept { return data_.size(); }

Tensor::size_type Tensor::ndim() const noexcept{ return shape_.size(); }

std::string Tensor::dtype() const noexcept {
  return "float"; // currently only float is supported
}

bool Tensor::empty() const noexcept { return data_.empty(); }

