#include "tensor.h"

Tensor::Tensor(const Storage &data, const Shape &shape)
    : data_(data), shape_(shape) {}
