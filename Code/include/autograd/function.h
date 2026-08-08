#pragma once

#include "tensor/tensor.h"
#include <vector>

class Function {
  public:
    virtual ~Function() = default;

    // Receives the upstream gradient and returns gradients for each input
    [[nodiscard]] virtual std::vector<Tensor> apply(const Tensor& grad_output) = 0;

    // Pin activations in memory for the backward pass
    std::vector<Tensor> saved_tensors;
};
