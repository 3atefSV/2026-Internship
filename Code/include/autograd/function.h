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
class AddBackward : public Function {
  public:
    AddBackward(const Tensor& a, const Tensor& b);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;
};

class SubBackward : public Function {
  public:
    SubBackward(const Tensor& a, const Tensor& b);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;
};

class MulBackward : public Function {
  public:
    MulBackward(const Tensor& a, const Tensor& b);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;
};

class DivBackward : public Function {
  public:
    DivBackward(const Tensor& a, const Tensor& b);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;
};

class MatmulBackward : public Function {
  public:
    MatmulBackward(const Tensor& a, const Tensor& b);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;
};

class ReshapeBackward : public Function {
  public:
    explicit ReshapeBackward(const Tensor& a);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;
};

class TransposeBackward : public Function {
  public:
    TransposeBackward(const Tensor& a, const Tensor::Shape& axes);
    [[nodiscard]] std::vector<Tensor> apply(const Tensor& grad_output) override;
  private:
    Tensor::Shape axes_;
};
