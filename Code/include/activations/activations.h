#pragma once

#include "layer/layer.h"
#include "tensor/tensor.h"
class Activation : public Layer {
  public:
    ~Activation() override = default;

    [[nodiscard]] virtual Tensor forward(const Tensor& x, bool training = true) = 0;
    [[nodiscard]] virtual Tensor backward(const Tensor& grad) const = 0;
};

class ReLU : public Activation {
  public:
    [[nodiscard]] Tensor forward(const Tensor& x, bool training = true) override;
    [[nodiscard]] Tensor backward(const Tensor& grad) const override;
};

class Sigmoid : public Activation {
  public:
    [[nodiscard]] Tensor forward(const Tensor& x, bool training = true) override;
    [[nodiscard]] Tensor backward(const Tensor& grad) const override;
};

class Tanh : public Activation {
  public:
    [[nodiscard]] Tensor forward(const Tensor& x, bool training = true) override;
    [[nodiscard]] Tensor backward(const Tensor& grad) const override;
};

class GELU : public Activation {
  public:
    [[nodiscard]] Tensor forward(const Tensor& x, bool training = true) override;
    [[nodiscard]] Tensor backward(const Tensor& grad) const override;
};

class Softmax : public Activation {
  public:
    explicit Softmax(int dim = -1) : dim_(dim) {}

    [[nodiscard]] Tensor forward(const Tensor& x, bool training = true) override;
    [[nodiscard]] Tensor backward(const Tensor& grad) const override;

  private:
    int dim_;
};
