#pragma once

#include "tensor/tensor.h"

class Activation {
  public:
    virtual ~Activation() = default;

    virtual Tensor forward(const Tensor& x) const = 0;
    virtual Tensor backward(const Tensor& grad) const = 0;

    Tensor operator()(const Tensor& x) const { return forward(x); }
};

class ReLU : public Activation {
  public:
    Tensor forward(const Tensor& x) const override;
    Tensor backward(const Tensor& grad) const override;
};

class Sigmoid : public Activation {
  public:
    Tensor forward(const Tensor& x) const override;
    Tensor backward(const Tensor& grad) const override;
};

class Tanh : public Activation {
  public:
    Tensor forward(const Tensor& x) const override;
    Tensor backward(const Tensor& grad) const override;
};

class GELU : public Activation {
  public:
    Tensor forward(const Tensor& x) const override;
    Tensor backward(const Tensor& grad) const override;
};

class Softmax : public Activation {
  public:
    explicit Softmax(int dim = -1) : dim_(dim) {}

    Tensor forward(const Tensor& x) const override;
    Tensor backward(const Tensor& grad) const override;

  private:
    int dim_;
};
