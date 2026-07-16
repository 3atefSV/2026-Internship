#include "activations/activations.h"

Tensor ReLU::forward(const Tensor& x) const { return x.relu(); }

Tensor ReLU::backward(const Tensor& grad) const { return grad; }

Tensor Sigmoid::forward(const Tensor& x) const { return x.sigmoid(); }

Tensor Sigmoid::backward(const Tensor& grad) const { return grad; }

Tensor Tanh::forward(const Tensor& x) const { return x.tanh(); }

Tensor Tanh::backward(const Tensor& grad) const { return grad; }

Tensor GELU::forward(const Tensor& x) const { return x.gelu(); }

Tensor GELU::backward(const Tensor& grad) const { return grad; }

Tensor Softmax::forward(const Tensor& x) const { return x.softmax(dim_); }

Tensor Softmax::backward(const Tensor& grad) const { return grad; }
