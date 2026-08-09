# Neural Network Modules

This module builds upon the Tensor core to provide high-level, PyTorch-like **Neural Network components**. It is organized into three main categories:

* **Architectural Layers**
* **Non-linear Activations**
* **Mathematically Stable Loss Criteria**

---

## Base Layer Architecture

All network modules, except loss functions, derive from a unified abstract `Layer` base class defined in `layer.h`.

This design provides a consistent API across the network architecture.

### Core Interface

#### `forward()`

```cpp
forward(const Tensor& x, bool training = true)
```

The core computation step of every layer.

The `training` flag is particularly important for stateful layers such as `Dropout`, where the behavior differs between training and inference.

#### Callable Operator

Layers implement `operator()``, allowing them to be used directly like functions:

```cpp
Tensor output = layer(input);
```

This provides a PyTorch-like interface for composing neural network components.

#### `parameters()`

```cpp
std::vector<Tensor*> parameters();
```

Returns mutable pointers to the layer's trainable tensors.

This allows external **Optimizers** to collect and update weights and biases without requiring each layer to expose its internal implementation.

---

# Core Layers

TinyTorch provides simple and composable neural-network layers.

## 1. Linear Layer

The `Linear` layer implements a fully connected neural network layer:

```text
y = xW + b
```

### Features

* Configurable input features.
* Configurable output features.
* Optional bias.
* Deterministic weight initialization through an optional seed.
* Exposes layer configuration through accessor methods.

### Reproducibility

The layer supports:

```cpp
std::optional<unsigned int> seed
```

which can be used to produce deterministic weight initialization.

### Properties

```cpp
in_features()
out_features()
has_bias()
```

These methods expose the configuration of the layer.

---

## 2. Dropout Layer

`Dropout` is a regularization layer that randomly sets elements to zero during training.

Each element is independently dropped with probability `p`.

### Training Behavior

During training, the remaining elements are scaled by:

```text
1 / (1 - p)
```

This preserves the expected value of the activations.

Conceptually:

```text
output = input * mask / (1 - p)
```

where the mask contains randomly selected zero and one values.

### Inference Behavior

When:

```cpp
training = false
```

Dropout behaves as an identity function:

```text
output = input
```

No elements are randomly removed during inference.

### Random Number Generation

The layer maintains an internal:

```cpp
std::mt19937
```

random number generator across forward passes.

An optional seed can be provided for deterministic and reproducible behavior.

---

## 3. Sequential Container

`Sequential` is a composable container that chains multiple `LayerPtr` objects together.

### Construction

Layers can be provided using an initializer list:

```cpp
Sequential({layer1, layer2, layer3});
```

### Forward Pass

The output of each layer automatically becomes the input of the next:

```text
input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Layer 3
  ↓
output
```

Conceptually:

```cpp
x = layer1(x);
x = layer2(x);
x = layer3(x);
```

### Parameter Collection

The `parameters()` method recursively aggregates the trainable parameters from all child layers.

This allows an optimizer to work with the entire network through a single parameter collection.

---

# Activation Module

TinyTorch provides a dedicated activation module outside the Tensor core.

The activation classes support layer-style usage and Python access through PyBind11.

All activation classes inherit from an abstract `Activation` base class, which itself inherits from `Layer`.

Each activation forwards its computation to the corresponding Tensor operation.

## Available Activations

### ReLU

```cpp
ReLU
```

Applies the Rectified Linear Unit activation.

```text
ReLU(x) = max(0, x)
```

### Sigmoid

```cpp
Sigmoid
```

Maps values into the range `(0, 1)`.

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

### Tanh

```cpp
Tanh
```

Maps values into the range `(-1, 1)`.

### GELU

```cpp
GELU
```

Applies the Gaussian Error Linear Unit activation commonly used in modern neural networks.

### Softmax

```cpp
Softmax
```

Converts values into normalized probabilities along a configurable dimension.

The dimension is controlled through:

```cpp
dim_
```

allowing Softmax to operate on different tensor axes.

---

## Activation & Autograd

Since activations inherit from `Layer`, they expose the layer interface and implement the required `backward(grad)` functionality for standalone gradient tracking.

However, during normal execution, gradient propagation is primarily handled dynamically by the **Autograd engine**.

This keeps the activation implementations lightweight while allowing them to participate naturally in the computation graph.

---

# Loss Module

TinyTorch provides dedicated loss criteria used to train neural networks.

The implemented losses derive from an abstract `Loss` base class.

Although `Loss` follows a design similar to `Layer`, it does **not** inherit from `Layer`.

### Why?

`Layer::forward()` has a single-input interface:

```cpp
forward(const Tensor& x, bool training = true)
```

while a loss function requires two inputs:

```cpp
forward(predictions, targets)
```

Therefore, `Loss` provides its own two-input interface.

---

## Loss Interface

Example usage:

```cpp
MSELoss criterion;

Tensor loss = criterion(predictions, targets);
```

The loss interface provides:

* A pure virtual `forward()` method.
* A callable `operator()`.
* A `const` forward operation.

The `forward()` method is `const` because loss criteria do not maintain state during computation, unlike stateful layers such as `Dropout`.

---

# Reduction Modes

Every loss criterion supports a `Reduction` mode, mirroring the behavior of PyTorch.

| Mode              | Result                                                   |
| ----------------- | -------------------------------------------------------- |
| `Reduction::Mean` | Scalar tensor of shape `{1}` containing the mean loss.   |
| `Reduction::Sum`  | Scalar tensor of shape `{1}` containing the summed loss. |
| `Reduction::None` | Unreduced per-element or per-sample loss tensor.         |

### Example

```cpp
MSELoss loss(Reduction::Mean);
```

---

# Loss Criteria

TinyTorch currently provides three primary loss functions.

| Criterion                    | Formula                              | Input                                    | Complexity |
| ---------------------------- | ------------------------------------ | ---------------------------------------- | ---------- |
| **`MSELoss`**                | `mean((prediction - target)^2)`      | Any matching shapes                      | `O(N)`     |
| **`CrossEntropyLoss`**       | `-mean_i log_softmax(logits_i)[y_i]` | Logits `{C}` or `{N, C}` + class indices | `O(N × C)` |
| **`BinaryCrossEntropyLoss`** | `-mean(y*log(p) + (1-y)*log(1-p))`   | Probabilities and labels in `[0, 1]`     | `O(N)`     |

---

# MSE Loss

`MSELoss` computes the Mean Squared Error between predictions and targets.

```text
MSE = mean((prediction - target)^2)
```

It supports any pair of tensors with matching shapes.

The computational complexity is:

```text
O(N)
```

where `N` is the number of elements.

---

# Cross Entropy Loss

`CrossEntropyLoss` is designed for multi-class classification.

It supports:

* Single-sample logits with shape `{C}`.
* Batch logits with shape `{N, C}`.
* Class-index targets.

The mathematical formulation is:

```text
L = -mean_i log_softmax(logits_i)[y_i]
```

Its computational complexity is:

```text
O(N × C)
```

where:

* `N` = number of samples.
* `C` = number of classes.

---

# Binary Cross Entropy Loss

`BinaryCrossEntropyLoss` is designed for binary classification.

The loss is computed as:

```text
L = -mean(y * log(p) + (1 - y) * log(1 - p))
```

where:

* `p` is the predicted probability.
* `y` is the target label.

Inputs must represent probabilities and labels within:

```text
[0, 1]
```

The computational complexity is:

```text
O(N)
```

---

# Numerical Stability & Safety

TinyTorch's loss implementations prioritize **mathematical stability** rather than relying on naive formulas.

This is particularly important when working with very large logits or probabilities close to `0` and `1`.

---

## Stable Softmax Integration

`CrossEntropyLoss` does **not** calculate:

```text
log(softmax(x))
```

directly.

Instead, it uses:

```cpp
Tensor::log_softmax()
```

which implements the numerically stable **Log-Sum-Exp** formulation:

```text
log_softmax(x_i)
=
(x_i - m)
-
log(sum_j exp(x_j - m))
```

where:

```text
m = max_j(x_j)
```

Subtracting the maximum value provides numerical stability.

Because:

```text
x_j - m <= 0
```

every exponential satisfies:

```text
exp(x_j - m) <= 1
```

Therefore, `exp()` cannot overflow.

Additionally:

```text
sum_j exp(x_j - m) >= 1
```

so the logarithm remains finite.

This allows logits with values around `1e3` to remain numerically stable, whereas naive:

```text
log(softmax(x))
```

can produce `NaN` or `-inf`.

---

## Epsilon Clipping

`BinaryCrossEntropyLoss` clips probabilities before applying logarithms.

The probability is constrained to:

```text
[eps, 1 - eps]
```

For the default:

```text
eps = 1e-7
```

this prevents:

```text
log(0) = -inf
```

from appearing during the calculation.

For example, a confidently incorrect prediction is bounded by approximately:

```text
-log(eps) ≈ 16.1
```

instead of producing negative infinity and potentially poisoning the entire batch mean.

---

# Target Overloads

`CrossEntropyLoss` supports multiple target representations.

### Floating-Point Tensor Targets

Targets can be provided as flat floating-point tensors containing class indices.

### `IndexList` Targets

Typed `IndexList` arrays are also supported.

This provides flexibility when interfacing with different data-loading and Python binding workflows.

---

# Strict Shape Validation

Loss functions intentionally **do not allow broadcasting between predictions and targets**.

This is an important design decision.

For example, consider:

```text
prediction: {2, 1}
target:     {2}
```

Normal Tensor arithmetic with broadcasting could silently transform these shapes into:

```text
{2, 2}
```

This would produce a mathematically valid tensor operation but an incorrect loss calculation.

TinyTorch therefore performs strict validation before computing a loss:

* Prediction and target shapes must match.
* Empty tensors are rejected.
* Invalid target formats are rejected.
* Broadcasting is not permitted for loss computation.

This prevents silent numerical errors and makes incorrect training data easier to detect.

---

# Summary

The Neural Network module provides the high-level building blocks required to construct and train neural networks in TinyTorch.

It includes:

* Unified `Layer` architecture.
* Fully connected `Linear` layers.
* Training-aware `Dropout`.
* Composable `Sequential` containers.
* ReLU, Sigmoid, Tanh, GELU, and Softmax activations.
* MSE, Cross Entropy, and Binary Cross Entropy losses.
* PyTorch-like reduction modes.
* Numerically stable loss implementations.
* Strict input and shape validation.
* Integration with the Tensor and Autograd systems.

Together, these components provide a clean foundation for building complete neural network architectures while keeping the underlying implementation transparent and suitable for learning and embedded-oriented development.
