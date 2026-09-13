# TinyTorch Tensor Library

A lightweight Tensor library implemented in modern C++17 as part of the 2026 Embedded Software Internship.

## Features

### Tensor Properties
- Dynamic N-dimensional tensor representation.
- Shape and storage management.
- Tensor metadata:
  - `shape()`
  - `size()`
  - `ndim()`
  - `empty()`
  - `dtype()`

---

### Arithmetic Operations

Element-wise operations between tensors:

- Tensor + Tensor
- Tensor - Tensor
- Tensor * Tensor
- Tensor / Tensor

Scalar operations:

- Tensor + scalar
- Tensor - scalar
- Tensor * scalar
- Tensor / scalar

Reverse scalar operations:

- scalar + Tensor
- scalar - Tensor
- scalar * Tensor
- scalar / Tensor

Compound assignment operators:

- `+=`
- `-=`
- `*=`
- `/=`

---

### Broadcasting

Implemented NumPy-style broadcasting for element-wise operations.

Supported features:

- Automatic broadcast shape calculation.
- Broadcasting tensors with different dimensions.
- Broadcasting singleton dimensions.
- Shape validation.

Helper functions:

- `broadcast_shape()`
- `broadcast_index()`
- `compute_size()`
- `unravel_index()`
- `ravel_index()`

---

### Matrix Multiplication

Implemented matrix multiplication supporting:

- 2D matrices
- Batched matrix multiplication
- N-dimensional tensors

Features:

- Batch dimension broadcasting
- Arbitrary batch dimensions
- Shape validation
- Dynamic result shape generation

Helper functions:

- `extract_batch_shape()`
- `build_result_shape()`

---

### Tensor Element Access

Supports:

- N-dimensional access using

```cpp
tensor({i, j, k, ...});
```

---

### Tensor Operations and Python Bindings

The tensor core now exposes common shape and reduction helpers through Python bindings using pybind11:

- `reshape(...)`
- `transpose(...)`
- `sum()` and `sum(axis)`
- `mean()` and `mean(axis)`
- `max()` and `min()`

These are demonstrated in the Python example script:

```bash
cd Code
python3 python/demo.py
```

The same script also exercises:

- tensor arithmetic and matrix multiplication
- indexing and assignment
- activation methods (`relu`, `sigmoid`, `softmax`)
- activation layer objects (`ReLU`, `Sigmoid`, `Softmax`)

---

### Operators

Implemented:

- `operator==`
- Copy assignment
- Move assignment
- Stream operator

Example:

```cpp
std::cout << tensor << std::endl;
```

---
### Shape Operations and Reductions

- Added true n-dimensional transpose support to `Tensor::Transpose(const Shape& axis) const`.
- Updated shape operation tests to cover:
  - 2D transpose with default and explicit axes
  - 1D transpose behavior
  - 3D transpose with default reverse-axes behavior
  - 3D transpose with explicit axis permutation
- Kept `reShape` behavior unchanged and verified it still preserves element order.

### Activation Layer Module

Added a dedicated activation module outside the tensor core to support layer-style usage and Python access:

- `ReLU`
- `Sigmoid`
- `Tanh`
- `GELU`
- `Softmax` with configurable dimension support

The activation classes are implemented as callable objects inheriting from an abstract `Activation` base class and forward to the corresponding tensor methods.

### Layer Module

Added simple neural-network-style layer components:

- `Linear` layer with optional bias support
- `Dropout` layer with training/inference modes
- `Sequential` layer for composing modules

These layers are organized in their own module folders and include dedicated unit tests.

### Loss Module

Added the loss criteria used to train the layers above:

- `MSELoss`
- `CrossEntropyLoss`
- `BinaryCrossEntropyLoss`

All three derive from an abstract `Loss` base class built to the same shape as `Layer`
(pure-virtual `forward` plus a callable `operator()`), but with a two-input signature:

```cpp
MSELoss criterion;
Tensor loss = criterion(predictions, targets);
```

`Loss` does not inherit from `Layer`, because `Layer::forward(x, training)` takes a single
input. Its `forward` is `const`, since unlike `Dropout` no criterion carries state.

#### Reduction Modes

Every criterion accepts a `Reduction`, mirroring PyTorch:

| Mode | Result |
| --- | --- |
| `Reduction::Mean` (default) | scalar tensor of shape `{1}` |
| `Reduction::Sum` | scalar tensor of shape `{1}` |
| `Reduction::None` | the unreduced per-element / per-sample tensor |

#### Formulas

| Criterion | Formula | Input |
| --- | --- | --- |
| `MSELoss` | `mean((prediction - target)^2)` | any matching shapes |
| `CrossEntropyLoss` | `-mean_i log_softmax(logits_i)[y_i]` | logits `{C}` or `{N, C}` + class indices |
| `BinaryCrossEntropyLoss` | `-mean(y*log(p) + (1-y)*log(1-p))` | probabilities and labels in `[0, 1]` |

Complexity is `O(n)` for `MSELoss` and `BinaryCrossEntropyLoss`, and `O(N * C)` for
`CrossEntropyLoss`.

#### Numerical Stability

- `CrossEntropyLoss` applies the softmax internally and never composes `log(softmax(x))`.
  It routes through `Tensor::log_softmax`, which uses the Log-Sum-Exp form
  `log_softmax(x_i) = (x_i - m) - log(sum_j exp(x_j - m))` with `m = max_j x_j`.
  Every exponent is therefore `<= 0`, so `exp()` cannot overflow, and the sum is always
  `>= 1`, so its logarithm is always finite. Logits of `1e3` stay exact where the naive
  composition returns `NaN` or `-inf`.
- `BinaryCrossEntropyLoss` clips probabilities into `[eps, 1 - eps]` before any logarithm,
  bounding a confidently-wrong element at `-log(eps)` (~16.1 for the default `1e-7`)
  instead of letting a single `log(0) = -inf` poison the whole batch mean.
- Criteria refuse to broadcast mismatched shapes. `{2, 1}` against `{2}` would silently
  broadcast to `{2, 2}` under plain tensor arithmetic and yield a meaningless loss.

#### New Tensor Operations

The loss module required three tensor primitives, which were added to the tensor core
rather than special-cased inside the criteria:

- `Tensor::log_softmax(dim)` — stable fused log-softmax
- `Tensor::log()` — element-wise natural logarithm, rejecting non-positive input
- `Tensor::clamp(min, max)` — element-wise bounding, used for the BCE epsilon clipping

### New Files Added and Changed

- `Code/include/activations/activations.h`
  - Added the activation base class and concrete activation layer wrappers.

- `Code/lib/activations/activations.cpp`
  - Implemented the activation layer wrappers.

- `Code/include/tensor/tensor.h`
  - Updated tensor API to include activation methods and support for dimension-aware softmax.

- `Code/lib/tensor/Tensor_ActivationOps.cpp`
  - Implemented tensor-level activation operations, including stable softmax and GELU approximation.

- `Code/bindings/bindings.cpp`
  - Exposed tensor operations, activation methods, and activation layer classes to Python via pybind11.

- `Code/tests/test_tensor_activations.cpp`
  - Added regression tests for activation operations.

- `Code/tests/layer/test_linear.cpp`
  - Added tests for the linear layer.

- `Code/tests/layer/test_dropout.cpp`
  - Added tests for the dropout layer.

- `Code/tests/layer/test_sequential.cpp`
  - Added tests for the sequential container.

- `Code/lib/layer/linear.cpp`
- `Code/lib/layer/dropout.cpp`
- `Code/lib/layer/sequential.cpp`
  - Added the core layer implementations.

- `python/demo.py`
  - Added example scenarios demonstrating the implemented tensor and layer features.

- `Code/include/losses/loss.h`
  - Added the `Reduction` enum and the abstract `Loss` base class with its shared
    reduction and validation helpers.

- `Code/include/losses/mse_loss.h`
- `Code/include/losses/cross_entropy_loss.h`
- `Code/include/losses/binary_cross_entropy_loss.h`
  - Added the three concrete criteria.

- `Code/lib/losses/loss.cpp`
  - Implemented the shared reduction and shape/emptiness validation.

- `Code/lib/losses/mse_loss.cpp`
- `Code/lib/losses/cross_entropy_loss.cpp`
- `Code/lib/losses/binary_cross_entropy_loss.cpp`
  - Implemented the criteria on top of existing tensor arithmetic.

- `Code/lib/tensor/Tensor_ActivationOps.cpp`
  - Added `Tensor::log_softmax` next to `Tensor::softmax`, plus a shared `normalize_dim`
    helper for resolving negative dimensions.

- `Code/lib/tensor/tensor_arithmetic.cpp`
  - Added `Tensor::log` and `Tensor::clamp`.

- `Code/bindings/loss.cpp`
  - Exposed `Reduction` and all three criteria to Python.

- `Code/tests/losses/test_mse_loss.cpp`
- `Code/tests/losses/test_cross_entropy_loss.cpp`
- `Code/tests/losses/test_binary_cross_entropy_loss.cpp`
- `Code/tests/tensor/test_tensor_log_softmax.cpp`
  - Added unit tests for the criteria and the new tensor primitives.

- `python/loss.py`
  - Added a demo covering every criterion, reduction mode and both target forms.

### Problems encountered and fixed

- The original transpose implementation only handled 2D tensors and rejected higher-rank tensors.
- The first generic transpose implementation had an incorrect `ravel_index` argument order, which produced wrong element placement.
- The 3D test expectations were also updated once the correct index mapping became clear.


---

### Testing

Implemented unit tests using GoogleTest.

Coverage includes:

- Tensor properties
- Arithmetic operations
- Broadcasting
- Matrix multiplication
- Scalar operations
- Exception handling
- Batch matrix multiplication
- N-dimensional matrix multiplication
- Activation functions
- Stable softmax along a specified dimension
- Linear layer behavior
- Dropout layer behavior
- Sequential module composition
- Stable log-softmax, element-wise log and clamp
- Loss values, reduction modes and batch handling
- Loss input validation (mismatched shapes, invalid class indices, out-of-range probabilities)
- Numerical stability under very large and very negative logits
- BCE epsilon clipping

The full suite currently reports 168 passing tests, and the Python demos run successfully.

---

### Build Improvements

- Refactored test CMake using helper function

```cmake
add_tensor_test(...)
```

- Enabled compiler warnings

```text
-Wall
-Wextra
-Wpedantic
```

---

## Project Structure

```
Code/
├── include/
│   ├── activations/
│   │   └── activations.h
│   ├── losses/
│   │   ├── binary_cross_entropy_loss.h
│   │   ├── cross_entropy_loss.h
│   │   ├── loss.h
│   │   └── mse_loss.h
│   └── tensor/
│       └── tensor.h
│
├── lib/
│   ├── activations/
│   │   └── activations.cpp
│   ├── layer/
│   │   ├── dropout.cpp
│   │   ├── linear.cpp
│   │   └── sequential.cpp
│   ├── losses/
│   │   ├── binary_cross_entropy_loss.cpp
│   │   ├── cross_entropy_loss.cpp
│   │   ├── loss.cpp
│   │   └── mse_loss.cpp
│   └── tensor/
│       ├── Tensor_ActivationOps.cpp
│       ├── tensor_access.cpp
│       ├── tensor_arithmetic.cpp
│       ├── tensor_broadcast.cpp
│       ├── tensor_matrix.cpp
│       ├── tensor_properties.cpp
│       ├── tensor_reduction.cpp
│       └── tensor_shapeOps.cpp
├── tests/
│   ├── layer/
│   │   ├── test_dropout.cpp
│   │   ├── test_linear.cpp
│   │   └── test_sequential.cpp
│   ├── losses/
│   │   ├── test_binary_cross_entropy_loss.cpp
│   │   ├── test_cross_entropy_loss.cpp
│   │   └── test_mse_loss.cpp
│   ├── tensor/
│   │   ├── test_tensor_arithmetic.cpp
│   │   ├── test_tensor_broadcast.cpp
│   │   ├── test_tensor_matmul.cpp
│   │   ├── test_tensor_properties.cpp
│   │   ├── test_tensor_reduction.cpp
│   │   ├── test_tensor_shapeOps.cpp
│   │   ├── test_tensor_log_softmax.cpp
│   │   └── test_tensor_activations.cpp
│   └── CMakeLists.txt
├── python/
│   ├── demo.py
│   └── loss.py
└── CMakeLists.txt
```

---

## Future Improvements

- Support chained indexing

```cpp
tensor[i][j][k]
```

using proxy classes.

- Optimize matrix multiplication performance
  - Cache-friendly implementation
  - SIMD/vectorization
  - Parallel execution
  - Blocked matrix multiplication

---

## Technologies

- C++17
- CMake
- GoogleTest
