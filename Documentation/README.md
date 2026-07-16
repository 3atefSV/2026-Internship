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


The full suite currently reports 76 passing tests, and the Python demo runs successfully.

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
│   ├── tensor/
│   │   ├── test_tensor_arithmetic.cpp
│   │   ├── test_tensor_broadcast.cpp
│   │   ├── test_tensor_matmul.cpp
│   │   ├── test_tensor_properties.cpp
│   │   ├── test_tensor_reduction.cpp
│   │   ├── test_tensor_shapeOps.cpp
│   │   └── test_tensor_activations.cpp
│   └── CMakeLists.txt
├── python/
│   └── demo.py
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
