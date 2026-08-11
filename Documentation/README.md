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

### Activation Functions

Added activation-layer style tensor methods to the public API:

- `relu()` for element-wise ReLU
- `sigmoid()` using a numerically stable implementation
- `tanh()` for hyperbolic tangent
- `gelu()` using the fast sigmoid-based approximation: $GELU(x) = x \cdot \sigma(1.702x)$
- `softmax(int dim = -1)` for stable softmax over either the whole tensor or a specified dimension

The softmax implementation now supports multidimensional tensors, handles negative dimension indices, and uses max-subtraction before exponentiation for numerical stability.

### New Files Added and Changed

- `Code/include/tensor.h`
  - Added declarations for the activation methods and the dimension-aware softmax signature.

- `Code/lib/Tensor_ActivationOps.cpp`
  - Implemented the activation functions and the stable multidimensional softmax logic.

- `Code/tests/test_tensor_activations.cpp`
  - Added regression tests for ReLU, sigmoid, tanh, GELU, and softmax (including softmax over a specific dimension).

- `Code/lib/tensor_shapeOps.cpp`
  - Rewrote `Transpose` to support arbitrary permutations for tensors of any rank.
  - Added validation for axis length, axis range, and duplicate axes.
  - Used `unravel_index` and `ravel_index` to map between flattened storage and multidimensional coordinates.

- `Code/tests/test_tensor_shapeOps.cpp`
  - Added 3D transpose tests and corrected expected output for explicit axis permutations.

- `Code/tests/test_tensor_reduction.cpp`
  - Added tests for the following reductions: sum, mean, max, min.

- `Code/lib/tensor_reduction.cpp`
  - Added implementations for the following reductions: sum, mean, max, min.

- `python/demo.py`
  - Added example scenarios demonstrating the implemented tensor features.

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

Verified with:

```bash
./build/bin/test_tensor_activations
```

The activation test suite currently reports 6 passing tests.

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
│   └── tensor.h
│
├── lib/
│   ├── tensor_access.cpp
│   ├── tensor_arithmetic.cpp
│   ├── tensor_broadcast.cpp
│   ├── tensor_matrix.cpp
│   ├── tensor_properties.cpp
│   ├── tensor_reduction.cpp
│   ├── tensor_shapeOps.cpp
│   └── Tensor_ActivationOps.cpp
├── tests/
│   ├── test_tensor_properties.cpp
│   ├── test_tensor_arithmetic.cpp
│   ├── test_tensor_broadcast.cpp
│   ├── test_tensor_matmul.cpp
│   ├── test_tensor_reduction.cpp
│   ├── test_tensor_shapeOps.cpp
│   └── test_tensor_activations.cpp
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
