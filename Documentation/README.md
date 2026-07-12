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
│   ├── tensor.cpp
│   ├── tensor_access.cpp
│   ├── tensor_arithmetic.cpp
│   ├── tensor_broadcast.cpp
│   ├── tensor_matrix.cpp
│   └── tensor_print.cpp
│
├── tests/
│   ├── test_tensor_properties.cpp
│   ├── test_tensor_arithmetic.cpp
│   ├── test_tensor_broadcast.cpp
│   └── test_tensor_matmul.cpp
│
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
