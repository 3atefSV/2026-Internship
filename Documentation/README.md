# Tensor Library (C++)

A lightweight Tensor library implemented in modern C++ as part of an Embedded Software internship. This project represents the foundation of a AI framework built from scratch, with the Tensor library serving as the core component for future numerical operations, automatic differentiation, and deep learning modules.

---

## Features

### Tensor Properties

- Create tensors from data and shape
- Get tensor shape
- Get tensor size
- Get number of dimensions
- Check if a tensor is empty
- Get tensor data type

### Arithmetic Operations

#### Tensor ↔ Tensor

- Addition (`+`)
- Subtraction (`-`)
- Multiplication (`*`)
- Division (`/`)

#### Tensor ↔ Scalar

- Addition (`+`)
- Subtraction (`-`)
- Multiplication (`*`)
- Division (`/`)

### Matrix Operations

- Matrix multiplication (`matmul`)

### Element Access

Access tensor elements using:

```cpp
tensor(row, column)
```

Example:

```cpp
Tensor t(
    {1,2,
     3,4},
    {2,2}
);

std::cout << t(1,0);   // 3

t(0,1) = 100;
```

---

# Project Structure

```
Tensor/
│
├── CMakeLists.txt          # Root CMake
│
├── include/
│   ├── tensor.h
│   └── CMakeLists.txt
│
├── lib/
│   ├── tensor.cpp
│   └── CMakeLists.txt
│
├── tests/
│   ├── test_tensor_properties.cpp
│   ├── test_tensor_arithmetic.cpp
│   ├── test_tensor_matmul.cpp
│   └── CMakeLists.txt
│
└── README.md
```

---

# Build

Create the build directory and generate the project:

```bash
mkdir build
cd build

cmake ..
```

Build the project:

```bash
cmake --build .
```

---

# Run Tests

Run all unit tests:

```bash
ctest
```

or execute the test binary directly:

```bash
./bin/test_tensor_properties
```

---

# Example

```cpp
Tensor a(
    {1,2,
     3,4},
    {2,2}
);

Tensor b(
    {5,6,
     7,8},
    {2,2}
);

Tensor c = a + b;

Tensor d = a.matmul(b);
```

---

# Unit Testing

The project uses **GoogleTest** to verify:

- Tensor properties
- Arithmetic operations
- Matrix multiplication
- Element access
- Exception handling

---

# Technologies

- C++17
- CMake (Multi-directory project)
- GoogleTest
- Git