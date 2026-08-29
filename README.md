# TinyTorch Tensor Library

A lightweight, PyTorch-like **Tensor and Deep Learning framework** implemented in modern **C++17** as part of the 2026 Embedded Software Internship.

TinyTorch is built from the ground up to explore how modern deep learning frameworks work internally, from N-dimensional tensor operations and broadcasting to automatic differentiation, neural network layers, loss functions, and data loading.

The C++ backend is exposed to Python through **PyBind11**, providing a familiar Python interface while keeping the core implementation in modern C++.

---

## 🚀 Overview

TinyTorch is designed to provide a deep understanding of how modern deep learning frameworks operate under the hood. It features a fully functional dynamic N-dimensional tensor core, neural network layers, mathematically stable loss functions, a robust data loading pipeline, a dynamic Autograd engine for reverse-mode automatic differentiation, and a family of optimizers (SGD, Adam, AdamW) that train the models.

The C++ backend is seamlessly exposed to Python using PyBind11, allowing you to write familiar, Pythonic deep learning code.

The overall architecture can be summarized as:

```text
                    TinyTorch
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
     Tensor          Autograd       Data Pipeline
        │               │                │
        ▼               ▼                ▼
     Layers        Computation       Dataset
        │            Graph             │
        ▼               │              ▼
   Activations          │          Transforms
        │               │              │
        ▼               │              ▼
      Loss ◄────────────┘          DataLoader
        │                              │
        └──────────────┬───────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Optimizers    │  ◄── SGD / Adam / AdamW
              └────────┬────────┘
                       │
                       ▼
                  Training Loop
```

---

# 📚 Documentation

Detailed documentation is organized into separate files under the `Documentation/` directory.

| Component          | Documentation                                          | Description                                                                           |
| ------------------ | ------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| 🧮 Tensor Core     | [tensor_core.md](Documentation/tensor_core.md)         | N-dimensional tensors, broadcasting, reductions, transpose, and matrix multiplication |
| 🧠 Neural Networks | [neural_network.md](Documentation/neural_network.md)   | Layers, activations, Sequential containers, and loss functions                        |
| ⚙️ Autograd        | [autograd_engine.md](Documentation/autograd_engine.md) | Computation graphs, gradient tracking, and reverse-mode automatic differentiation     |
| 📊 Data Pipeline   | [data_pipeline.md](Documentation/data_pipeline.md)     | Datasets, image loading, transforms, batching, and DataLoader                         |
| ⚙️ Optimizers      | [optimizers.md](Documentation/optimizers.md)           | SGD, Adam, and AdamW update rules, weight decay, and memory-efficient step()          |

---

# 📁 Project Structure

The project is organized into separate components for public C++ interfaces, implementations, Python bindings, examples, documentation, and tests.

```text
TinyTorch/
├── Documentation/
│   ├── README.md
│   ├── tensor_core.md
│   ├── neural_network.md
│   ├── autograd_engine.md
│   ├── data_pipeline.md
│   └── optimizers.md
│
└── Code/
    ├── CMakeLists.txt
    ├── .clang-format
    │
    ├── include/                    # Public C++ Headers
    │   ├── activations/
    │   │   └── activations.h
    │   ├── autograd/
    │   │   ├── function.h
    │   │   └── loss_function.h
    │   ├── data/
    │   │   ├── dataloader.h
    │   │   ├── dataset.h
    │   │   ├── image_dataset.h
    │   │   ├── tensor_dataset.h
    │   │   └── transforms.h
    │   ├── layer/
    │   │   ├── layer.h
    │   │   ├── linear.h
    │   │   ├── dropout.h
    │   │   └── sequential.h
    │   ├── losses/
    │   │   ├── loss.h
    │   │   ├── mse_loss.h
    │   │   ├── cross_entropy_loss.h
    │   │   └── ...
    │   ├── optimizers/
    │   │   ├── optimizer.h
    │   │   ├── sgd.h
    │   │   ├── adam.h
    │   │   └── adamw.h
    │   └── tensor/
    │       └── tensor.h
    │
    ├── lib/                        # C++ Core Implementations
    │   ├── CMakeLists.txt
    │   ├── activations/
    │   ├── autograd/
    │   ├── data/
    │   ├── layer/
    │   ├── losses/
    │   ├── optimizers/
    │   └── tensor/
    │
    ├── bindings/                   # PyBind11 Bindings
    │   ├── CMakeLists.txt
    │   ├── core/
    │   │   ├── bindings.h
    │   │   ├── bindings.cpp
    │   │   ├── module.cpp
    │   │   └── tensor.cpp
    │   ├── data/
    │   ├── layer/
    │   ├── losses/
    │   ├── optimizers/
    │   └── transforms/
    │
    ├── python/                     # Python Frontend & Examples
    │   ├── autograd.py
    │   ├── dataloader.py
    │   ├── dataset.py
    │   ├── layer.py
    │   ├── loss.py
    │   ├── optimizers.py
    │   ├── tensor.py
    │   └── transforms.py
    │
    └── tests/                      # GoogleTest Suites
        ├── CMakeLists.txt
        ├── activations/
        ├── autograd/
        ├── data/
        ├── layer/
        ├── losses/
        ├── optimizers/
        └── tensor/
```

---

# 🛠️ Build & Testing

TinyTorch uses **CMake + Ninja** as its build system and **GoogleTest** for C++ unit testing.

The Python examples validate the PyBind11 integration and expose the C++ backend through a Python-friendly API.

## Prerequisites

The project requires:

* C++17-compatible compiler
* CMake
* Ninja
* Python 3
* PyBind11
* GoogleTest

---

## Building

From the project root:

```bash
cd Code

mkdir -p build
cd build

cmake -G Ninja ..
ninja
```

---

## Running Tests

After building:

```bash
ctest --output-on-failure
```

This executes the complete GoogleTest suite and reports detailed output for failing tests.

---

## Running Python Examples

From the `Code/` directory:

```bash
cd ..

python3 python/tensor.py
python3 python/loss.py
python3 python/autograd.py
python3 python/optimizers.py
```

These examples demonstrate the Python interface exposed by the C++ backend through PyBind11.

---

# 🧪 Testing & Code Quality

The project uses GoogleTest to validate the C++ implementation across its major components.

Test coverage is organized by module:

```text
tests/
├── activations/
├── autograd/
├── data/
├── layer/
├── losses/
├── optimizers/
└── tensor/
```

The build configuration also enables strict compiler warnings:

```text
-Wall
-Wextra
-Wpedantic
```

Test CMake configuration was refactored using the helper function:

```cmake
add_tensor_test(...)
```

This keeps test registration consistent and reduces duplicated CMake configuration.

---

# 🐛 Development Notes

Several important issues were identified and fixed during development.

## Generic N-Dimensional Transpose

The original transpose implementation only supported **2D tensors** and rejected higher-rank tensors.

The implementation was generalized to support arbitrary N-dimensional tensors using axis permutations and index remapping.

---

## `ravel_index` Mapping Bug

The first generic transpose implementation used an incorrect argument order when calling `ravel_index()`.

This caused incorrect element placement after transposition.

The index mapping was corrected and validated against higher-dimensional tensor cases.

---

## 3D Transpose Tests

The expected results of the 3D transpose tests were updated after verifying the correct N-dimensional index mapping.

This ensured that the tests matched the mathematical definition of axis permutation rather than the behavior of the previous implementation.

---

# 🚀 Future Improvements

TinyTorch is still under active development. Planned improvements include both API features and performance optimizations.

## Tensor Indexing

Add support for chained indexing such as:

```python
tensor[i][j][k]
```

using proxy classes.

This would provide a more natural indexing interface for multidimensional tensors.

---

## Matrix Multiplication Optimization

The current matrix multiplication implementation can be further optimized through:

* Cache-friendly memory access
* SIMD/vectorization
* Parallel execution
* Blocked/Tiled matrix multiplication
* Improved memory locality

These optimizations are particularly relevant for the project's eventual embedded and hardware-oriented targets.

---

# 💻 Technologies

| Technology     | Purpose                               |
| -------------- | ------------------------------------- |
| **C++17**      | Core Tensor and Deep Learning backend |
| **CMake**      | Build system                          |
| **Ninja**      | Fast build execution                  |
| **GoogleTest** | C++ unit testing                      |
| **PyBind11**   | C++/Python interoperability           |
| **Python**     | High-level API and examples           |

---

# 🎯 Project Goals

TinyTorch is intended to be more than a collection of Tensor operations.

The main goal is to understand and implement the fundamental building blocks behind modern deep learning frameworks while maintaining a clean, modular C++ architecture.

The project focuses on understanding:

* N-dimensional tensor representation
* Contiguous memory and indexing
* Broadcasting and shape manipulation
* Matrix multiplication
* Reduction operations
* Neural network layers
* Activation functions
* Numerically stable loss functions
* Reverse-mode automatic differentiation
* Dynamic computation graphs
* Gradient propagation and accumulation
* Optimization algorithms (SGD, Adam, AdamW)
* Memory-efficient, allocation-free parameter updates
* Dataset abstractions
* Data preprocessing
* Mini-batch data loading
* C++/Python interoperability
* Building ML infrastructure for embedded systems

---

# 📌 Current Status

TinyTorch currently provides a functional foundation covering the complete path from data loading through optimization:

```text
Raw Data
   │
   ▼
Dataset
   │
   ▼
Transforms
   │
   ▼
DataLoader
   │
   ▼
Tensor
   │
   ▼
Neural Network
   │
   ▼
Loss
   │
   ▼
Autograd
   │
   ▼
Gradients
   │
   ▼
Optimizer (SGD / Adam / AdamW)
```

The architecture is intentionally modular so that additional layers, operations, optimizers, datasets, and hardware-specific backends can be added without redesigning the existing core.
