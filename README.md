# TinyTorch Tensor Library

A lightweight, PyTorch-like Tensor library and Deep Learning framework implemented in modern C++17 as part of the 2026 Embedded Software Internship.

## 🚀 Overview

TinyTorch is designed to provide a deep understanding of how modern deep learning frameworks operate under the hood. It features a fully functional dynamic N-dimensional tensor core, neural network layers, mathematically stable loss functions, a robust data loading pipeline, and a dynamic Autograd engine for reverse-mode automatic differentiation.

The C++ backend is seamlessly exposed to Python using **PyBind11**, allowing the framework to be used through a familiar Python API while keeping the core implementation in modern C++17.

---

## 📚 Documentation

Detailed explanations of the library's internal components are organized in the `Documentation/` directory:

* 🧮 **[Tensor Core & Operations](Documentation/tensor_core.md)**
  Dynamic N-dimensional tensors, broadcasting, optimized arithmetic, matrix multiplication, reductions, indexing, and shape operations.

* 🧠 **[Neural Network Modules](Documentation/neural_network.md)**
  Linear layers, Dropout, Sequential containers, activations, and numerically stable loss functions.

* ⚙️ **[Autograd Engine](Documentation/autograd_engine.md)**
  Reverse-mode automatic differentiation, computation graphs, backward functions, and gradient accumulation.

* 📊 **[Data Pipeline](Documentation/data_pipeline.md)**
  Datasets, image loading, transforms, batching, shuffling, and the DataLoader.

---

## 📁 Project Structure

The project is organized to separate public interfaces, C++ implementations, Python bindings, high-level Python APIs, documentation, and tests.

```text
TinyTorch/
├── Documentation/
│   ├── README.md
│   ├── tensor_core.md
│   ├── neural_network.md
│   ├── autograd_engine.md
│   └── data_pipeline.md
│
└── Code/
    ├── CMakeLists.txt
    ├── .clang-format
    │
    ├── include/                # Public C++ Headers
    │   ├── activations/
    │   ├── autograd/
    │   ├── data/
    │   ├── layer/
    │   ├── losses/
    │   └── tensor/
    │
    ├── lib/                    # C++ Core Implementations
    │   ├── CMakeLists.txt
    │   ├── activations/
    │   ├── autograd/
    │   ├── data/
    │   ├── layer/
    │   ├── losses/
    │   └── tensor/
    │
    ├── bindings/               # PyBind11 Bindings
    │   ├── CMakeLists.txt
    │   ├── core/
    │   ├── data/
    │   ├── layer/
    │   ├── losses/
    │   └── transforms/
    │
    ├── python/                 # Python Frontend & Examples
    │   ├── autograd.py
    │   ├── benchmark.py
    │   ├── dataloader.py
    │   ├── dataset.py
    │   ├── layer.py
    │   ├── loss.py
    │   ├── tensor.py
    │   └── transforms.py
    │
    └── tests/                  # GoogleTest Suites
        ├── CMakeLists.txt
        ├── activations/
        ├── autograd/
        ├── data/
        ├── layer/
        ├── losses/
        └── tensor/
```

---

## ⚡ Performance Optimization

The Tensor core has been optimized across multiple computational paths rather than focusing exclusively on matrix multiplication.

The optimization work targets:

* Tensor-to-Tensor arithmetic.
* Tensor-to-scalar arithmetic.
* Broadcasting operations.
* Matrix multiplication.
* Element-wise mathematical operations.
* Memory access and indexing overhead.

### Optimized Tensor Arithmetic

For element-wise operations such as:

```text
+
-
*
/
```

TinyTorch provides a dedicated fast path when both tensors have identical shapes.

Instead of performing broadcasting and multidimensional index calculations for every element, the implementation directly accesses the contiguous storage:

```cpp
data_[i]
other.data_[i]
resultData[i]
```

This avoids unnecessary index transformations and significantly reduces overhead for the common same-shape case.

### Parallel SIMD Execution

Operations that can safely execute without exceptions use OpenMP parallelization and SIMD vectorization:

```cpp
#pragma omp parallel for simd schedule(static)
```

This allows independent elements to be processed concurrently across CPU threads while also providing SIMD opportunities to the compiler.

The approach is used for suitable operations such as:

* Same-shape arithmetic.
* Tensor-scalar arithmetic.
* `clamp()`.

### Broadcasting Optimization

Broadcasting operations use a separate execution path because they require index mapping between different Tensor shapes.

For operations that do not require exception handling, the broadcasting loop is parallelized using OpenMP:

```cpp
#pragma omp parallel for schedule(static)
```

Each output element independently calculates its corresponding broadcasted input indices.

### Safe Exception Handling

Operations that may need to throw exceptions during element processing are intentionally kept sequential.

For example, tensor division validates every divisor before performing the operation:

```cpp
if (other.data_[i] == 0) {
    throw std::invalid_argument(
        "Division by zero in tensor-tensor operation."
    );
}
```

This avoids introducing exception-related complications inside parallel regions while preserving the original validation behavior.

The same principle is applied to operations such as logarithm and scalar/tensor division where invalid input must be detected safely.

### Optimized Matrix Multiplication

Matrix multiplication uses additional HPC optimizations:

* OpenMP parallel execution.
* SIMD vectorization.
* Cache-friendly access patterns.
* Transposition of the right-hand matrix.
* Reduced multidimensional indexing overhead.
* Batched matrix multiplication.
* Batch-dimension broadcasting.

The implementation calculates batch base offsets once and then uses direct flat indexing inside the computational `O(N × M × K)` loops.

---

## 📊 Matrix Multiplication Benchmark

The optimized matrix multiplication implementation was benchmarked using the Python frontend.

Each benchmark performs 5 iterations after a warmup pass.

### `128 × 128`

```text
Matrix Size  : 128x128
Total Time   : 8.1362 seconds
Average Time : 1627.24 ms per operation
```

### `1024 × 1024`

```text
Matrix Size  : 1024x1024
Total Time   : 0.9590 seconds
Average Time : 191.81 ms per operation
```

| Matrix Size   | Iterations | Total Time | Average Time |
| ------------- | ---------: | ---------: | -----------: |
| `128 × 128`   |          5 | `8.1362 s` | `1627.24 ms` |
| `1024 × 1024` |          5 | `0.9590 s` |  `191.81 ms` |

> **Benchmark Note:** These measurements use different matrix sizes and should not be interpreted as a direct speedup comparison. They document the observed execution times of the optimized implementation for each workload.

Run the benchmark with:

```bash
cd Code
python3 python/benchmark.py
```

---

## 🛠️ Build and Testing

TinyTorch uses **GoogleTest** for C++ unit testing, while Python scripts validate the PyBind11 integration and provide examples and benchmarks.

### Build Improvements

The build system includes:

* Refactored test CMake configuration using the helper function `add_tensor_test(...)`.
* Strict compiler warnings:

  * `-Wall`
  * `-Wextra`
  * `-Wpedantic`
* Native CPU optimization through:

  * `-O3`
  * `-march=native`
* OpenMP support for parallel execution.
* Ninja support for fast incremental builds.

### Building the Project

```bash
cd Code

mkdir build
cd build

cmake -G Ninja ..
ninja
```

### Running C++ Tests

```bash
ctest --output-on-failure
```

### Running Python Examples

From the `Code/` directory:

```bash
python3 python/tensor.py
python3 python/loss.py
python3 python/autograd.py
```

### Running the Benchmark

```bash
python3 python/benchmark.py
```

---

## 🐛 Development Notes

### Generic N-Dimensional Transpose

The original transpose implementation only supported **2D tensors** and rejected higher-rank tensors.

The implementation was generalized to support arbitrary N-dimensional tensors using axis permutations and multidimensional index remapping.

### `ravel_index` Mapping Bug

The first generic transpose implementation used an incorrect argument order when calling `ravel_index()`.

This caused incorrect element placement after transposition.

The index mapping was corrected and validated using higher-dimensional Tensor cases.

### 3D Transpose Tests

The expected results of the 3D transpose tests were updated after verifying the correct N-dimensional index mapping.

This ensured that the tests matched the mathematical definition of axis permutation rather than the behavior of the previous implementation.

---

## 🚀 Future Improvements

### Python Testing Suite

Implement a comprehensive Python test suite using `pytest` to systematically validate the PyBind11 API.

### Model Training Pipeline

Build complete end-to-end training workflows with optimizers such as:

* SGD
* Adam

This will allow the Autograd engine to be used in complete neural network training pipelines.

### Advanced CPU Optimization

Further optimize the Tensor backend through:

* Blocked / tiled matrix multiplication.
* Better cache-aware algorithms.
* Memory reuse.
* More advanced SIMD strategies.
* Reduction of temporary allocations.
* Further optimization of broadcasting and indexing.

### GPU Acceleration

Explore CUDA-based Tensor operations and matrix multiplication for supported NVIDIA GPUs.

### Tensor Indexing

Support chained indexing such as:

```python
tensor[i][j][k]
```

using proxy classes.

---

## 💻 Technologies

| Technology     | Purpose                                  |
| -------------- | ---------------------------------------- |
| **C++17**      | Core Tensor and Deep Learning backend    |
| **CMake**      | Build system                             |
| **Ninja**      | Fast build execution                     |
| **OpenMP**     | Parallel CPU execution                   |
| **GoogleTest** | C++ unit testing                         |
| **PyBind11**   | C++/Python interoperability              |
| **Python**     | High-level API, examples, and benchmarks |

---

## 🎯 Project Goals

TinyTorch is intended not only to be a functional deep learning framework, but also an educational implementation for understanding the internal architecture of modern frameworks such as PyTorch.

The project focuses on understanding:

* N-dimensional Tensor representation.
* Broadcasting and shape manipulation.
* Efficient Tensor arithmetic.
* CPU parallelism and SIMD vectorization.
* Matrix multiplication optimization.
* Neural network layers.
* Activation functions.
* Numerically stable loss functions.
* Reverse-mode automatic differentiation.
* Computation graphs.
* Gradient propagation and accumulation.
* Data loading pipelines.
* C++/Python interoperability.
* ML infrastructure for embedded systems.

---

## 📌 Project Status

TinyTorch currently provides a functional C++ Tensor backend with Python bindings, optimized Tensor operations, neural network primitives, loss functions, Autograd, and a data loading pipeline.

The project is actively being extended toward a lightweight deep learning framework suitable for experimentation and future embedded ML applications.
