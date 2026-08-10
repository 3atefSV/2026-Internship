# TinyTorch

A lightweight, PyTorch-like deep learning framework implemented in **C++17**, with a Python interface powered by **PyBind11**.

TinyTorch is built from scratch to explore the internal architecture of modern deep learning frameworks while providing a foundation for performance-oriented ML and embedded systems.

---

## ✨ Features

* 🧮 **N-Dimensional Tensor Core**

  * Broadcasting
  * Arithmetic operations
  * Reductions
  * Reshape & transpose
  * Matrix multiplication

* ⚙️ **Autograd**

  * Reverse-mode automatic differentiation
  * Dynamic computation graphs
  * Gradient tracking & accumulation

* 🧠 **Neural Networks**

  * Linear
  * Dropout
  * Sequential
  * ReLU, Sigmoid, Tanh, GELU, Softmax

* 📉 **Loss Functions**

  * MSE
  * Cross Entropy
  * Binary Cross Entropy
  * Mean, Sum, and None reductions

* 📊 **Data Pipeline**

  * Dataset & TensorDataset
  * ImageDataset
  * Transforms
  * DataLoader
  * Batching & shuffling

* ⚡ **CPU Optimizations**

  * OpenMP parallelism
  * SIMD vectorization
  * Cache-friendly access
  * Optimized arithmetic & matrix multiplication

---

## 📊 Performance

Tensor operations are optimized across arithmetic, broadcasting, and matrix multiplication.

### Matrix Multiplication Benchmark

5 iterations after a warmup pass:

| Matrix Size   | Total Time | Average Time |
| ------------- | ---------: | -----------: |
| `128 × 128`   | `8.1362 s` | `1627.24 ms` |
| `1024 × 1024` | `0.9590 s` |  `191.81 ms` |

> Results depend on the CPU and build configuration. Different matrix sizes should not be interpreted as a direct speedup comparison.

Run the benchmark:

```bash
cd Code
python3 python/benchmark.py
```

---

## 🎯 Training Demo

`Code/python/demo.py` provides an end-to-end training example using:

* 40,000 XOR samples
* `DataLoader` with batch size `2048`
* Wide MLP: `2 → 1024 → 1`
* ReLU + Sigmoid
* MSE Loss
* Autograd backpropagation
* 2 training epochs

Run it with:

```bash
cd Code
python3 python/demo.py
```

---

## 📚 Documentation

Detailed implementation documentation:

| Documentation                                                | Description                                                                       |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| 🧮 [Tensor Core & Operations](Documentation/tensor_core.md)  | Tensor operations, broadcasting, reductions, transpose, and matrix multiplication |
| 🧠 [Neural Network Modules](Documentation/neural_network.md) | Layers, activations, and loss functions                                           |
| ⚙️ [Autograd Engine](Documentation/autograd_engine.md)       | Computation graphs and gradient propagation                                       |
| 📊 [Data Pipeline](Documentation/data_pipeline.md)           | Datasets, transforms, batching, and DataLoader                                    |

---

## 📁 Project Structure

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
    ├── include/                # Public C++ headers
    ├── lib/                    # C++ implementations
    ├── bindings/               # PyBind11 bindings
    ├── python/                 # Python API & examples
    └── tests/                  # GoogleTest suites
```

---

## 🛠️ Build

### Requirements

* C++17
* CMake
* Ninja
* OpenMP
* Python
* PyBind11
* GoogleTest

### Build

```bash
cd Code

mkdir build
cd build

cmake -G Ninja ..
ninja
```

### Run Tests

```bash
ctest --output-on-failure
```

### Run Python Examples

```bash
cd ..

python3 python/tensor.py
python3 python/loss.py
python3 python/autograd.py
```

---

## 💻 Technologies

* **C++17** — Core backend
* **CMake** — Build system
* **Ninja** — Build execution
* **OpenMP** — CPU parallelism
* **GoogleTest** — Unit testing
* **PyBind11** — C++/Python interoperability
* **Python** — API and examples

---

## 🚀 Roadmap

* Python test suite with `pytest`
* SGD & Adam optimizers
* Complete training pipelines
* Further CPU optimization
* Blocked / tiled matrix multiplication
* Memory reuse
* CUDA acceleration
* Chained Tensor indexing

---

## 🎯 Project Goals

TinyTorch aims to provide both a functional lightweight deep learning framework and a hands-on implementation of modern ML infrastructure.

The project focuses on:

**Tensor → Operations → Autograd → Neural Networks → Data Pipeline → Training**

with an emphasis on understanding and optimizing ML infrastructure for future embedded systems.
