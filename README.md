# TinyTorch Tensor Library

A lightweight, PyTorch-like **Tensor and Neural Network framework** implemented in modern **C++17** as part of the 2026 Embedded Software Internship.

TinyTorch is built from the ground up to explore how modern deep learning frameworks work internally, from N-dimensional tensor operations and broadcasting to automatic differentiation, neural network layers, loss functions, data loading, and optimization.

The C++ backend is exposed to Python through **PyBind11**, providing a familiar Python interface while keeping the core implementation in modern C++.

---

## 🚀 Overview

TinyTorch provides a modular deep learning infrastructure covering the complete training pipeline:

* N-dimensional Tensor operations
* Broadcasting and reductions
* Matrix multiplication
* Neural network layers and activations
* Loss functions
* Reverse-mode Autograd
* Dynamic computation graphs
* Dataset and DataLoader abstractions
* SGD, Adam, and AdamW optimizers
* Gradient clipping
* Learning-rate scheduling
* Training orchestration
* Binary checkpointing
* C++/Python interoperability

The overall architecture can be summarized as:

```text
                         TinyTorch
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
       Tensor             Autograd        Data Pipeline
          │                  │                  │
          ▼                  ▼                  ▼
       Layers          Computation          Dataset
          │               Graph                │
          ▼                  │                 ▼
    Activations              │             Transforms
          │                  │                 │
          ▼                  │                 ▼
        Loss ◄───────────────┘             DataLoader
          │
          └──────────────────┬──────────────────┐
                             │                  │
                             ▼                  ▼
                       Optimizers           Trainer
                    SGD / Adam / AdamW          │
                             │                  │
                             └────────┬─────────┘
                                      ▼
                               Training Pipeline
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
| ⚙️ Optimizers      | [optimizers.md](Documentation/optimizers.md)           | SGD, Adam, and AdamW update rules, weight decay, and optimizer state                  |
| 🎯 Training        | [training.md](Documentation/training.md)               | Trainer, gradient accumulation, clipping, scheduling, and checkpointing               |

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
│   ├── optimizers.md
│   └── training.md
│
└── Code/
    ├── CMakeLists.txt
    ├── .clang-format
    │
    ├── include/
    │   ├── activations/
    │   ├── autograd/
    │   ├── data/
    │   ├── layer/
    │   ├── losses/
    │   ├── optimizers/
    │   ├── tensor/
    │   └── training/
    │       ├── trainer.h
    │       ├── scheduler.h
    │       └── checkpoint.h
    │
    ├── lib/
    │   ├── CMakeLists.txt
    │   ├── activations/
    │   ├── autograd/
    │   ├── data/
    │   ├── layer/
    │   ├── losses/
    │   ├── optimizers/
    │   ├── tensor/
    │   └── training/
    │
    ├── bindings/
    │   ├── CMakeLists.txt
    │   ├── core/
    │   ├── data/
    │   ├── layer/
    │   ├── losses/
    │   ├── optimizers/
    │   ├── transforms/
    │   └── training/
    │
    ├── python/
    │   ├── autograd.py
    │   ├── dataloader.py
    │   ├── dataset.py
    │   ├── layer.py
    │   ├── loss.py
    │   ├── optimizers.py
    │   ├── tensor.py
    │   ├── transforms.py
    │   └── training.py
    │
    └── tests/
        ├── activations/
        ├── autograd/
        ├── data/
        ├── layer/
        ├── losses/
        ├── optimizers/
        ├── tensor/
        └── training/
```

---

# 🛠️ Build & Testing

TinyTorch uses **CMake + Ninja** as its build system and **GoogleTest** for C++ unit testing.

## Prerequisites

* C++17-compatible compiler
* CMake
* Ninja
* Python 3
* PyBind11
* GoogleTest
* OpenMP

## Building

From the project root:

```bash
cd Code
mkdir -p build
cd build
cmake -G Ninja ..
ninja
```

## Running Tests

```bash
ctest --output-on-failure
```

This executes the GoogleTest suite and reports detailed output for failing tests.

---

# 🧠 Training Infrastructure

TinyTorch includes a C++17 training infrastructure that connects the Tensor engine, Autograd, neural-network modules, DataLoader, loss functions, and optimizers into a unified training pipeline.

## Trainer

The `Trainer` class orchestrates the complete training lifecycle:

```text
DataLoader
    │
    ▼
Forward Pass
    │
    ▼
Loss Computation
    │
    ▼
Backward Pass
    │
    ▼
Gradient Accumulation
    │
    ▼
Gradient Clipping
    │
    ▼
Optimizer Step
    │
    ▼
Zero Gradients
```

The Trainer also supports:

* Mini-batch training
* Gradient accumulation
* Gradient clipping
* Learning-rate scheduling
* Training history
* Epoch and step tracking
* Checkpoint management

## Gradient Accumulation

Gradient accumulation allows multiple mini-batches to contribute to one optimizer update.

```text
Batch 1 ──► Forward ──► Backward ──┐
Batch 2 ──► Forward ──► Backward ──┤
Batch 3 ──► Forward ──► Backward ──┤──► Optimizer Step
Batch 4 ──► Forward ──► Backward ──┘
```

This allows the effective batch size to be increased without requiring the entire batch to fit into memory at once.

## Global Norm Gradient Clipping

`clip_grad_norm` computes the global norm across parameter gradients and scales gradients when they exceed a configured threshold.

This helps prevent unstable parameter updates caused by exploding gradients while preserving the relative direction of the gradient vector.

## Cosine Learning Rate Schedule

`CosineSchedule` provides smooth learning-rate annealing between configurable `max_lr` and `min_lr` values over the training epochs.

```text
Learning Rate
     │
max  │──────╮
     │       ╲
     │        ╲
     │         ╲
     │          ╲──────
min  │
     └──────────────────► Epoch
```

## Binary Checkpointing

TinyTorch supports binary checkpointing for saving and restoring training state.

Checkpoint data can include:

* Model parameters
* Epoch
* Training step
* Optimizer/training state

This allows training to resume after interruption without restarting from the beginning.

---

# ⚡ Performance Optimizations

Several parts of the Tensor and training engine were optimized using low-level C++ techniques.

## Parallel Tensor Operations

Independent tensor element operations use OpenMP parallel execution where appropriate.

```cpp
#pragma omp parallel for simd schedule(static)
for (size_type i = 0; i < resultData.size(); ++i) {
    resultData[i] = op(data_[i], other.data_[i]);
}
```

The implementation also provides specialized execution paths for tensors with identical shapes and tensors requiring broadcasting.

## SIMD Vectorization

Element-wise operations on contiguous data can use compiler-generated SIMD instructions through OpenMP SIMD directives.

This allows multiple independent elements to be processed in parallel by the CPU's vector units.

## Cache-Friendly Matrix Multiplication

Matrix multiplication was optimized by improving memory locality.

The right-hand matrix can be rearranged into a transposed layout so that the inner multiplication loop accesses contiguous memory instead of repeatedly traversing matrix columns.

Additional optimization directions include:

* Cache-friendly memory access
* SIMD/vectorization
* OpenMP parallel execution
* Blocked/tiled matrix multiplication
* Improved memory locality

---

# 🧪 Stress Test & Benchmark

TinyTorch was evaluated using a synthetic XOR training workload designed to exercise the Tensor, Autograd, neural-network, DataLoader, and optimizer components together.

### Workload

| Parameter        | Configuration                                        |
| ---------------- | ---------------------------------------------------- |
| Dataset          | 40,000 synthetic XOR samples                         |
| Batch Size       | 2,048                                                |
| Epochs           | 2                                                    |
| Architecture     | `Linear(2, 1024) → ReLU → Linear(1024, 1) → Sigmoid` |
| Optimizer        | SGD                                                  |
| Backend          | C++17 + OpenMP                                       |
| Python Interface | PyBind11                                             |

### Example Training Output

```text
==================================================
TinyTorch Stress Test (Massive Data & Wide MLP)
==================================================

1. Preparing Massive XOR dataset...

Number of samples: 40000
Batch size: 2048

2. Creating Wide Model (Heavy Matmul Ops)...

3. Starting High-Performance Training...

--- Epoch 1 ---

Batch 01/20 | Loss: 0.302506
...
```

### Benchmark Results

The following benchmark was obtained from the current stress-test configuration:

| Framework | Optimizer           | Training Time | Final Epoch Loss |
| --------- | ------------------- | ------------: | ---------------: |
| PyTorch   | `torch.optim.SGD`   |   **1.143 s** |         ≈ 0.0689 |
| TinyTorch | `tt.optimizers.SGD` |  **10.463 s** |         ≈ 0.0010 |

These numbers are provided as an engineering comparison for the current implementation and workload. They should not be interpreted as a general framework-level performance comparison, since backend implementations, threading, initialization, hardware utilization, and training configurations can significantly affect results.

The benchmark demonstrates that TinyTorch can execute a complete end-to-end training pipeline and successfully optimize a nonlinear problem to a very low loss.

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
├── tensor/
└── training/
```

The build configuration also enables strict compiler warnings:

```text
-Wall
-Wextra
-Wpedantic
```

Test registration was refactored using helper functions such as:

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

## `ravel_index` Mapping Bug

The first generic transpose implementation used an incorrect argument order when calling `ravel_index()`.

This caused incorrect element placement after transposition.

The index mapping was corrected and validated against higher-dimensional tensor cases.

## 3D Transpose Tests

The expected results of the 3D transpose tests were updated after verifying the correct N-dimensional index mapping.

This ensured that the tests matched the mathematical definition of axis permutation rather than the behavior of the previous implementation.

## Autograd Gradient Accumulation

The Autograd engine supports gradient accumulation during reverse-mode differentiation.

Parameter gradients are explicitly reset between optimizer updates to prevent gradients from previous training iterations from affecting subsequent updates.

---

# 🚀 Future Improvements

TinyTorch is still under active development.

Planned improvements include:

## Tensor Indexing

Add support for chained indexing such as:

```python
tensor[i][j][k]
```

using proxy classes.

## Matrix Multiplication

Further optimize matrix multiplication through:

* Cache blocking / tiling
* Improved SIMD utilization
* Better memory packing
* More efficient thread scheduling

## Training Infrastructure

Future training improvements may include:

* More learning-rate schedulers
* Additional optimizers
* Improved checkpoint serialization
* Mixed-precision support
* More advanced model serialization

---

# 💻 Technologies

| Technology     | Purpose                                |
| -------------- | -------------------------------------- |
| **C++17**      | Core Tensor and Neural Network backend |
| **OpenMP**     | CPU parallelism and SIMD support       |
| **CMake**      | Build system                           |
| **Ninja**      | Fast build execution                   |
| **GoogleTest** | C++ unit testing                       |
| **PyBind11**   | C++/Python interoperability            |
| **Python**     | High-level API and examples            |

---

# 🎯 Project Goals

TinyTorch is intended to be more than a collection of Tensor operations.

The main goal is to understand and implement the fundamental building blocks behind modern neural-network frameworks while maintaining a clean, modular C++ architecture.

The project focuses on:

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
* Optimization algorithms
* Gradient clipping
* Learning-rate scheduling
* Checkpointing
* Dataset abstractions
* Data preprocessing
* Mini-batch data loading
* C++/Python interoperability
* CPU parallelism and SIMD optimization
* Building neural-network infrastructure with an embedded-systems-oriented C++ architecture

---

# 📌 Current Status

TinyTorch currently provides a functional end-to-end training infrastructure:

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
   ├───────────────┐
   ▼               ▼
Gradient       Gradient
Clipping       Accumulation
   │               │
   └───────┬───────┘
           ▼
      Optimizer
  SGD / Adam / AdamW
           │
           ▼
      LR Scheduler
           │
           ▼
        Trainer
           │
           ▼
      Checkpointing
```

The architecture is intentionally modular so that additional layers, operations, optimizers, datasets, schedulers, and hardware-specific backends can be added without redesigning the existing core.
