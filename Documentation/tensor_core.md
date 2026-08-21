# Tensor Core & Operations

The **Tensor Core** is the backbone of TinyTorch. It provides a fully dynamic, PyTorch-like **N-dimensional tensor implementation** defined in `tensor.h`.

It handles memory management, mathematical operations, shape manipulation, broadcasting, matrix multiplication, and provides the foundational data structure required by the **Autograd engine**.

---

## Tensor Properties & Initialization

Tensors are built on top of standard C++ vectors, providing dynamic sizing and contiguous memory storage.

### Memory & Storage

* Data is managed using `std::vector<value_type>`.
* `value_type` is `float` by default.
* Tensor elements are stored in a contiguous memory layout where applicable.

### Initialization

The Tensor class supports:

* Empty initialization.
* Shape-based initialization with zero-initialized elements.
* Direct data and shape injection.

### Utility Methods

| Method        | Description                                             |
| ------------- | ------------------------------------------------------- |
| `shape()`     | Returns the tensor dimensions as `std::vector<size_t>`. |
| `size()`      | Returns the total number of elements.                   |
| `ndim()`      | Returns the number of dimensions.                       |
| `empty()`     | Checks whether the tensor contains no elements.         |
| `dtype()`     | Returns the underlying data type (`float`).             |
| `fill(value)` | Fills the entire tensor with a specified scalar value.  |

---

## Shape Operations

TinyTorch supports complex tensor shape transformations natively.

### `reShape()`

```cpp
Tensor reShape(const Shape& new_shape);
```

Returns a new tensor with the requested shape while preserving the original underlying element order.

### `Transpose()`

```cpp
Tensor Transpose(const Shape& axis);
```

Performs a true **N-dimensional transpose** by remapping tensor dimensions and elements according to the provided axis permutation.

Unlike a 2D-only transpose, this implementation supports arbitrary tensor ranks.

---

## Reduction Operations

TinyTorch provides standard mathematical reduction operations both globally and along specific dimensions.

### Summation

```cpp
tensor.sum();
tensor.sum(axis);
```

* `sum()` reduces the entire tensor to a scalar.
* `sum(axis)` reduces the tensor along the specified dimension.

### Mean

```cpp
tensor.mean();
tensor.mean(axis);
```

* `mean()` computes the global mean.
* `mean(axis)` computes the mean along a specific dimension.

### Extrema

```cpp
tensor.max();
tensor.min();
```

Returns the maximum or minimum value of the tensor.

---

## Arithmetic Operations

The Tensor core supports element-wise and scalar arithmetic operations.

All standard arithmetic operations return new Tensor instances without modifying the original operands.

### Tensor-to-Tensor Operations

```cpp
tensor_a + tensor_b;
tensor_a - tensor_b;
tensor_a * tensor_b;
tensor_a / tensor_b;
```

### Tensor-to-Scalar Operations

```cpp
tensor + scalar;
tensor - scalar;
tensor * scalar;
tensor / scalar;
```

Scalar operations are also supported in the reverse direction where applicable:

```cpp
scalar + tensor;
scalar - tensor;
scalar * tensor;
scalar / tensor;
```

### In-Place Operations

For memory-efficient updates:

```cpp
tensor += other;
tensor -= other;
tensor *= other;
tensor /= other;
```

---

## Broadcasting

TinyTorch implements **NumPy-style broadcasting** for element-wise operations.

### Capabilities

The broadcasting system supports:

* Automatic broadcast shape calculation.
* Operations between tensors with different numbers of dimensions.
* Broadcasting singleton dimensions with size `1`.
* Strict shape validation to detect incompatible operations.

### Internal Helpers

The broadcasting implementation relies on several helper functions:

```text
broadcast_shape()
broadcast_index()
compute_size()
unravel_index()
ravel_index()
```

These functions handle shape compatibility, index mapping, and conversion between flat and multidimensional indices.

---

## Matrix Multiplication

TinyTorch provides a `matmul()` implementation supporting both standard and batched matrix multiplication.

### Supported Operations

* 2D matrix multiplication.
* N-dimensional batched matrix multiplication.
* Automatic broadcasting of batch dimensions.
* Dynamic generation of the resulting tensor shape.

### Helper Functions

```text
extract_batch_shape()
```

The implementation separates batch dimensions from the final two matrix dimensions and performs matrix multiplication across the resulting batches.

---

## Element Access & Operators

### N-Dimensional Access

Tensor elements can be accessed or modified using multidimensional indices:

```cpp
tensor({i, j, k, ...});
```

For example:

```cpp
Tensor tensor({2, 3, 4});

float value = tensor({1, 2, 3});
tensor({1, 2, 3}) = 10.0f;
```

### Flat Access

Standard `operator[]` provides direct flat-memory access:

```cpp
tensor[index];
```

### Comparison

The Tensor class provides:

```cpp
operator==
```

for deep equality checks between tensors.

### Copy & Move Semantics

The Tensor implementation supports standard C++:

* Copy constructor.
* Move constructor.
* Copy assignment.
* Move assignment.

### Stream Output

The `std::ostream` operator (`<<`) is implemented for convenient tensor printing and debugging:

```cpp
std::cout << tensor << std::endl;
```

---

## Autograd Integration

The Tensor object directly stores the metadata required by TinyTorch's **reverse-mode automatic differentiation engine**.

### Gradient Tracking

```cpp
tensor.set_requires_grad(true);
tensor.requires_grad();
```

These methods enable or query gradient tracking for a tensor.

### Computation Graph

Each tensor can store the function node responsible for generating it:

```cpp
grad_fn_
```

This pointer connects the tensor to the corresponding node in the computation graph.

### Gradient Storage

Accumulated gradients are stored through:

```cpp
grad_
```

### Backpropagation

The Tensor class provides:

```cpp
tensor.backward(gradient);
```

which triggers the reverse-mode backward pass starting from the tensor.

### Clearing Gradients

```cpp
tensor.zero_grad();
```

Clears accumulated gradients before the next training iteration.

---

## Neural Network & Loss Primitives

Although activations and losses belong logically to higher-level modules, the Tensor core provides the fundamental mathematical operations required to implement them efficiently.

### Activation Functions

The Tensor API provides:

```cpp
tensor.relu();
tensor.sigmoid();
tensor.tanh();
tensor.gelu();
tensor.softmax(dim);
```

These operations form the mathematical building blocks for neural network layers.

### Loss & Numerical Operations

The Tensor core also provides:

```cpp
tensor.log();
tensor.clamp(min, max);
tensor.log_softmax(dim);
```

These operations are particularly important for implementing numerically stable loss functions such as **Cross Entropy Loss**.

---

## Summary

The Tensor core provides the fundamental numerical infrastructure for TinyTorch:

* Dynamic N-dimensional tensor representation.
* Shape manipulation and N-dimensional transpose.
* NumPy-style broadcasting.
* Element-wise and scalar arithmetic.
* Global and axis-based reductions.
* 2D and batched matrix multiplication.
* Multidimensional and flat element access.
* Reverse-mode Autograd integration.
* Activation and loss-related mathematical primitives.

Together, these features form the foundation on which the higher-level **Neural Network**, **Loss**, **Data Pipeline**, and **Autograd** components of TinyTorch are built.
