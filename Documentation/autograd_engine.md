# Autograd Engine

TinyTorch includes a custom-built, dynamic **Reverse-Mode Automatic Differentiation (Autograd)** engine.

The engine automatically builds a **Directed Acyclic Graph (DAG)** of tensor operations during the forward pass. When `backward()` is called, the graph is traversed in reverse to compute and accumulate gradients for all tensors that require them.

The design keeps the Tensor operations themselves simple while adding gradient tracking through lightweight computation-graph metadata.

---

# 1. How Autograd Works

The core idea behind TinyTorch Autograd is simple:

> Every differentiable operation records how its output was produced.

Consider:

```text
x ──┐
    ├── Add ──> a ──┐
y ──┘               │
                    ├── Multiply ──> z
x ──┐               │
    ├── Subtract ───┘
y ──┘
```

During the forward pass, TinyTorch creates the output tensors normally and attaches a corresponding backward function to each result.

When `backward()` is called, these functions are executed in reverse order to propagate gradients from the output back toward the input tensors.

---

# 2. Gradient Tracking

Autograd is enabled on a tensor using:

```cpp
tensor.set_requires_grad(true);
```

or through the Python interface:

```python
tensor.requires_grad = True
```

Each Tensor stores three important pieces of Autograd state:

```text
requires_grad
grad_
grad_fn_
```

### `requires_grad`

Indicates whether the tensor participates in gradient tracking.

### `grad_`

Stores the accumulated gradient of the tensor.

### `grad_fn_`

Points to the backward operation responsible for producing the tensor.

Together, these fields allow TinyTorch to reconstruct and traverse the computation graph.

---

# 3. Autograd Wiring

The mathematical implementation of an operation remains independent from gradient tracking.

A differentiable operation follows a common pattern:

```cpp
// 1. Perform the standard mathematical operation
Tensor result = /* ... standard computation ... */;

// 2. Attach Autograd metadata
if (lhs.requires_grad() || rhs.requires_grad()) {
    result.set_requires_grad(true);
    result.grad_fn_ = std::make_shared<BackwardOp>(lhs, rhs);
}

return result;
```

This pattern is intentionally small so that gradient tracking does not complicate the core mathematical implementation.

---

## Breaking Down the Pattern

### Step 1 — Perform the Operation

The tensor operation first performs its normal mathematical computation:

```cpp
Tensor result = /* standard tensor operation */;
```

At this point, the result is simply a normal Tensor.

---

### Step 2 — Check Gradient Requirements

```cpp
if (lhs.requires_grad() || rhs.requires_grad())
```

The engine checks whether at least one input participates in gradient computation.

If neither input requires gradients, no Autograd node is created.

This avoids unnecessary graph construction for operations that do not participate in training.

---

### Step 3 — Propagate `requires_grad`

```cpp
result.set_requires_grad(true);
```

If an input requires a gradient, its output must also participate in gradient tracking.

This allows gradient tracking to propagate through a chain of operations.

For example:

```text
x.requires_grad = true

x
 ↓
operation
 ↓
a.requires_grad = true
 ↓
operation
 ↓
b.requires_grad = true
```

This is what allows the computation graph to remain connected throughout the forward pass.

---

### Step 4 — Attach the Backward Function

```cpp
result.grad_fn_ = std::make_shared<BackwardOp>(lhs, rhs);
```

This is the point where the computation graph is constructed.

The operation creates a specialized backward node such as:

```text
AddBackward
SubBackward
MulBackward
DivBackward
MatmulBackward
```

The node stores the information required to compute gradients for the operation's inputs.

---

# 4. The `Function` Architecture

Every differentiable operation derives from the abstract `Function` class defined in:

```text
autograd/function.h
```

The base interface is:

```cpp
class Function {
public:
    virtual ~Function() = default;

    // Receives the upstream gradient and returns
    // gradients corresponding to the operation inputs.
    [[nodiscard]]
    virtual std::vector<Tensor>
    apply(const Tensor& grad_output) = 0;

    // Tensors required during the backward pass.
    std::vector<Tensor> saved_tensors;
};
```

The architecture separates:

* **Graph construction**
* **Gradient computation**
* **Tensor storage**
* **Operation-specific backward logic**

This makes the engine modular and allows new differentiable operations to be added independently.

---

# 5. Upstream and Downstream Gradients

Autograd uses the chain rule to propagate gradients through the graph.

Suppose:

```text
x → f → y → g → z
```

During the backward pass, the engine starts with:

```text
∂z/∂z = 1
```

and propagates the gradient backward:

```text
∂z/∂y
   ↓
∂z/∂x
```

The gradient received by a backward node is called the **upstream gradient**.

The node then computes the gradients required by its inputs and returns them.

Conceptually:

```text
grad_output
     │
     ▼
 Function::apply()
     │
     ├──> grad_input_1
     └──> grad_input_2
```

These gradients are then propagated to the previous nodes in the graph.

---

# 6. `saved_tensors`

Some backward formulas require values from the forward pass.

For example, the derivative of Sigmoid can be written as:

```text
σ'(x) = σ(x)(1 - σ(x))
```

Instead of recomputing the forward operation during backpropagation, TinyTorch can save the required tensor.

The `Function` class therefore contains:

```cpp
std::vector<Tensor> saved_tensors;
```

These tensors are retained by the backward node and used when `apply()` is executed.

This mechanism is used by activation functions and other operations whose backward formulas depend on forward-pass values.

---

# 7. Backward Pass

Calling:

```python
loss.backward()
```

starts the reverse-mode differentiation process.

Conceptually, the process is:

```text
Forward Pass
──────────────────────────────>

x → operation → operation → loss

<──────────────────────────────
          Backward Pass

loss
 ↓
Backward Node
 ↓
Previous Node
 ↓
Previous Node
 ↓
leaf tensors
```

The backward pass:

1. Starts from the output tensor.
2. Initializes its gradient.
3. Retrieves the tensor's `grad_fn_`.
4. Calls the corresponding `Function::apply()`.
5. Receives gradients for the operation inputs.
6. Accumulates those gradients into the corresponding tensors.
7. Continues through their `grad_fn_` nodes.
8. Stops when leaf tensors are reached.

---

# 8. Gradient Accumulation

TinyTorch accumulates gradients rather than replacing them.

For a tensor `x`, multiple paths through the computation graph can contribute to:

```text
∂L/∂x
```

The final gradient is therefore the sum of all contributions:

```text
∂L/∂x =
    contribution_1
  + contribution_2
  + ...
```

This behavior is essential for correctly handling computation graphs where a tensor is used by multiple operations.

Before starting a new optimization step, accumulated gradients can be cleared using:

```python
tensor.zero_grad()
```

---

# 9. Supported Backward Operations

TinyTorch currently provides analytical backward implementations for arithmetic operations, matrix operations, activations, and loss functions.

---

## Arithmetic & Tensor Operations

The engine supports:

* `AddBackward`
* `SubBackward`
* `MulBackward`
* `DivBackward`
* `MatmulBackward`
* `TransposeBackward`
* `ReshapeBackward`

### Matrix Multiplication

`MatmulBackward` computes the gradients of matrix multiplication using explicit implementations.

The implementation avoids unnecessary intermediate transposes where possible and uses hand-written loops for the required gradient calculations.

This keeps the backward implementation aligned with the Tensor core's memory layout and avoids introducing unnecessary overhead.

---

# 10. Activation Backward Operations

Activation functions provide dedicated backward implementations.

Supported activations include:

| Activation | Backward Node     |
| ---------- | ----------------- |
| ReLU       | `ReLUBackward`    |
| Sigmoid    | `SigmoidBackward` |
| Tanh       | `TanhBackward`    |
| Softmax    | `SoftmaxBackward` |
| GELU       | `GELUBackward`    |

These implementations use `saved_tensors` when forward-pass values are required.

For example, if the forward pass produces:

```text
y = sigmoid(x)
```

the backward function can use the saved output `y` to compute:

```text
dy/dx = y(1-y)
```

without recomputing the Sigmoid operation.

---

# 11. Loss Backward Operations

Loss functions act as the starting point of the training backward pass.

They convert the loss gradient into gradients with respect to the model predictions.

TinyTorch provides dedicated backward implementations for:

### `MSEBackward`

Computes the gradient of Mean Squared Error.

For:

```text
L = mean((prediction - target)^2)
```

the gradient is:

```text
∂L/∂prediction
=
2(prediction - target) / N
```

for mean reduction.

---

### `CrossEntropyBackward`

Computes the gradient of Cross Entropy Loss using the stable combination of Softmax and Negative Log-Likelihood.

For class-index targets, the core gradient is based on:

```text
softmax(logits) - one_hot(target)
```

with the appropriate scaling applied according to the selected reduction mode.

This avoids constructing unnecessary intermediate operations such as:

```text
softmax → log → loss
```

and keeps the backward computation numerically stable.

---

### `BCEBackward`

Computes Binary Cross Entropy gradients.

The implementation takes the same numerical-safety considerations as the forward pass into account, including epsilon-clamped probabilities to avoid division by zero.

---

# 12. Reduction-Aware Gradients

Loss functions support:

```text
Reduction::Mean
Reduction::Sum
Reduction::None
```

The backward implementation must account for the selected reduction.

For example, with:

```text
Reduction::Mean
```

the gradient is divided by the number of contributing elements.

With:

```text
Reduction::Sum
```

the individual gradients are accumulated without the mean scaling factor.

This ensures that the backward pass is mathematically consistent with the forward loss calculation.

---

# 13. Computation Graph Example

Consider:

```python
z = (x + y) * (x - y) / x
```

The forward pass creates a graph similar to:

```text
        x ─────┐
               ├── Add ──> a ─────┐
        y ─────┘                  │
                                  ├── Mul ──> b ──┐
        x ─────┐                  │              │
               ├── Sub ──> c ─────┘              │
        y ─────┘                                 │
                                                 ├── Div ──> z
        x ───────────────────────────────────────┘
```

Each intermediate result stores its corresponding backward function:

```text
a.grad_fn_ → AddBackward
c.grad_fn_ → SubBackward
b.grad_fn_ → MulBackward
z.grad_fn_ → DivBackward
```

When:

```python
z.backward()
```

is called, TinyTorch traverses these operations in reverse.

```text
DivBackward
     ↓
MulBackward
     ↓
AddBackward / SubBackward
     ↓
x, y
```

The gradients from all paths leading to the same tensor are accumulated.

---

# 14. Python API

Because the C++ backend is exposed through **PyBind11**, the Python API provides a familiar PyTorch-like interface.

```python
import tinytorch as tt

# Create tensors
x = tt.Tensor([4.0], [1])
x.requires_grad = True

y = tt.Tensor([2.0], [1])
y.requires_grad = True

# Forward pass
z = (x + y) * (x - y) / x

# Backward pass
z.backward()

# Access gradients
print(x.grad)
print(y.grad)
```

For:

```text
z = (x + y)(x - y) / x
```

the expression simplifies to:

```text
z = x - y² / x
```

Therefore:

```text
∂z/∂x = 1 + y²/x²
∂z/∂y = -2y/x
```

For:

```text
x = 4
y = 2
```

the expected gradients are:

```text
∂z/∂x = 1.25
∂z/∂y = -1.0
```

so:

```text
x.grad = [1.25]
y.grad = [-1.0]
```

---

# 15. Design Principles

The TinyTorch Autograd engine is built around several principles.

### Minimal Intrusion

Tensor operations perform their normal mathematical work first, then attach lightweight Autograd metadata.

This keeps the Tensor implementation readable and modular.

### Dynamic Graph Construction

The computation graph is constructed dynamically during the forward pass.

This allows arbitrary Python control flow and dynamically constructed models without requiring a static graph definition.

### Operation-Specific Backward Functions

Every differentiable operation owns its gradient logic through a specialized `Function` implementation.

This makes the system easy to extend.

Adding a new differentiable operation generally requires:

1. Implementing the forward Tensor operation.
2. Creating its corresponding `Function` / backward node.
3. Saving any tensors required for backpropagation.
4. Implementing `apply()` to compute input gradients.
5. Attaching the backward node to the output tensor.

### Gradient Accumulation

Gradients are accumulated across all paths in the computation graph, allowing shared tensors and branching graphs to be handled correctly.

---

# 16. Overall Architecture

The relationship between the main components can be summarized as:

```text
                 Forward Pass
                      │
                      ▼
              ┌───────────────┐
              │     Tensor    │
              └───────┬───────┘
                      │
              requires_grad?
                      │
                      ▼
              ┌───────────────┐
              │   grad_fn_    │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │   Function    │
              │  Backward Op  │
              └───────┬───────┘
                      │
                saved_tensors
                      │
                      ▼
              ┌───────────────┐
              │    backward() │
              └───────┬───────┘
                      │
                Reverse Graph
                  Traversal
                      │
                      ▼
              ┌───────────────┐
              │ Gradient Acc. │
              │    grad_      │
              └───────────────┘
```

---

# Summary

TinyTorch's Autograd engine provides a dynamic implementation of **reverse-mode automatic differentiation**.

Its core architecture consists of:

* Dynamic computation-graph construction.
* Tensor-level `requires_grad` tracking.
* `grad_fn_` links between operations.
* Specialized backward `Function` nodes.
* `saved_tensors` for forward-pass values required during backpropagation.
* Reverse graph traversal through `backward()`.
* Gradient accumulation in `.grad_`.
* Analytical gradients for arithmetic, matrix, activation, and loss operations.
* PyBind11 integration for a PyTorch-like Python API.

The result is a compact but complete automatic differentiation system that exposes the mechanisms behind modern deep learning frameworks while remaining closely integrated with the TinyTorch Tensor core.
