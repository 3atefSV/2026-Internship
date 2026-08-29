# Optimizers (Module 07)

The optimizer module closes the training loop by consuming the gradients produced by the Autograd engine and updating the model's parameters in-place. It provides a PyTorch-like interface exposed to Python via PyBind11.

Implements a small hierarchy of algorithms:

* **SGD** — Stochastic Gradient Descent with optional momentum and L2 weight decay
* **Adam** — Adaptive Moment Estimation with bias correction
* **AdamW** — Adam with decoupled (Adam-style) weight decay

---

## Architecture

All optimizers derive from a single abstract `Optimizer` base class (`optimizer.h`).

### The `Optimizer` Base Class

```cpp
class Optimizer {
  public:
    using ParamRef = Tensor*;
    using ParamList = std::vector<ParamRef>;

    explicit Optimizer(const ParamList& params);

    virtual void step() = 0;
    void zero_grad();

    std::size_t step_count() const noexcept;
};
```

* Parameters are stored as **raw pointers** (`Tensor*`). This lets `step()` write directly into each parameter's data buffer rather than copying it.
* `step()` is a **pure virtual** method that each algorithm implements.
* `step_count_` tracks the number of updates performed and drives the bias-correction exponentials in `Adam` / `AdamW`.

### `zero_grad()`

```cpp
void Optimizer::zero_grad() {
    for (Tensor* param : params_) {
        if (param && param->grad_) {
            std::fill(param->grad_->data().begin(), param->grad_->data().end(), 0.0f);
        }
    }
}
```

Gradients are reset by **zeroing the existing buffer in place** with `std::fill`. The gradient tensor is never deallocated, so no memory is freed and re-allocated between mini-batches (O(1) memory overhead per step).

---

## SGD

Header: `sgd.h` · Source: `sgd.cpp`

| Hyper-parameter | Default | Meaning                             |
| --------------- | ------- | ----------------------------------- |
| `lr`            | `0.01`  | Learning rate                       |
| `momentum`      | `0.0`   | Momentum coefficient (0 = disabled) |
| `weight_decay`  | `0.0`   | L2 weight decay coefficient         |

### Update rules

Traditional **L2 weight decay** (when `weight_decay > 0`) is folded into the gradient:

```
g = g + λ·θ
```

**Momentum** (when `momentum > 0`) maintains a per-parameter velocity buffer, lazily initialized to zero on the first `step()`:

```
v = β·v + g
θ = θ − α·v
```

Without momentum the update is the plain SGD rule:

```
θ = θ − α·g
```

### Velocity buffer

`velocities_` is a vector of `std::shared_ptr<Tensor>`, one per parameter. It is allocated **once**, on the first `step()`, and reused across calls:

```cpp
void SGD::ensure_velocities() {
    if (!velocities_.empty()) return;
    for (Tensor* param : params_) {
        auto velocity = std::make_shared<Tensor>(param->shape());
        velocity->fill(0.0f);
        velocities_.push_back(velocity);
    }
}
```

---

## Adam

Header: `adam.h` · Source: `adam.cpp`

| Hyper-parameter | Default  | Meaning                               |
| --------------- | -------- | ------------------------------------- |
| `lr`            | `0.001`  | Learning rate                         |
| `beta1`         | `0.9`    | Exponential decay for first moment    |
| `beta2`         | `0.999`  | Exponential decay for second moment   |
| `eps`           | `1e-8`   | Numerical stability term              |
| `weight_decay`  | `0.0`    | Coupled L2 weight decay coefficient   |

Adam keeps two per-parameter moment buffers, `m_buffers_` (first moment) and `v_buffers_` (second moment), both lazily initialized to zero.

When `weight_decay > 0`, it is incorporated into the gradient (coupled, traditional L2):

```
g = g + λ·θ
```

### Biased moments

```
m_t = β1·m_{t-1} + (1 − β1)·g_t
v_t = β2·v_{t-1} + (1 − β2)·g_t²
```

### Bias correction

Because both moments start at zero, they are biased low early in training. They are corrected using `step_count_`:

```
m̂_t = m_t / (1 − β1^t)
v̂_t = v_t / (1 − β2^t)
```

### Parameter update

```
θ = θ − α · m̂_t / (√v̂_t + ε)
```

---

## AdamW

Header: `adamw.h` · Source: `adamw.cpp`

| Hyper-parameter | Default  | Meaning                             |
| --------------- | -------- | ----------------------------------- |
| `lr`            | `0.001`  | Learning rate                       |
| `beta1`         | `0.9`    | Exponential decay for first moment  |
| `beta2`         | `0.999`  | Exponential decay for second moment |
| `eps`           | `1e-8`   | Numerical stability term            |
| `weight_decay`  | `0.01`   | **Decoupled** weight decay          |

`AdamW` derives from `Adam`, inheriting its moment buffers and hyper-parameters. It differs in how weight decay is applied:

1. The exact **Adam update runs on the pure gradients** — weight decay is *not* added to `g_t`.
2. Weight decay is then applied **directly to the parameters**:

```
θ = θ · (1 − α·λ)
```

### Coupled vs. Decoupled decay

|            | Adam                                   | AdamW                                        |
| ---------- | -------------------------------------- | -------------------------------------------- |
| Mechanism  | Added to gradient (`g = g + λθ`)       | Applied to parameter (`θ = θ(1 − αλ)`)       |
| Type       | Coupled (depends on gradient scale)    | Decoupled (independent of gradient scale)    |
| Default λ  | `0.0`                                  | `0.01`                                       |

---

## Memory & Performance

All `.step()` kernels follow these rules:

1. Grab the underlying raw buffers **once** via `Tensor::data()` before looping.
2. Iterate flat, using `#pragma omp parallel for` for element-wise parallelism.
3. **Never** allocate inside the update loop — state buffers (`m`, `v`, velocity) are pre-allocated or lazily created on step 1 only.

This keeps the per-step memory overhead at O(1): no heap re-allocations, no temporary tensors.

---

## Python Interface

The module is exposed under `tinytorch.optimizers`:

```python
import tinytorch as tt

model = tt.Sequential()
model.add(tt.Linear(2, 4, bias=True))
model.add(tt.ReLU())
model.add(tt.Linear(4, 1, bias=True))

for param in model.parameters():
    param.requires_grad = True

optimizer = tt.optimizers.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Available classes: `tt.optimizers.SGD`, `tt.optimizers.Adam`, `tt.optimizers.AdamW`. Each exposes `step()`, `zero_grad()`, `step_count`, and its hyper-parameters as read-only properties.
