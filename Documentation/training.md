# Training Infrastructure

TinyTorch provides a C++17 training infrastructure that connects the model, loss function, Autograd, DataLoader, and optimizers into one training pipeline.

```text
DataLoader
    ↓
Forward Pass
    ↓
Loss
    ↓
Backward Pass
    ↓
Gradient Clipping
    ↓
Optimizer Step
    ↓
Zero Grad
```

## 1. Trainer

The `Trainer` class handles the main training and evaluation workflow.

```cpp
float train_epoch(
    DataLoader& dataloader,
    int accumulation_steps
);
```

For each batch it performs:

* Forward pass
* Loss computation
* Backward pass
* Gradient accumulation
* Optional gradient clipping
* Optimizer update
* Gradient reset

It also stores training loss, evaluation loss, and learning-rate history.

---

## 2. Gradient Clipping

TinyTorch supports global L2 gradient clipping:

```cpp
float clip_grad_norm(
    const std::vector<Tensor*>& parameters,
    float max_norm
);
```

First, the global gradient norm is calculated across all parameters.

If it exceeds `max_norm`, all gradients are scaled by the same factor:

```cpp
clip_coef = max_norm / (total_norm + 1e-6f);
```

This prevents excessively large gradients from destabilizing training while preserving their relative magnitudes.

---

## 3. Gradient Accumulation

The trainer supports accumulating gradients across multiple batches before updating the parameters.

```text
Batch 1 → Backward
Batch 2 → Backward
Batch 3 → Backward
Batch 4 → Backward
             ↓
        Optimizer Step
             ↓
          Zero Grad
```

This allows the effective training batch to be larger without requiring all samples to be processed at once.

---

## 4. Cosine Learning-Rate Schedule

TinyTorch includes a cosine learning-rate scheduler:

```cpp
CosineSchedule(
    float max_lr,
    float min_lr,
    int total_epochs
);
```

The learning rate smoothly decreases from `max_lr` toward `min_lr` following a cosine curve.

The scheduler currently calculates and records the learning rate. The actual optimizer update through `set_lr()` is prepared but not enabled in the current implementation.

---

## 5. Evaluation

Evaluation runs only the forward pass and loss calculation.

```text
DataLoader
    ↓
Forward
    ↓
Loss
```

No backward pass or optimizer update is performed.

---

## 6. Binary Checkpointing

TinyTorch supports saving and loading training checkpoints directly in C++.

A checkpoint stores:

```text
Epoch
Step
Number of Parameters
Parameter Sizes
Parameter Data
```

The checkpoint is stored in binary format using standard C++ file streams.

When loading, the number and size of parameters are checked to detect model/checkpoint mismatches.

---

## 7. Training Stack

```text
                Trainer
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   DataLoader   Autograd   Optimizer
                              │
                         ┌────┼────┐
                         ▼    ▼    ▼
                        SGD  Adam AdamW
```

Together, these components provide a complete C++ training pipeline from data loading to parameter optimization.
