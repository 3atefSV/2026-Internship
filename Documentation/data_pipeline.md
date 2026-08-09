# Data Pipeline

The **Data Pipeline** module provides a PyTorch-like interface for loading, preprocessing, and batching data for training and evaluation.

It separates **data access**, **preprocessing**, and **batch generation** from the neural network training loop, making the overall training architecture cleaner and easier to extend.

The pipeline is built around three main components:

```text
Dataset
   │
   ▼
Transforms
   │
   ▼
DataLoader
   │
   ▼
Mini-batches
   │
   ▼
Training Loop
```

---

# 1. Datasets

The `Dataset` abstraction is the foundation of the data pipeline.

It provides a unified interface for accessing individual samples regardless of where the data comes from.

---

## `Dataset`

The abstract `Dataset` interface is defined in:

```text
data/dataset.h
```

Any custom dataset must inherit from `Dataset` and implement the core data-access operations.

### `size()`

Returns the total number of samples in the dataset.

```cpp
size_t size() const;
```

### `get(index)`

Returns the sample stored at the specified index.

Conceptually:

```cpp
auto sample = dataset.get(index);
```

A sample typically consists of:

```text
(data, label)
```

where both components are represented using TinyTorch Tensor objects.

### Custom Dataset

The abstraction allows users to create datasets for different data sources while keeping the DataLoader independent of the underlying storage mechanism.

Conceptually:

```text
              Dataset
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
 TensorDataset       ImageDataset
       │                   │
   In-memory             Disk
      data               images
```

---

# 2. TensorDataset

`TensorDataset` is a concrete dataset implementation designed for data that is already loaded into memory.

It is defined in:

```text
data/tensor_dataset.h
```

### Construction

It takes two tensors:

```text
features
labels
```

For example:

```text
features: {1000, 10}
labels:   {1000, 1}
```

The first dimension represents the number of samples.

Therefore:

```text
number of samples = features.shape()[0]
                  = labels.shape()[0]
```

### Shape Validation

`TensorDataset` validates that both tensors contain the same number of samples.

For example:

```text
features: {1000, 10}
labels:   {1000, 1}
```

is valid because both contain `1000` samples.

However:

```text
features: {1000, 10}
labels:   {900, 1}
```

is rejected because the dataset cannot produce a valid `(feature, label)` pair for every sample.

This validation prevents mismatched datasets from reaching the training loop.

---

# 3. ImageDataset

`ImageDataset` is designed specifically for computer vision workloads.

It is defined in:

```text
data/image_dataset.h
```

Unlike `TensorDataset`, which expects data to already be loaded into memory, `ImageDataset` provides infrastructure for working with image files stored on disk.

### Main Purpose

It is intended to:

* Load image files dynamically.
* Decode images when samples are requested.
* Associate images with their corresponding labels.
* Avoid loading the entire image dataset into memory at once.

This makes the dataset abstraction suitable for larger computer vision datasets where keeping every image in RAM would be inefficient.

The resulting interface remains compatible with the same `Dataset` abstraction used by `TensorDataset`.

---

# 4. Transforms

Preprocessing and data augmentation are handled through the **Transforms** module.

The implementation is defined in:

```text
data/transforms.h
```

A transform is a callable operation that receives a Tensor and returns a transformed Tensor.

Conceptually:

```text
raw sample
    │
    ▼
Transform 1
    │
    ▼
Transform 2
    │
    ▼
Transform 3
    │
    ▼
processed sample
```

This separates preprocessing logic from the dataset itself.

---

## Normalization

Transforms can be used to normalize input data.

For example, image pixel values can be converted from:

```text
[0, 255]
```

to:

```text
[0, 1]
```

or normalized to a zero-mean/unit-variance representation.

Normalization allows neural network layers to operate on inputs with more suitable numerical ranges.

---

## Type Casting

Transforms can also convert the representation of the input data before it is passed to the model.

This allows preprocessing to be separated from the model implementation.

---

## Composition

Multiple transforms can be applied sequentially.

For example:

```text
Image
  │
  ▼
Resize
  │
  ▼
Normalize
  │
  ▼
Type Conversion
  │
  ▼
Tensor
```

This allows a preprocessing pipeline to be defined independently from the training loop.

---

# 5. DataLoader

The `DataLoader` is the component responsible for connecting a `Dataset` to the training loop.

It is defined in:

```text
data/dataloader.h
```

Its main responsibility is to retrieve individual samples from a dataset and combine them into **mini-batches**.

The overall relationship is:

```text
Dataset
   │
   │ individual samples
   ▼
DataLoader
   │
   │ mini-batches
   ▼
Training Loop
```

---

# 6. Batching

The DataLoader automatically combines individual samples into batched tensors.

For example, suppose the dataset contains images with:

```text
{C, H, W}
```

and the batch size is:

```text
32
```

The DataLoader produces:

```text
{32, C, H, W}
```

where the first dimension represents the batch size.

Conceptually:

```text
Sample 1 ─┐
Sample 2 ─┤
Sample 3 ─┤
    ...    ├──► DataLoader ──► Batch Tensor
Sample 32 ─┘
```

This allows the neural network to process multiple samples in a single forward pass.

---

# 7. Shuffling

The DataLoader supports random shuffling of dataset samples.

```python
dataloader = tt.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

When shuffling is enabled, the order in which samples are retrieved is randomized.

This is particularly important during training because consistently presenting samples in the same order can introduce unwanted sequence-based biases.

The DataLoader maintains an internal random number generator to perform the shuffling.

---

# 8. Iteration

The DataLoader is designed to integrate naturally with both C++ and Python.

### C++ Usage

It supports integration with C++ range-based iteration.

Conceptually:

```cpp
for (auto& batch : dataloader) {
    // Process batch
}
```

### Python Usage

Through PyBind11, the DataLoader exposes a Python iterator interface:

```python
for batch_features, batch_labels in dataloader:
    # Process batch
```

This provides a familiar Python workflow while the underlying data pipeline remains implemented in C++.

---

# 9. Complete Data Flow

The complete pipeline can be represented as:

```text
                  Dataset
                     │
              Retrieve sample
                     │
                     ▼
                 Transform
                     │
              Preprocess sample
                     │
                     ▼
                 DataLoader
                     │
              Collate samples
                     │
                     ▼
                 Mini-batch
                     │
                     ▼
              Neural Network
                     │
                     ▼
                    Loss
                     │
                     ▼
                 Backward
```

This separation of responsibilities makes each component independently reusable.

---

# 10. Python API Example

Because the data pipeline is exposed through PyBind11, it can be used from Python with a PyTorch-like interface.

```python
import tinytorch as tt

# 1. Prepare raw tensors
features = tt.Tensor(...)  # Shape: [1000, 10]
labels = tt.Tensor(...)    # Shape: [1000, 1]

# 2. Wrap tensors in a Dataset
dataset = tt.TensorDataset(features, labels)

# 3. Create a DataLoader
dataloader = tt.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# 4. Iterate through mini-batches
for batch_features, batch_labels in dataloader:

    # Forward pass
    predictions = model(batch_features)

    # Compute loss
    loss = criterion(predictions, batch_labels)

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()

    # Clear accumulated gradients
    model.zero_grad()
```

---

# 11. Training Pipeline Architecture

The Data Pipeline integrates directly with the other TinyTorch modules.

A typical training architecture looks like:

```text
┌────────────────────┐
│      Dataset       │
│                    │
│  Raw Data / Images │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│     Transforms     │
│                    │
│ Preprocessing /    │
│ Normalization      │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│     DataLoader     │
│                    │
│ Batching /         │
│ Shuffling          │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│   Neural Network   │
│                    │
│ Tensor + Layers    │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│       Loss         │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│      Autograd      │
│                    │
│ Backpropagation    │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│     Optimizer      │
│                    │
│ Parameter Update   │
└────────────────────┘
```

This architecture keeps data loading independent from model computation while allowing all components to work together as a complete training system.

---

# Summary

The TinyTorch Data Pipeline provides the infrastructure required to move data from raw storage into a neural network efficiently.

Its main components are:

* **`Dataset`** — Abstract interface for accessing individual samples.
* **`TensorDataset`** — In-memory dataset backed by Tensor objects.
* **`ImageDataset`** — Dataset infrastructure for dynamically loading image files.
* **Transforms** — Preprocessing and data transformation pipeline.
* **`DataLoader`** — Mini-batch generation, shuffling, and iteration.
* **PyBind11 Integration** — Python-friendly iteration and usage.

Together, these components create a clean data flow:

```text
Dataset
   ↓
Transforms
   ↓
DataLoader
   ↓
Mini-batches
   ↓
Model
   ↓
Loss
   ↓
Autograd
   ↓
Optimizer
```

This design keeps data management separate from model logic while providing a familiar, PyTorch-like workflow for training and evaluating TinyTorch models.
