import sys
from pathlib import Path

build_path = Path(__file__).resolve().parents[1] / "build" / "bindings"

sys.path.append(str(build_path))

import tinytorch

print("=" * 50)
print("TinyTorch Python Demo")
print("=" * 50)

a = tinytorch.Tensor(
    [1, 2, 3, 4],
    [2, 2]
)

b = tinytorch.Tensor(
    [5, 6, 7, 8],
    [2, 2]
)

print("\nTensor A")
print(a)

print("\nTensor B")
print(b)

print("\nProperties")
print("shape :", a.shape)
print("size  :", a.size)
print("ndim  :", a.ndim)
print("dtype :", a.dtype)
print("empty :", a.empty)
print("len   :", len(a))
print("bool  :", bool(a))

print("\nEquality")
print(a == a)
print(a == b)

print("\nTensor + Tensor")
print(a + b)

print("\nTensor - Tensor")
print(a - b)

print("\nTensor * Tensor")
print(a * b)

print("\nTensor / Tensor")
print(a / b)

print("\nTensor + Scalar")
print(a + 10)

print("\nScalar + Tensor")
print(10 + a)

print("\nTensor - Scalar")
print(a - 1)

print("\nScalar - Tensor")
print(10 - a)

print("\nTensor * Scalar")
print(a * 2)

print("\nScalar * Tensor")
print(2 * a)

print("\nTensor / Scalar")
print(a / 2)

print("\nScalar / Tensor")
print(20 / a)

print("\nMatrix Multiplication")
print(a.matmul(b))

print("\nUsing @ Operator")
print(a @ b)

print("\nIndexing")
print(a[0, 0])
print(a[1, 1])

a[0, 1] = 100

print("\nAfter Modification")
print(a)

print("\nCopy")
c = a.copy()
print(c)

print("\nTo List")
print(c.tolist())

print("\n3D Tensor")

t = tinytorch.Tensor(
    list(range(24)),
    [2, 3, 4]
)

print(t)

print("\nAccess element")
print(t[1, 2, 3])

t[1, 2, 3] = 999

print("\nModified value")
print(t[1, 2, 3])
