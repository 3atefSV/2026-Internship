import sys
from pathlib import Path

build_path = Path(__file__).resolve().parents[1] / "build" / "bindings"
sys.path.append(str(build_path))
import tinytorch as tt

print("=" * 60)
print("Autograd Core Engine Demo")
print("=" * 60)

# 1. Test requires_grad property
x = tt.Tensor([1.5, 2.5], [2])
print(f"Default requires_grad: {x.requires_grad}")

x.requires_grad = True
print(f"Updated requires_grad: {x.requires_grad}")
print(f"Initial grad values  : {x.grad.tolist()}")

# 2. Test scalar backward (Base case of backprop)
print("\n--- Scalar Backward Test ---")
scalar_loss = tt.Tensor([42.0], [1])
scalar_loss.requires_grad = True

print("Calling backward() on scalar...")
scalar_loss.backward()
print(f"Scalar gradient (should be 1.0): {scalar_loss.grad.tolist()}")

# 3. Test zero_grad
print("\n--- Zero Grad Test ---")
scalar_loss.zero_grad()
print(f"Gradient after zero_grad()     : {scalar_loss.grad.tolist()}")
