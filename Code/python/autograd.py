import sys
from pathlib import Path

build_path = Path(__file__).resolve().parents[1] / "build" / "bindings"
sys.path.append(str(build_path))
import tinytorch as tt

def print_test_header(title):
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50)

print_test_header("1. Arithmetic Operations Chain Rule")

x = tt.Tensor([4.0], [1])
y = tt.Tensor([2.0], [1])

x.requires_grad = True
y.requires_grad = True

# Equation: z = (x + y) * (x - y) / x
# Values: x = 4, y = 2
# (4 + 2) * (4 - 2) / 4 = (6 * 2) / 4 = 3.0
z = (x + y) * (x - y) / x

print(f"z (Forward) : {z.tolist()} (Expected: [3.0])")

# Run backward pass
z.backward()

print(f"x.grad      : {x.grad.tolist()} (Expected: [1.25])")
print(f"y.grad      : {y.grad.tolist()} (Expected: [-1.0])")

print_test_header("2. Gradient Accumulation (zero_grad)")
# Call backward again without zeroing
z.backward()
print(f"x.grad (accumulated): {x.grad.tolist()} (Expected: [2.5])")

x.zero_grad()
y.zero_grad()
print(f"x.grad (after zero): {x.grad.tolist()} (Expected: [0.0])")

print_test_header("3. Matrix Multiplication (Linear Layer)")

# Input: 1x2
X = tt.Tensor([1.0, 2.0], [1, 2])
X.requires_grad = True

# Weights: 2x2
W = tt.Tensor([3.0, 4.0, 
               5.0, 6.0], [2, 2])
W.requires_grad = True

# Forward pass (simulating a linear layer without bias)
Z = X.matmul(W)
print(f"Z (Forward) : {Z.tolist()} (Expected: [13.0, 16.0])")
print(f"Z shape     : {Z.shape}")

# Create upstream gradient [1.0, 1.0] manually
grad_output = tt.Tensor([1.0, 1.0], [1, 2])

# Backward pass
Z.backward(grad_output)

print(f"\nW.grad (Expected shape [2, 2], values [[1, 1], [2, 2]]):\n{W.grad}")
print(f"\nX.grad (Expected shape [1, 2], values [[7, 11]]):\n{X.grad}")
