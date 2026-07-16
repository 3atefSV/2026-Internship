import sys
from pathlib import Path

build_path = Path(__file__).resolve().parents[1] / "build" / "bindings"

sys.path.append(str(build_path))

import tinytorch as tt

print("=" * 60)
print("Layer Demo")
print("=" * 60)

x = tt.Tensor(
    [
        1, 2, 3,
        4, 5, 6
    ],
    [2, 3]
)

print("Input")
print(x)

print("\n================ Linear ================")

linear = tt.Linear(
    in_features=3,
    out_features=4,
    bias=True,
    seed=42
)

print("Input Features :", linear.in_features)
print("Output Features:", linear.out_features)
print("Has Bias       :", linear.has_bias)

y = linear(x)

print("\nOutput")
print(y)

print("\nParameters")

params = linear.parameters()

for i, p in enumerate(params):
    print(f"Parameter {i}")
    print(p)

print("\n================ Dropout ================")

drop = tt.Dropout(
    p=0.5,
    seed=42
)

print("\nTraining")

print(drop(y, True))

print("\nInference")

print(drop(y, False))

print("\n================ Sequential ================")

model = tt.Sequential()

model.add(
    tt.Linear(
        3,
        8,
        seed=1
    )
)

model.add(
    tt.Dropout(
        0.3,
        seed=1
    )
)

model.add(
    tt.Linear(
        8,
        2,
        seed=2
    )
)

print("Number of Layers :", len(model))
print("Is Empty         :", model.empty)

output = model(x)

print("\nFinal Output")
print(output)

print("\nAll Parameters")

params = model.parameters()

print("Total Parameter Tensors :", len(params))

for i, p in enumerate(params):
    print(f"\nParameter {i}")
    print("Shape :", p.shape)