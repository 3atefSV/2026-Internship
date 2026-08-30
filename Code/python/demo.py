import sys 
from pathlib import Path 
import time 

build_path = Path(__file__).resolve().parents[1] / "build" / "bindings" 
sys.path.append(str(build_path)) 

import tinytorch as tt 

print("=" * 50) 
print("🚀 TinyTorch Stress Test (Massive Data & Wide MLP)") 
print("=" * 50) 

print("\n1. Preparing Massive XOR dataset...") 

X = [] 
Y = [] 

for _ in range(10000): 
    X.extend([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]) 
    Y.extend([0.0, 1.0, 1.0, 0.0]) 

X = tt.Tensor(X, [40000, 2]) 
Y = tt.Tensor(Y, [40000, 1]) 

dataset = tt.TensorDataset([X, Y]) 
dataloader = tt.DataLoader(dataset, batch_size=2048, shuffle=True) 

print(f"Number of samples: {len(dataset)}") 
print("Batch size: 2048") 

print("\n2. Creating Wide Model (Heavy Matmul Ops)...") 

model = tt.Sequential()
model.add(tt.Linear(2, 1024, bias=True)) 
model.add(tt.ReLU())
model.add(tt.Linear(1024, 1, bias=True)) 
model.add(tt.Sigmoid())

for param in model.parameters(): 
    param.requires_grad = True 

criterion = tt.MSELoss() 
optimizer = tt.optimizers.AdamW(model.parameters(), lr=0.01) 
epochs = 2 

print("\n3. Starting High-Performance Training...") 

start_time = time.perf_counter() 

for epoch in range(epochs): 
    print(f"\n--- Epoch {epoch + 1} ---") 

    for batch_idx, batch in enumerate(dataloader): 
        batch_X = batch[0] 
        batch_Y = batch[1] 

        # 1. Forward Pass
        predictions = model(batch_X) 
        
        # 2. Compute Loss
        loss = criterion(predictions, batch_Y) 
        # 3. Clear Gradients ششششششش
        optimizer.zero_grad()
        
        # 4. Backward Pass
        loss.backward() 
        # if batch_idx == 0 and epoch == 0:
        #     print("Gradient Check:", model.parameters()[0].grad)
        # 5. Update Weights
        optimizer.step()

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader): 
            print(f"  Batch {batch_idx + 1:02d}/{len(dataloader)} | Loss: {loss[0]:.6f}") 

end_time = time.perf_counter() 

print("\n✅ Training finished!")
print(f"⏱️ Total Training Time for 80,000 forward/backward passes: {end_time - start_time:.3f} seconds") 