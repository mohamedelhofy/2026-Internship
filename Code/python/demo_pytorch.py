import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time

print("=" * 50)
print("🔥 PyTorch Stress Test (Massive Data & Wide MLP)")
print("=" * 50)

print("\n1. Preparing Massive XOR dataset...")

X_flat = []
Y_flat = []

for _ in range(10000):
    X_flat.extend([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])  
    Y_flat.extend([0.0, 1.0, 1.0, 0.0])  

X = torch.tensor(X_flat, dtype=torch.float32).view(40000, 2)  
Y = torch.tensor(Y_flat, dtype=torch.float32).view(40000, 1)  

dataset = TensorDataset(X, Y)  
dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)  

print(f"Number of samples: {len(dataset)}")  
print("Batch size: 2048")  

print("\n2. Creating Wide Model (Heavy Matmul Ops)...")  

model = nn.Sequential(
    nn.Linear(2, 1024, bias=True),  
    nn.ReLU(),  
    nn.Linear(1024, 1, bias=True),  
    nn.Sigmoid()  
)

criterion = nn.MSELoss()  
optimizer = optim.SGD(model.parameters(), lr=0.1) 
epochs = 2  

print("\n3. Starting High-Performance Training...")  

start_time = time.perf_counter()  

for epoch in range(epochs):  
    print(f"\n--- Epoch {epoch + 1} ---")  

    for batch_idx, (batch_X, batch_Y) in enumerate(dataloader):  
        
        # 1. Forward Pass
        predictions = model(batch_X)  
        
        # 2. Compute Loss
        loss = criterion(predictions, batch_Y)  
        
        # 3. Clear Gradients 
        optimizer.zero_grad()
        
        # 4. Backward Pass
        loss.backward()  
        
        # 5. Update Weights
        optimizer.step()

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):  
            print(f"  Batch {batch_idx + 1:02d}/{len(dataloader)} | Loss: {loss.item():.6f}")  

end_time = time.perf_counter()  

print("\n✅ Training finished!")
print(f"⏱️ Total Training Time (PyTorch): {end_time - start_time:.3f} seconds")  