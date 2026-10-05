import sys
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

build_path = Path(__file__).resolve().parents[1] / "build" / "bindings"
sys.path.append(str(build_path))

import tinytorch as tt

print("=" * 60)
print("TinyTorch: End-to-End MNIST Training Pipeline")
print("=" * 60)

# 1. Prepare Data
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Normalize pixels to [0, 1]
X = (mnist["data"] / 255.0).astype(np.float32)
y_raw = mnist["target"].astype(int)

# One-hot encoding for MSELoss compatibility (10 classes for digits 0-9)
num_classes = 10
y_onehot = np.zeros((y_raw.shape[0], num_classes), dtype=np.float32)
y_onehot[np.arange(y_raw.shape[0]), y_raw] = 1.0

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=10000, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")

# Note: Converting numpy arrays to flat lists to strictly match tt.Tensor(data, shape)
# If your PyBind11 bindings support direct numpy array inputs, you can pass X_train directly.
X_train_list = X_train.flatten().tolist()
y_train_list = y_train.flatten().tolist()

X_train_tensor = tt.Tensor(X_train_list, list(X_train.shape))
y_train_tensor = tt.Tensor(y_train_list, list(y_train.shape))

dataset = tt.TensorDataset([X_train_tensor, y_train_tensor])
dataloader = tt.DataLoader(dataset, batch_size=64, shuffle=True)

# 2. Build Model
print("Building the Network Architecture...")
model = tt.Sequential()
model.add(tt.Linear(784, 256, bias=True))
model.add(tt.ReLU())
model.add(tt.Linear(256, 128, bias=True))
model.add(tt.ReLU())
model.add(tt.Linear(128, 10, bias=True))
model.add(tt.Sigmoid())

for param in model.parameters():
    param.requires_grad = True

# 3. Setup Optimizer, Loss, and Scheduler
optimizer = tt.optimizers.AdamW(model.parameters(), lr=0.001)
loss_fn = tt.MSELoss()
epochs = 5
scheduler = tt.CosineSchedule(max_lr=0.001, min_lr=0.0001, total_epochs=epochs)

# 4. Initialize The Trainer
trainer = tt.Trainer(
    model=model, 
    optimizer=optimizer, 
    loss_fn=loss_fn, 
    scheduler=scheduler, 
    grad_clip_norm=1.0
)

# 5. Training Loop
print("\nStarting Training Pipeline...")
for epoch in range(epochs):
    train_loss = trainer.train_epoch(dataloader)
    print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f}")

# 6. Save Checkpoint
ckpt_path = "mnist_checkpoint.bin"
trainer.save_checkpoint(ckpt_path)
print(f"\nModel successfully saved to {ckpt_path}")
print("Training Complete!")