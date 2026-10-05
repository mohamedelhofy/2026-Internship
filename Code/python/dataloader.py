import sys
from pathlib import Path

build_path = Path(__file__).resolve().parents[1] / "build" / "bindings"
sys.path.append(str(build_path))

import tinytorch as tt

print("=" * 60)
print("DataLoader & ImageDataset Demo")
print("=" * 60)

# 1. ImageDataset with Mock Data and Transforms
print("Initializing ImageDataset with RandomCrop...")
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"]
labels = [tt.Tensor([0], [1]), tt.Tensor([1], [1]), tt.Tensor([0], [1]), tt.Tensor([1], [1]), tt.Tensor([2], [1])]

transform = tt.RandomCrop((28, 28), padding=4, seed=42)
img_dataset = tt.ImageDataset(image_paths, labels, transform)

print(f"Dataset contains {len(img_dataset)} samples")

# 2. DataLoader (Batch Size = 2, Shuffle = True)
print("\nInitializing DataLoader (batch_size=2, shuffle=True)...")
loader = tt.DataLoader(img_dataset, batch_size=2, shuffle=True, seed=99)

print(f"Total Batches per epoch: {len(loader)}")

# 3. Simulate Training Loop (Lazy Evaluation)
print("\n--- Starting Training Epoch ---")
for batch_idx, batch in enumerate(loader):
    features, targets = batch[0], batch[1]
    
    print(f"\nBatch {batch_idx}:")
    print(f"  Features Shape: {features.shape}  <-- Note the Batch Dimension!")
    print(f"  Targets Shape : {targets.shape}")
    print(f"  Targets Values: {targets.tolist()}")

print("\nTraining Epoch Finished!")
