import sys
from pathlib import Path

# Setup path to import tinytorch
build_path = Path(__file__).resolve().parents[1] / "build" / "bindings"
sys.path.append(str(build_path))

import tinytorch as tt

print("=" * 60)
print("TensorDataset Demo")
print("=" * 60)

# 1. Create dummy data (e.g., 4 samples, each has 3 features)
features = tt.Tensor(
    [
        1.1, 1.2, 1.3,  # Sample 0
        2.1, 2.2, 2.3,  # Sample 1
        3.1, 3.2, 3.3,  # Sample 2
        4.1, 4.2, 4.3   # Sample 3
    ],
    [4, 3]
)

# 2. Create dummy labels (4 samples, 1 label each)
labels = tt.Tensor(
    [0, 1, 0, 1],
    [4]
)

print("Features Shape :", features.shape)
print("Labels Shape   :", labels.shape)

# 3. Initialize TensorDataset
print("\nInitializing TensorDataset...")
dataset = tt.TensorDataset([features, labels])

print("Dataset Size (Number of samples):", len(dataset))

# 4. Test Indexing (__getitem__)
print("\n" + "=" * 60)
print("Testing Indexing (Sample Extraction)")
print("=" * 60)

for i in range(len(dataset)):
    sample = dataset[i]
    print(f"\nSample {i}:")
    print(f"  Feature Tensor: {sample[0].tolist()}")
    print(f"  Feature Shape : {sample[0].shape}")
    print(f"  Label Tensor  : {sample[1].tolist()}")
    print(f"  Label Shape   : {sample[1].shape}")
    