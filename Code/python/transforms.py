import sys
from pathlib import Path

build_path = Path(__file__).resolve().parents[1] / "build" / "bindings"
sys.path.append(str(build_path))

import tinytorch as tt

print("=" * 60)
print("Transforms Demo")
print("=" * 60)

# Create a dummy 1-channel 4x4 image
img = tt.Tensor(
    [
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    ],
    [1, 4, 4]
)

print("Original Image (4x4):")
print(img)

# 1. Test Horizontal Flip (p=1.0 to guarantee flip)
print("\n--- Horizontal Flip ---")
flip = tt.RandomHorizontalFlip(p=1.0)
flipped_img = flip(img)
print(flipped_img)

# 2. Test Crop (crop to 2x2 with 0 padding for clear visualization)
print("\n--- Random Crop (2x2) ---")
crop = tt.RandomCrop((2, 2), padding=0, seed=42)
cropped_img = crop(img)
print(cropped_img)

# 3. Test Compose
print("\n--- Compose (Flip + Crop) ---")
composed = tt.Compose([
    tt.RandomHorizontalFlip(p=1.0),
    tt.RandomCrop((2, 2), padding=0, seed=42)
])
final_img = composed(img)
print(final_img)
