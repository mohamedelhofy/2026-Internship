import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

build_path = Path(__file__).resolve().parents[1] / "build" / "bindings"
sys.path.append(str(build_path))

import tinytorch as tt

print("=" * 60)
print("TinyTorch: MNIST Inference & Visualization")
print("=" * 60)

# 1. Load Data
print("Loading MNIST dataset for testing...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = (mnist["data"] / 255.0).astype(np.float32)
y = mnist["target"].astype(int)

# 2. Rebuild Model
print("Rebuilding Model architecture...")
model = tt.Sequential()
model.add(tt.Linear(784, 256, bias=True))
model.add(tt.ReLU())
model.add(tt.Linear(256, 128, bias=True))
model.add(tt.ReLU())
model.add(tt.Linear(128, 10, bias=True))
model.add(tt.Sigmoid())

# 3. Load Checkpoint
# Initializing a dummy Trainer just to access the load_checkpoint method
# (If your load_checkpoint is implemented directly in the model, you can call model.load_weights instead)
dummy_optimizer = tt.optimizers.AdamW(model.parameters(), lr=0.001)
dummy_loss = tt.MSELoss()
dummy_scheduler = tt.CosineSchedule(0.1, 0.1, 1)

trainer = tt.Trainer(
    model=model, 
    optimizer=dummy_optimizer, 
    loss_fn=dummy_loss, 
    scheduler=dummy_scheduler, 
    grad_clip_norm=1.0
)

ckpt_path = "mnist_checkpoint.bin"
trainer.load_checkpoint(ckpt_path)
print(f"Weights successfully loaded from {ckpt_path}")

# 4. Inference on Random Samples
num_samples = 5
random_indices = np.random.choice(X.shape[0], num_samples, replace=False)

fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

for i, idx in enumerate(random_indices):
    image = X[idx]
    true_label = y[idx]
    
    # Prepare input tensor (batch_size=1)
    input_data = image.flatten().tolist()
    input_tensor = tt.Tensor(input_data, [1, 784])
    
    # Forward pass
    prediction_tensor = model.forward(input_tensor)
    
    # Extract results
    # Use .numpy() if implemented, otherwise fallback to .data() assuming it returns a list
    if hasattr(prediction_tensor, 'numpy'):
        pred_np = prediction_tensor.numpy()
    else:
        pred_np = np.array(prediction_tensor.data).reshape(1, 10)
        
    predicted_label = np.argmax(pred_np, axis=1)[0]
    
    # Visualization
    ax = axes[i]
    ax.imshow(image.reshape(28, 28), cmap='gray')
    ax.axis('off')
    
    color = 'green' if predicted_label == true_label else 'red'
    ax.set_title(f"Pred: {predicted_label} | True: {true_label}", color=color, fontsize=12, fontweight='bold')

plt.tight_layout()
print("Displaying results window...")
plt.show()