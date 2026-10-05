import sys
from pathlib import Path

build_path = Path(__file__).resolve().parents[1] / "build" / "bindings"
sys.path.append(str(build_path))

import tinytorch as tt

# ============================================================================
# Demonstrate the Optimizer module (Module 07) using AdamW on a Linear model.
# ============================================================================
print("=" * 50)
print(" TinyTorch Optimizers Demo (Module 07)")
print("=" * 50)

model = tt.Sequential()
model.add(tt.Linear(2, 4, bias=True))
model.add(tt.ReLU())
model.add(tt.Linear(4, 1, bias=True))
model.add(tt.Sigmoid())

params = model.parameters()
for param in params:
    param.requires_grad = True

# Create an AdamW optimizer over the model's parameters.
optimizer = tt.optimizers.AdamW(params, lr=1e-2, weight_decay=0.01)
print(f"\nOptimizer step_count before: {optimizer.step_count}")
print(f"lr={optimizer.lr}, beta1={optimizer.beta1}, beta2={optimizer.beta2}, "
      f"eps={optimizer.eps}, weight_decay={optimizer.weight_decay}")

criterion = tt.MSELoss()

X = tt.Tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2])
Y = tt.Tensor([0.0, 1.0, 1.0, 0.0], [4, 1])

epochs = 3
print("\nTraining (using AdamW)...")
for epoch in range(epochs):
    pred = model(X)
    loss = criterion(pred, Y)

    optimizer.zero_grad()   # clear previous gradients
    loss.backward()          # accumulate gradients
    optimizer.step()         # apply update

    print(f"  Epoch {epoch + 1}: loss = {loss[0]:.6f}")

print(f"\nOptimizer step_count after: {optimizer.step_count}")
print("\nAlso available: tt.optimizers.SGD and tt.optimizers.Adam")
