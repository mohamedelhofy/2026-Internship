import sys
from pathlib import Path

build_path = Path(__file__).resolve().parents[1] / "build" / "bindings"

sys.path.append(str(build_path))

import tinytorch as tt

print("=" * 60)
print("Loss Demo")
print("=" * 60)

print("\n================ MSELoss ================")

predictions = tt.Tensor([1, 2, 3], [3])
targets = tt.Tensor([2, 4, 6], [3])

criterion = tt.MSELoss()

loss = criterion(predictions, targets)

print("Predictions :", predictions)
print("Targets     :", targets)
print("Loss        :", loss)

print("\nSum reduction  :", tt.MSELoss(tt.Reduction.SUM)(predictions, targets))
print("No reduction   :", tt.MSELoss(tt.Reduction.NONE)(predictions, targets))

print("\n================ CrossEntropyLoss ================")

logits = tt.Tensor(
    [
        1, 2, 3,
        1, 2, 3
    ],
    [2, 3]
)

criterion = tt.CrossEntropyLoss()

# Class indices may be given as a plain Python list...
print("Logits        :", logits)
print("Loss (list)   :", criterion(logits, [2, 0]))

# ...or as a Tensor of indices, like the rest of the framework.
print("Loss (tensor) :", criterion(logits, tt.Tensor([2, 0], [2])))

print("Per sample    :", tt.CrossEntropyLoss(tt.Reduction.NONE)(logits, [2, 0]))

print("\nSoftmax is applied internally, so raw logits go straight in:")
print("log_softmax   :", logits.log_softmax(-1))

print("\nHuge logits stay finite (Log-Sum-Exp trick):")
big = tt.Tensor([1000, 1000, 1000], [1, 3])
print("Loss          :", criterion(big, [0]))

print("\n================ BinaryCrossEntropyLoss ================")

probabilities = tt.Tensor([0.9, 0.1, 0.8, 0.2], [4])
labels = tt.Tensor([1, 0, 1, 0], [4])

criterion = tt.BinaryCrossEntropyLoss()

print("Probabilities :", probabilities)
print("Labels        :", labels)
print("Epsilon       :", criterion.epsilon)
print("Loss          :", criterion(probabilities, labels))

print("\nA hopeless prediction is clipped instead of becoming infinite:")
print("Loss          :", criterion(tt.Tensor([0.0], [1]), tt.Tensor([1.0], [1])))

print("\n================ Training-style Usage ================")

model = tt.Sequential(
    tt.Linear(3, 8, seed=1),
    tt.ReLU(),
    tt.Linear(8, 2, seed=2),
)

batch = tt.Tensor([1, 2, 3, 4, 5, 6], [2, 3])

logits = model(batch)
criterion = tt.CrossEntropyLoss()

print("Model logits :", logits)
print("Batch loss   :", criterion(logits, [0, 1]))