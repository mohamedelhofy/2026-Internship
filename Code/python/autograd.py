import sys
from pathlib import Path

# Setup path to import tinytorch
build_path = Path(__file__).resolve().parents[1] / "build" / "bindings"
sys.path.append(str(build_path))
import tinytorch as tt

def print_test_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

# ===================================================================
# 1. Arithmetic Operations Chain Rule
# ===================================================================
print_test_header("1. Arithmetic Operations: z = (x + y) * (x - y) / x")

x = tt.Tensor([4.0], [1])
y = tt.Tensor([2.0], [1])
x.requires_grad = True
y.requires_grad = True

z = (x + y) * (x - y) / x
z.backward()

print(f"z value : {z.tolist()} (Expected: [3.0])")
print(f"x.grad  : {x.grad.tolist()} (Expected: [1.25])") 
print(f"y.grad  : {y.grad.tolist()} (Expected: [-1.0])") 


# ===================================================================
# 2. Activation Functions
# ===================================================================
print_test_header("2. Activation Gradients (ReLU & Sigmoid)")

# ReLU Test
x_relu = tt.Tensor([-2.0, 1.0, 3.0], [3])
x_relu.requires_grad = True
y_relu = x_relu.relu()
y_relu.backward(tt.Tensor([1.0, 1.0, 1.0], [3]))
print(f"ReLU Forward    : {y_relu.tolist()} (Expected: [0.0, 1.0, 3.0])")
print(f"ReLU x.grad     : {x_relu.grad.tolist()} (Expected: [0.0, 1.0, 1.0])")

print("-" * 30)

# Sigmoid Test
x_sig = tt.Tensor([0.0], [1])
x_sig.requires_grad = True
y_sig = x_sig.sigmoid()
y_sig.backward()
print(f"Sigmoid Forward : {y_sig.tolist()} (Expected: [0.5])")
print(f"Sigmoid x.grad  : {x_sig.grad.tolist()} (Expected: [0.25])")


# ===================================================================
# 3. Matrix Multiplication
# ===================================================================
print_test_header("3. Matrix Multiplication: Z = X @ W")

X = tt.Tensor([1.0, 2.0], [1, 2])
W = tt.Tensor([3.0, 4.0, 
               5.0, 6.0], [2, 2])
X.requires_grad = True
W.requires_grad = True

Z = X.matmul(W)
grad_output = tt.Tensor([1.0, 1.0], [1, 2])
Z.backward(grad_output)

print(f"Z (Forward) :\n{Z}")
print(f"W.grad (Expected: [[1, 1], [2, 2]]):\n{W.grad}")
print(f"X.grad (Expected: [[7, 11]]):\n{X.grad}")


# ===================================================================
# 4. Losses (MSE)
# ===================================================================
print_test_header("4. Loss Gradients (MSE)")

pred = tt.Tensor([1.0, 2.0], [2])
target = tt.Tensor([1.0, 3.0], [2])
pred.requires_grad = True

mse = tt.MSELoss(tt.Reduction.MEAN)
loss = mse(pred, target)
loss.backward()

print(f"MSE Loss  : {loss.tolist()} (Expected: [0.5])")
print(f"pred.grad : {pred.grad.tolist()} (Expected: [0.0, -1.0])")


# ===================================================================
# 5. Full Neural Network Step (Linear -> ReLU -> MSE)
# ===================================================================
print_test_header("5. Full Mini-Network Forward & Backward")

# Input: 1 sample, 2 features
inputs = tt.Tensor([1.0, 1.0], [1, 2])
# Targets: 1 sample, 2 outputs
targets = tt.Tensor([5.0, 0.0], [1, 2])

# Weights: 2x2
W_net = tt.Tensor([2.0, -1.0, 
                   1.0,  1.0], [2, 2])
W_net.requires_grad = True

# Forward Pass
logits = inputs.matmul(W_net)   # [1*2 + 1*1, 1*-1 + 1*1] = [3.0, 0.0]
activated = logits.relu()       # [3.0, 0.0]
loss_val = mse(activated, targets) # mean( (3-5)^2 + (0-0)^2 ) = mean(4, 0) = 2.0

# Backward Pass
loss_val.backward()

print(f"Logits        : {logits.tolist()}")
print(f"Final Loss    : {loss_val.tolist()} (Expected: [2.0])")
print(f"W_net.grad    :\n{W_net.grad}")
