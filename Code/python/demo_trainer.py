import sys
from pathlib import Path

build_path = Path(__file__).resolve().parents[1] / "build" / "bindings"
sys.path.append(str(build_path))

import tinytorch as tt

print("=" * 60)
print("🚀 TinyTorch Trainer Architecture Demo")
print("=" * 60)

# 1. Prepare Data
X = tt.Tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2])
Y = tt.Tensor([0.0, 1.0, 1.0, 0.0], [4, 1])
dataset = tt.TensorDataset([X, Y])
dataloader = tt.DataLoader(dataset, batch_size=2, shuffle=True)

# 2. Build Model
model = tt.Sequential()
model.add(tt.Linear(2, 8, bias=True))
model.add(tt.ReLU())
model.add(tt.Linear(8, 1, bias=True))
model.add(tt.Sigmoid())

for param in model.parameters():
    param.requires_grad = True

# 3. Setup Optimizer, Loss, and Scheduler
optimizer = tt.optimizers.AdamW(model.parameters(), lr=0.1)
loss_fn = tt.MSELoss()
scheduler = tt.CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=20)

# 4. Initialize The Trainer!
trainer = tt.Trainer(
    model=model, 
    optimizer=optimizer, 
    loss_fn=loss_fn, 
    scheduler=scheduler, 
    grad_clip_norm=1.0
)

print("\n⚙️ Starting Training Pipeline...")
for epoch in range(20):
    # One line replaces the whole internal loop!
    train_loss = trainer.train_epoch(dataloader)
    
    if (epoch + 1) % 5 == 0:
        eval_loss, _ = trainer.evaluate(dataloader)
        print(f"Epoch {epoch+1:02d}/20 | Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f}")

# 5. Test Checkpointing (C++ Binary)
ckpt_path = "xor_checkpoint.bin"
trainer.save_checkpoint(ckpt_path)
print(f"\n💾 Model successfully saved to {ckpt_path}")

print("✅ End-to-End Orchestration complete!")
