import torch
import torch.nn as nn

# Simple test: model on GPU 1, computation on GPU 0, gradients should flow back
model = nn.Linear(10, 5).to('cuda:1')  # Model on GPU 1
x = torch.randn(3, 10, requires_grad=True).to('cuda:1')  # Input on GPU 1

# Forward pass on GPU 1
y = model(x)  # Output on GPU 1

# Move to GPU 0 for "simulation" (just a simple operation)
y_gpu0 = y.to('cuda:0')  # Move to GPU 0
loss = (y_gpu0 ** 2).sum()  # Simple loss computation on GPU 0

# Backward pass - gradients should flow back to GPU 1
loss.backward()

# Test: Check if model parameters have gradients
print(f"Model parameters have gradients: {model.weight.grad is not None}")
print(f"Input has gradients: {x.grad is not None}")
print(f"Loss value: {loss.item():.4f}")
print("✅ Cross-GPU gradient flow works!" if model.weight.grad is not None else "❌ Gradient flow broken!")