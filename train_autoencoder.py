import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn

print("=== Minimal Autoencoder Training ===")


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 6), nn.ReLU(),
            nn.Linear(6, 3), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 6), nn.ReLU(),
            nn.Linear(6, 10)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Build model
model = Autoencoder()
print("Model architecture:")
print(model)

# Simple training
x = torch.randn(50, 10)
target = x.clone()  # Autoencoder target is input itself

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("\nTraining...")
for epoch in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

print(f"Final loss: {loss.item():.6f}")

# Save the trained model
torch.save(model.state_dict(), "minimal_autoencoder.pth")
print("✓ Saved: minimal_autoencoder.pth")

# Test reconstruction
model.eval()
with torch.no_grad():
    test_input = torch.randn(1, 10)
    reconstructed = model(test_input)
    error = torch.mean((test_input - reconstructed) ** 2)
    print(f"Test reconstruction error: {error.item():.6f}")

print("✅ Done! You can convert to CoreML later if needed.")