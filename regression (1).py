"""
Regression Task: California Housing Price Prediction
Dataset: sklearn's California Housing dataset
Model: Multi-layer Feedforward Neural Network (MLP)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. REPRODUCIBILITY
# ─────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 2. DATASET & PREPROCESSING
# ─────────────────────────────────────────────
print("\n--- Loading California Housing Dataset ---")
data = fetch_california_housing()
X, y = data.data, data.target  # y = median house value (in $100k)

print(f"Features      : {data.feature_names}")
print(f"Total samples : {X.shape[0]}")
print(f"Feature shape : {X.shape}")
print(f"Target range  : {y.min():.2f} – {y.max():.2f} ($100k)")

# Train / Validation / Test split  (70 / 15 / 15)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Feature scaling (StandardScaler fit only on train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Convert to PyTorch tensors
def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32)

X_train_t = to_tensor(X_train)
X_val_t   = to_tensor(X_val)
X_test_t  = to_tensor(X_test)
y_train_t = to_tensor(y_train).unsqueeze(1)
y_val_t   = to_tensor(y_val).unsqueeze(1)
y_test_t  = to_tensor(y_test).unsqueeze(1)

# DataLoaders
BATCH_SIZE = 64
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=BATCH_SIZE)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=BATCH_SIZE)

# ─────────────────────────────────────────────
# 3. MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class RegressionMLP(nn.Module):
    """
    Multi-Layer Perceptron for regression.
    Architecture: Input(8) → 128 → 64 → 32 → Output(1)
    Each hidden layer uses BatchNorm + ReLU + Dropout.
    """
    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1)   # single continuous output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


model = RegressionMLP(input_dim=X_train.shape[1]).to(DEVICE)
print("\n--- Model Architecture ---")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# ─────────────────────────────────────────────
# 4. TRAINING SETUP
# ─────────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

EPOCHS = 100
history = {"train_loss": [], "val_loss": [], "val_mae": []}

# ─────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────
print("\n--- Training ---")
best_val_loss = float("inf")
best_model_state = None

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)

    # --- Validate ---
    model.eval()
    val_loss, val_mae = 0.0, 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(X_batch)
            val_loss += criterion(pred, y_batch).item() * X_batch.size(0)
            val_mae  += torch.abs(pred - y_batch).sum().item()
    val_loss /= len(val_loader.dataset)
    val_mae  /= len(val_loader.dataset)

    scheduler.step(val_loss)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_mae"].append(val_mae)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch:3d}/{EPOCHS}]  "
              f"Train MSE: {train_loss:.4f}  "
              f"Val MSE: {val_loss:.4f}  "
              f"Val MAE: {val_mae:.4f}")

# ─────────────────────────────────────────────
# 6. EVALUATION ON TEST SET
# ─────────────────────────────────────────────
print("\n--- Test Evaluation ---")
model.load_state_dict(best_model_state)
model.eval()

all_preds, all_targets = [], []
test_loss = 0.0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        pred = model(X_batch)
        test_loss += criterion(pred, y_batch).item() * X_batch.size(0)
        all_preds.append(pred.cpu())
        all_targets.append(y_batch.cpu())

test_loss /= len(test_loader.dataset)
all_preds   = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

test_mae  = np.mean(np.abs(all_preds - all_targets))
test_rmse = np.sqrt(test_loss)
ss_res    = np.sum((all_targets - all_preds) ** 2)
ss_tot    = np.sum((all_targets - np.mean(all_targets)) ** 2)
r2        = 1 - ss_res / ss_tot

print(f"Test MSE  : {test_loss:.4f}")
print(f"Test RMSE : {test_rmse:.4f}  (×$100k = ${test_rmse*100_000:,.0f})")
print(f"Test MAE  : {test_mae:.4f}   (×$100k = ${test_mae*100_000:,.0f})")
print(f"R² Score  : {r2:.4f}")

# ─────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Regression — California Housing Price Prediction", fontsize=14, fontweight="bold")

# Loss curves
axes[0].plot(history["train_loss"], label="Train MSE")
axes[0].plot(history["val_loss"],   label="Val MSE")
axes[0].set_title("Training & Validation Loss (MSE)")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE")
axes[0].legend(); axes[0].grid(True)

# Predicted vs Actual
axes[1].scatter(all_targets, all_preds, alpha=0.3, s=10, color="steelblue")
mn, mx = all_targets.min(), all_targets.max()
axes[1].plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect fit")
axes[1].set_title(f"Predicted vs Actual  (R²={r2:.3f})")
axes[1].set_xlabel("Actual ($100k)"); axes[1].set_ylabel("Predicted ($100k)")
axes[1].legend(); axes[1].grid(True)

# Residual distribution
residuals = all_preds.flatten() - all_targets.flatten()
axes[2].hist(residuals, bins=50, color="coral", edgecolor="white")
axes[2].axvline(0, color="black", lw=2, linestyle="--")
axes[2].set_title("Residual Distribution")
axes[2].set_xlabel("Residual (Predicted − Actual)")
axes[2].set_ylabel("Count"); axes[2].grid(True)

plt.tight_layout()
plt.savefig("regression/regression_results.png", dpi=150)
print("\nPlot saved → regression/regression_results.png")

# Save model
torch.save(best_model_state, "regression/regression_model.pth")
print("Model saved → regression/regression_model.pth")
