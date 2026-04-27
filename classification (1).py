"""
Classification Task: MNIST Handwritten Digit Recognition
Dataset: torchvision MNIST (70 000 grayscale 28×28 images, 10 classes)
Model: Convolutional Neural Network (CNN)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

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
print("\n--- Loading MNIST Dataset ---")

transform = transforms.Compose([
    transforms.ToTensor(),                          # [0,255] → [0,1]
    transforms.Normalize((0.1307,), (0.3081,))      # MNIST mean & std
])

train_full = torchvision.datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_set   = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Split training → 90% train / 10% validation
val_size   = int(0.10 * len(train_full))
train_size = len(train_full) - val_size
train_set, val_set = random_split(train_full, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(42))

print(f"Train samples      : {len(train_set):,}")
print(f"Validation samples : {len(val_set):,}")
print(f"Test samples       : {len(test_set):,}")
print(f"Classes            : {train_full.classes}")

BATCH_SIZE = 128
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ─────────────────────────────────────────────
# 3. MODEL ARCHITECTURE  (CNN)
# ─────────────────────────────────────────────
class MNISTClassifier(nn.Module):
    """
    CNN for 10-class digit classification.
    Input : (B, 1, 28, 28)
    Block1 : Conv(1→32, 3×3) → BN → ReLU → Conv(32→32, 3×3) → BN → ReLU → MaxPool → Dropout
    Block2 : Conv(32→64, 3×3) → BN → ReLU → Conv(64→64, 3×3) → BN → ReLU → MaxPool → Dropout
    Classifier : Flatten → FC(1024→128) → ReLU → Dropout → FC(128→10)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),         # 28×28 → 14×14
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),         # 14×14 → 7×7
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),               # 64 × 7 × 7 = 3136 → but we use adaptive pool
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


model = MNISTClassifier().to(DEVICE)
print("\n--- Model Architecture ---")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# ─────────────────────────────────────────────
# 4. TRAINING SETUP
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

EPOCHS = 15
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

# ─────────────────────────────────────────────
# 5. HELPER: ONE EPOCH PASS
# ─────────────────────────────────────────────
def run_epoch(loader, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total

# ─────────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────────
print("\n--- Training ---")
best_val_acc   = 0.0
best_model_state = None

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss,   val_acc   = run_epoch(val_loader,   train=False)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc   = val_acc
        best_model_state = model.state_dict()

    print(f"Epoch [{epoch:2d}/{EPOCHS}]  "
          f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc*100:.2f}%  "
          f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc*100:.2f}%")

# ─────────────────────────────────────────────
# 7. TEST EVALUATION
# ─────────────────────────────────────────────
print("\n--- Test Evaluation ---")
model.load_state_dict(best_model_state)
model.eval()

all_preds, all_labels = [], []
test_loss, correct, total = 0.0, 0, 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        test_loss += criterion(logits, labels).item() * imgs.size(0)
        preds      = logits.argmax(1)
        correct   += (preds == labels).sum().item()
        total     += imgs.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

test_loss /= total
test_acc   = correct / total
all_preds  = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds,
                             target_names=[str(i) for i in range(10)]))

# ─────────────────────────────────────────────
# 8. PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Classification — MNIST Digit Recognition", fontsize=14, fontweight="bold")

epochs_range = range(1, EPOCHS + 1)

# Loss curves
axes[0].plot(epochs_range, history["train_loss"], label="Train")
axes[0].plot(epochs_range, history["val_loss"],   label="Val")
axes[0].set_title("Loss over Epochs")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].legend(); axes[0].grid(True)

# Accuracy curves
axes[1].plot(epochs_range, [a*100 for a in history["train_acc"]], label="Train")
axes[1].plot(epochs_range, [a*100 for a in history["val_acc"]],   label="Val")
axes[1].set_title("Accuracy over Epochs")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
axes[1].legend(); axes[1].grid(True)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(10), yticklabels=range(10), ax=axes[2])
axes[2].set_title(f"Confusion Matrix  (Acc={test_acc*100:.2f}%)")
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True")

plt.tight_layout()
plt.savefig("classification/classification_results.png", dpi=150)
print("\nPlot saved → classification/classification_results.png")

# Save model
torch.save(best_model_state, "classification/classification_model.pth")
print("Model saved → classification/classification_model.pth")
