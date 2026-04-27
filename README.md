# PyTorch Implementation: Regression & Classification Tasks
## 👤 Author

Name: Vaibhav Vaid 
Roll No: 2301010289 
Course: B.Tech Cse Core 
Section: E


Assignment submission-  PyTorch deep learning project covering one regression task and one classification task, with full training, evaluation, and result discussion.

## 📁 Repository Structure

```
pytorch_project/
│
├── regression/
│   ├── regression.py            # Full regression pipeline
│   └── regression_results.png  # Loss curve, Predicted vs Actual, Residuals
│
├── classification/
│   ├── classification.py        # Full classification pipeline
│   └── classification_results.png  # Loss/Acc curves, Confusion Matrix
│
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🔧 Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### Step 1 — Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd pytorch_project
```

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Regression Task
```bash
cd regression
python regression.py
```

### Classification Task
```bash
cd classification
python classification.py
```

> MNIST data (~11 MB) is auto-downloaded on first run into a local `./data/` folder.

---
## 📊 Task 1 — Regression: California Housing Price Prediction

### Dataset Description
| Property        | Detail                                      |
|-----------------|---------------------------------------------|
| Source          | `sklearn.datasets.fetch_california_housing` |
| Samples         | 20,640                                      |
| Features        | 8 numerical (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude) |
| Target          | Median house value in $100,000 units        |
| Target range    | $0.15k – $5.00k ($100k units)               |

### Preprocessing Steps
1. **Train/Val/Test split** — 70% / 15% / 15% using `train_test_split` with `random_state=42`
2. **StandardScaler** — fit on train set only; applied to val and test to prevent data leakage
3. **Tensor conversion** — NumPy arrays → `torch.float32` tensors
4. **DataLoader** — batch size 64, shuffled for training

### Model Architecture

```
RegressionMLP
─────────────────────────────
Input          →  8
Linear(8→128)  + BatchNorm + ReLU + Dropout(0.2)
Linear(128→64) + BatchNorm + ReLU + Dropout(0.2)
Linear(64→32)  + BatchNorm + ReLU
Linear(32→1)   →  continuous output
─────────────────────────────
Trainable params: ~12,000
```

### Training Details
| Hyperparameter | Value                              |
|----------------|------------------------------------|
| Loss function  | MSELoss                            |
| Optimizer      | Adam (lr=1e-3, weight_decay=1e-4)  |
| Scheduler      | ReduceLROnPlateau (patience=5)     |
| Epochs         | 100                                |
| Batch size     | 64                                 |

### Evaluation Metrics & Results

| Metric      | Formula                                      | Result (approx.) |
|-------------|----------------------------------------------|------------------|
| **MSE**     | mean((ŷ − y)²)                               | ~0.28            |
| **RMSE**    | √MSE                                         | ~$52,900         |
| **MAE**     | mean(\|ŷ − y\|)                              | ~$36,000         |
| **R² Score**| 1 − SS_res/SS_tot                            | ~0.80            |

**Discussion:**  
The MLP achieves an R² of ~0.80, meaning ~80% of variance in house prices is explained by the model. The RMSE of ~$53K is competitive for a simple MLP without feature engineering. Residuals are approximately normally distributed around zero, indicating no systematic bias. Performance could be further improved with feature engineering (log-transform skewed features) or ensemble methods.

## 🔢 Task 2 — Classification: MNIST Digit Recognition

### Dataset Description
| Property      | Detail                                     |
|---------------|--------------------------------------------|
| Source        | `torchvision.datasets.MNIST`               |
| Train samples | 60,000 (split to 54,000 train / 6,000 val) |
| Test samples  | 10,000                                     |
| Image size    | 28 × 28 grayscale                          |
| Classes       | 10 (digits 0–9)                            |

### Preprocessing Steps
1. **ToTensor** — converts PIL images to `[0, 1]` float tensors
2. **Normalize** — subtract MNIST mean (0.1307), divide by std (0.3081)
3. **Train/Val split** — 90% / 10% random split from official training set
4. **DataLoader** — batch size 128, shuffled for training

### Model Architecture

```
MNISTClassifier (CNN)
──────────────────────────────────────────────
Input: (B, 1, 28, 28)

Conv Block 1:
  Conv2d(1→32, 3×3, pad=1) → BN → ReLU
  Conv2d(32→32, 3×3, pad=1) → BN → ReLU
  MaxPool2d(2×2)  →  14×14
  Dropout2d(0.25)

Conv Block 2:
  Conv2d(32→64, 3×3, pad=1) → BN → ReLU
  Conv2d(64→64, 3×3, pad=1) → BN → ReLU
  MaxPool2d(2×2)  →  7×7
  Dropout2d(0.25)

Classifier:
  Flatten  →  3136
  Linear(3136→128) → ReLU → Dropout(0.5)
  Linear(128→10)  →  logits

──────────────────────────────────────────────
Trainable params: ~840,000
```

### Training Details
| Hyperparameter | Value                         |
|----------------|-------------------------------|
| Loss function  | CrossEntropyLoss              |
| Optimizer      | Adam (lr=1e-3)                |
| Scheduler      | StepLR (step=5, gamma=0.5)    |
| Epochs         | 15                            |
| Batch size     | 128                           |

### Evaluation Metrics & Results

| Metric       | Result (approx.) |
|--------------|------------------|
| **Test Accuracy**   | ~99.2%    |
| **Test Loss**       | ~0.025    |
| **Precision** (avg) | ~99.2%    |
| **Recall** (avg)    | ~99.2%    |
| **F1-Score** (avg)  | ~99.2%    |

**Discussion:**  
The CNN achieves ~99.2% test accuracy after just 15 epochs — near state-of-the-art for a simple architecture. Batch Normalization accelerates convergence, while Dropout prevents overfitting. The confusion matrix shows the model rarely confuses digits; the most common misclassifications are between visually similar pairs (e.g., 4/9, 3/8). Further gains could come from data augmentation (random rotation, affine) or residual connections.


## 📈 Output Files

Each script generates:
 PNG plot with training curves and evaluation visualisations saved to the respective folder
`.pth` file — saved best model weights (lowest validation loss / highest val accuracy)

## 🛠️ Tech Stack

| Library        | Version   | Purpose                      |
|----------------|-----------|------------------------------|
| PyTorch        | ≥ 2.0     | Model building & training    |
| torchvision    | ≥ 0.15    | MNIST dataset & transforms   |
| scikit-learn   | ≥ 1.3     | California Housing, metrics  |
| NumPy          | ≥ 1.24    | Numerical operations         |
| Matplotlib     | ≥ 3.7     | Plotting                     |
| Seaborn        | ≥ 0.12    | Confusion matrix heatmap     |


## 👤 Author

Name: Vaibhav Vaid 
Roll No: 2301010289 
Course: B.Tech Cse Core 
Section: E


## 📄 License

This project is submitted for academic purposes.

