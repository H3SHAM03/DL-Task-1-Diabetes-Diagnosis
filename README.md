# Neural Network Diabetes Diagnosis
## Pima Indians Diabetes Dataset — Binary Classification

---

## Problem Statement

Predict whether a patient has **diabetes** (Outcome = 1) or **not** (Outcome = 0) based on 8 physiological measurements collected from female Pima Indian patients aged 21+.

This is a **binary classification** problem solved using a fully connected neural network built with TensorFlow/Keras.

---

## Dataset

| Property | Value |
|---|---|
| **Source** | Pima Indians Diabetes Database (UCI ML Repository) |
| **File** | `diabetes.csv` |
| **Samples** | 768 patients |
| **Features** | 8 numerical features |
| **Target** | `Outcome` — 0 = No Diabetes, 1 = Diabetes |
| **Class Balance** | 500 No Diabetes (65.1%) / 268 Diabetes (34.9%) |

### Features

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-Hour serum insulin (μU/ml) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Diabetes pedigree function (genetic risk score) |
| Age | Age in years |

---

## Approach

### 1. Preprocessing
- **Zero imputation**: Columns `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` contain biologically impossible zero values that represent missing data. These were replaced with their column **medians**.
- **Feature scaling**: `StandardScaler` was applied (fit on training set only) to normalize all features to zero mean and unit variance.

### 2. Model Architecture

```
Input(8) → Dense(64, ReLU) → Dropout(0.3)
         → Dense(32, ReLU) → Dropout(0.2)
         → Dense(16, ReLU)
         → Dense(1, Sigmoid)
```

| Layer | Output Shape | Parameters |
|---|---|---|
| Dense (64, ReLU) | (None, 64) | 576 |
| Dropout (0.3) | (None, 64) | 0 |
| Dense (32, ReLU) | (None, 32) | 2,080 |
| Dropout (0.2) | (None, 32) | 0 |
| Dense (16, ReLU) | (None, 16) | 528 |
| Dense (1, Sigmoid) | (None, 1) | 17 |
| **Total** | | **3,201** |

### 3. Training Configuration
- **Optimizer**: Adam (lr = 0.001)
- **Loss**: Binary Crossentropy
- **Metric**: Binary Accuracy
- **Max Epochs**: 100
- **Batch Size**: 32
- **Validation Split**: 20%
- **Early Stopping**: patience = 10, monitor = `val_loss`, restore best weights

### 4. Train/Test Split
- **80% training** / **20% testing**, stratified by class label
- Training: 614 samples | Testing: 154 samples

---

## Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **72.08%** |
| **Test Loss** | 0.5333 |
| **Precision** | 0.6122 |
| **Recall** | 0.5556 |
| **F1-Score** | 0.5825 |
| **AUC Score** | **0.8081** |

### Confusion Matrix

|  | Predicted: No Diabetes | Predicted: Diabetes |
|---|---|---|
| **Actual: No Diabetes** | 81 (TN) | 19 (FP) |
| **Actual: Diabetes** | 24 (FN) | 30 (TP) |

> The AUC of **0.808** indicates strong discriminative ability, well above random (0.5).  
> The 72.08% accuracy is within the expected 70–78% range for this dataset.

---

## Repository Structure

```
NN.ipynb                        ← Mini-Task completed
Assignment/
├── pima_diabetes_nn.ipynb      ← Complete notebook (fully runnable)
├── diabetes.csv                ← Dataset (local, no internet required)
├── README.md                   ← This file
├── training_curves.png         ← Accuracy & loss curves
├── confusion_matrix.png        ← Confusion matrix + ROC curve
└── results/
    └── metrics_summary.txt     ← All metrics in plain text
```

---

## How to Run

1. Open `pima_diabetes_nn.ipynb` in VS Code or Jupyter
2. Make sure `diabetes.csv` is in the **same folder** as the notebook
3. Run all cells top-to-bottom (`Run All`)
4. All outputs, plots, and the metrics file will be generated automatically

**Requirements**: `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## Analysis

The Pima Indians Diabetes dataset is a well-known benchmark that is harder than it appears due to three key challenges:

1. **Missing data encoded as zeros** — nearly 49% of `Insulin` values are zero, requiring careful imputation before any model can learn meaningful patterns.
2. **Class imbalance** — with only 35% positive cases, a naive classifier that always predicts "No Diabetes" would achieve ~65% accuracy; our model must learn to identify the minority class.
3. **Limited features** — only 8 features describe complex metabolic interactions, placing a hard ceiling on achievable accuracy.

The chosen architecture with **Dropout regularization** addresses the overfitting risk that comes with 614 training samples. The training curves confirm healthy learning: validation accuracy plateaus around 80% while training accuracy converges toward it, and early stopping fired at epoch 34 (out of max 100), preventing unnecessary overfitting.

The **AUC of 0.808** is the most informative metric here: it measures the model's ability to rank diabetic patients above non-diabetic ones across all thresholds, independent of the classification cutoff. This result is competitive with published baselines on this dataset (~0.80–0.84 AUC range).
