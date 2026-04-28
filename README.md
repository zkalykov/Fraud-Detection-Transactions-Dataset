# Fraud Detection on PaySim Synthetic Financial Dataset

A machine learning project that detects fraudulent financial transactions using Random Forest and Decision Tree classifiers trained on the [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) synthetic dataset (6.3M+ transactions).

**Author:** Zhyrgalbek Kalykov  
**Course:** COMP 2319 — North American University  
**Professor:** Sabina Adhikari

---

## 📄 Report

[PAYSIM_DATASET_TRAINING_REPORT.pdf](https://github.com/zkalykov/Fraud-Detection-Transactions-Dataset/blob/main/PAYSIM_DATASET_TRAINING_REPORT.pdf)

---

## Problem

Fraudulent transactions make up less than 0.14% of the dataset. Standard classifiers trained on this raw distribution learn to predict "not fraud" for everything and achieve ~99.8% accuracy while catching zero actual fraud. The challenge is building a model that reliably detects the rare fraud cases without being overwhelmed by the class imbalance.

## Dataset

**Source:** [PaySim1 on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)

| Property | Value |
|----------|-------|
| Total Transactions | 6,362,620 |
| Fraudulent Transactions | 8,213 (0.13%) |
| Transaction Types | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| Fraud-Prone Types | TRANSFER, CASH_OUT only |

## Methodology

The preprocessing pipeline was designed to avoid common pitfalls like data leakage and misleading evaluation metrics:

1. **Data Cleaning** — Dropped `nameOrig`, `nameDest`, and `isFlaggedFraud`. One-Hot Encoded the `type` column with `drop_first=True` to avoid multicollinearity.

2. **Stratified Train/Test Split** — Split the full dataset 80/20 using `stratify=y` to preserve the real-world 0.13% fraud rate in the test set.

3. **Feature Scaling** — Fit `StandardScaler` on the training set only, then transformed both sets to prevent data leakage. (Note: tree-based models are scale-invariant, so this was applied for pipeline consistency and future model comparisons.)

4. **Random Undersampling (Training Only)** — Downsampled normal transactions in the training set to match the fraud count, creating a balanced 50/50 training subset. The test set was left untouched at its original imbalanced distribution.

5. **Model Training** — Two models were trained on the same balanced training data:
   - **Random Forest Classifier** — 100 decision trees (`n_estimators=100`, `random_state=42`)
   - **Decision Tree Classifier** — Single tree (`max_depth=10`, `random_state=42`)

## Results

Both models were evaluated on the **realistic imbalanced test set** (~1.27M transactions).

### Model Comparison

| Metric | Random Forest | Decision Tree |
|--------|--------------|--------------|
| Accuracy | 98.73% | 98.53% |
| Fraud Recall | 99.63% (1,637 / 1,643) | 99.15% (1,629 / 1,643) |
| Fraud Precision | 9.19% | 8.02% |
| F1 (Fraud) | 0.168 | 0.148 |
| ROC-AUC | 0.9995 | 0.9922 |
| Missed Fraud Cases | 6 | 14 |
| False Positives | 16,181 | 18,686 |

**Winner: Random Forest** — higher recall, precision, F1, and ROC-AUC across every metric. The ensemble approach caught 8 additional fraud cases and produced ~2,500 fewer false positives than the single Decision Tree.

### Confusion Matrices

**Random Forest:**

|  | Predicted: Not Fraud | Predicted: Fraud |
|--|---------------------|-----------------|
| **Actual: Not Fraud** | 1,254,700 | 16,181 |
| **Actual: Fraud** | 6 | 1,637 |

**Decision Tree:**

|  | Predicted: Not Fraud | Predicted: Fraud |
|--|---------------------|-----------------|
| **Actual: Not Fraud** | 1,252,195 | 18,686 |
| **Actual: Fraud** | 14 | 1,629 |

The Random Forest's 99.6% recall means only 6 fraud cases slipped through. The false positives represent the expected precision–recall tradeoff — in production, flagged transactions would go to a human review queue rather than being auto-blocked.

### Top Features (Random Forest)

The model relied most on account balance features:

1. `oldbalanceOrg` (sender's starting balance)
2. `amount` (transaction amount)
3. `newbalanceOrig` (sender's ending balance)
4. `type_TRANSFER`
5. `type_CASH_OUT`

## How to Run

### Requirements

```
Python 3.8+
pandas
scikit-learn
matplotlib
seaborn
kagglehub
```

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/zkalykov/Fraud-Detection-Transactions-Dataset.git
   cd Fraud-Detection-Transactions-Dataset
   ```

2. Install dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn kagglehub
   ```

3. Open and run the notebook:
   ```bash
   jupyter notebook Synthetic_Financial_Datasets_For_Fraud_Detection.ipynb
   ```

   The notebook will automatically download the dataset from Kaggle via `kagglehub`.

## Disclaimer

This project was completed as part of the COMP 2319 coursework at North American University. The PaySim dataset is a synthetic simulation and does not contain real financial records.
