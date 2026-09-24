# Credit Card Fraud Detection

A data mining course project detecting fraudulent credit card transactions, using a real,
famous benchmark dataset from European cardholders (2013). Unlike a cost-ratio-based project,
this one evaluates models with standard metrics appropriate for **extreme class imbalance**
(0.17% fraud) — primarily **Precision-Recall AUC (Average Precision)** rather than accuracy or
plain ROC-AUC, which are both misleading at this level of imbalance.



**Note on the dataset file:** `creditcard.csv` (~98MB) is deliberately **not** committed to
this repository — it sits right at GitHub's 100MB hard file-size limit, which makes direct
commits slow and fragile. Run `python3 data/download_data.py` once to fetch it locally before
running anything else (the notebook and `pipeline.py` both expect it at `data/creditcard.csv`).

## Problem

Detect fraudulent credit card transactions among genuine ones. Fraud is extremely rare
(0.17% of transactions), so the central challenge isn't just building a classifier — it's
picking the right way to **measure** whether that classifier is actually useful.

## Dataset

**Credit Card Fraud Detection** (ULB / Kaggle): 284,807 real transactions from European
cardholders over two days in September 2013. 30 features (Time, Amount, and 28 PCA-anonymized
components V1–V28), no missing values. Only 492 transactions (0.17%) are fraudulent. See
`data/data_dictionary.md` for full documentation.

## Methods

- **Preprocessing:** dropped raw `Time` in favor of an engineered `Hour` feature (0–23);
  scaled all numeric features (V1–V28 arrive already PCA-transformed, but `Amount` and `Hour`
  are on very different scales and need it).
- **Models compared (2, as scoped):** Logistic Regression and Random Forest.
- **Imbalance handling:** SMOTE oversampling (to 10% minority ratio, not full 50/50 — chosen
  for realism and training speed) applied only inside the training pipeline.
- **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, and — the primary metric —
  **PR-AUC (Average Precision)**, with 3-fold cross-validation for stability. No cost-ratio
  assumption is used anywhere in this project.

## Key Result

| Model | Accuracy | Precision | Recall | PR-AUC |
|---|---|---|---|---|
| **Random Forest** | 99.9% | **75.4%** | 82.1% | **0.827** |
| Logistic Regression | 97.3% | 5.5% | 89.4% | 0.711 |

Both models look excellent on accuracy (97–99.9%) — which is exactly why accuracy is the wrong
metric here. Logistic Regression catches slightly more fraud (89.4% vs. 82.1% recall) but is
right only 1 time in 18 when it flags a transaction (5.5% precision) — nearly 1,900 false
alarms on the test set. Random Forest catches almost as much fraud with 33 false alarms
instead of 1,878. Full discussion in `reports/final_report.md` and the notebook.


## Tools Used

Python 3, scikit-learn, imbalanced-learn (SMOTE), pandas, matplotlib/seaborn, Jupyter Notebook.


