
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              average_precision_score, confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

RESULTS_DIR = "../results"
FIG_DIR = f"{RESULTS_DIR}/figures"
RANDOM_STATE = 42

# --- Ledger / case-file color theme (shared with the presentation) ---
PAPER = "#FAF9F5"
INK = "#1E1B18"
INK_SOFT = "#4A453E"
RULE = "#C9C2B4"
OXBLOOD = "#8B2E28"
FOREST = "#3F5738"
BRASS = "#9C7F3B"

plt.rcParams.update({
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.edgecolor": RULE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "grid.color": RULE,
    "grid.linewidth": 0.6,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "font.family": "serif",
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
})
sns.set_theme(style="white", rc=plt.rcParams)

# ----------------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------------
df = pd.read_csv("../data/creditcard.csv")
print("Shape:", df.shape)
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True))
print("Missing values:", df.isnull().sum().sum())

# ----------------------------------------------------------------------------
# 2. EDA
# ----------------------------------------------------------------------------
plt.figure(figsize=(5, 4))
ax = sns.countplot(x="Class", data=df, palette=[FOREST, OXBLOOD])
plt.xticks([0, 1], ["Genuine (0)", "Fraud (1)"])
plt.title("Class Distribution — Genuine vs. Fraud")
plt.yscale("log")
plt.ylabel("count (log scale)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/class_distribution.png", dpi=120)
plt.close()

# Transaction amount distribution by class
plt.figure(figsize=(7, 5))
sns.boxplot(x="Class", y="Amount", data=df[df["Amount"] < 500], palette=[FOREST, OXBLOOD])
plt.xticks([0, 1], ["Genuine", "Fraud"])
plt.title("Transaction Amount by Class (amounts under $500)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/amount_by_class.png", dpi=120)
plt.close()

# Fraud rate by hour of day
df["Hour"] = ((df["Time"] // 3600) % 24).astype(int)
hourly = df.groupby("Hour")["Class"].agg(["mean", "count"]).reset_index()
plt.figure(figsize=(8, 4.5))
sns.barplot(x="Hour", y="mean", data=hourly, color=OXBLOOD)
plt.ylabel("Fraud rate")
plt.title("Fraud Rate by Hour of Day")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fraud_rate_by_hour.png", dpi=120)
plt.close()

print("EDA figures saved.")

# ----------------------------------------------------------------------------
# 3. PREPROCESSING
# ----------------------------------------------------------------------------
# V1-V28 are already PCA components (roughly standardized). Time and Amount are on very
# different scales and need scaling. We scale all numeric features together for consistency
# inside the pipeline (fit only on training data).
data = df.drop(columns=["Time"]).copy()  # raw Time isn't meaningful on its own; Hour is used instead

X = data.drop(columns=["Class"])
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Fraud cases in train: {y_train.sum()}, in test: {y_test.sum()}")

scaler = StandardScaler()

# ----------------------------------------------------------------------------
# 4. MODELS — only 2, as requested
# ----------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced",
                                                random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=8,
                                             random_state=RANDOM_STATE, n_jobs=-1),
}

results = []
roc_data = {}
pr_data = {}
fitted_pipes = {}

for name, clf in models.items():
    pipe = ImbPipeline(steps=[
        ("scale", scaler),
        ("smote", SMOTE(sampling_strategy=0.1, random_state=RANDOM_STATE)),
        ("clf", clf),
    ])
    pipe.fit(X_train, y_train)
    fitted_pipes[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)  # PR-AUC

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision", n_jobs=1)

    results.append({
        "model": name, "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "roc_auc": auc, "pr_auc": ap,
        "cv_pr_auc_mean": cv_scores.mean(), "cv_pr_auc_std": cv_scores.std(),
    })

    roc_data[name] = roc_curve(y_test, y_proba) + (auc,)
    pr_data[name] = precision_recall_curve(y_test, y_proba) + (ap,)

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["Genuine", "Fraud"], digits=4))
    print(f"PR-AUC (Average Precision): {ap:.4f}")

results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False).reset_index(drop=True)
results_df.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)
print("\n\nFINAL MODEL COMPARISON (sorted by PR-AUC, higher = better):")
print(results_df.to_string(index=False))

# ----------------------------------------------------------------------------
# 5. ROC CURVES
# ----------------------------------------------------------------------------
line_colors = {"Logistic Regression": BRASS, "Random Forest": FOREST}
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, _, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=line_colors.get(name, INK), linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color=INK_SOFT, alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Model Comparison")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/roc_curves.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------------
# 6. PRECISION-RECALL CURVES (the key evaluation chart for this problem)
# ----------------------------------------------------------------------------
baseline = y_test.mean()
plt.figure(figsize=(7, 6))
for name, (prec_arr, rec_arr, _, ap) in pr_data.items():
    plt.plot(rec_arr, prec_arr, label=f"{name} (PR-AUC={ap:.3f})", color=line_colors.get(name, INK), linewidth=2)
plt.axhline(baseline, color=INK_SOFT, linestyle="--", alpha=0.5, label=f"Random baseline ({baseline:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves — Model Comparison")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/pr_curves.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
for ax, (name, pipe) in zip(axes, fitted_pipes.items()):
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    ax.set_xlim(-0.95, 2.05)
    ax.set_ylim(-1.15, 2.3)
    ax.invert_yaxis()
    ax.axis("off")
    labels = ["Genuine", "Fraud"]
    for i in range(2):
        for j in range(2):
            is_error = i != j
            face = "#F3EEE3" if is_error else PAPER
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=face, edgecolor=RULE, linewidth=1))
            ax.text(j + 0.5, i + 0.5, f"{cm[i, j]:,}", ha="center", va="center",
                     fontsize=15, fontweight="bold" if not is_error else "normal",
                     color=OXBLOOD if is_error else INK, family="monospace")
    ax.text(1.0, -0.9, name, ha="center", va="top", fontsize=14, fontweight="bold", color=INK)
    ax.text(1.0, -0.5, "Predicted", ha="center", va="top", fontsize=10.5, color=INK)
    for j, lab in enumerate(labels):
        ax.text(j + 0.5, -0.12, lab, ha="center", va="bottom", fontsize=10.5, color=INK_SOFT)
    for i, lab in enumerate(labels):
        ax.text(-0.1, i + 0.5, lab, ha="right", va="center", fontsize=10.5, color=INK_SOFT)
    ax.text(-0.75, 1.0, "Actual", ha="center", va="center", fontsize=11, color=INK, rotation=90)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/confusion_matrices.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------------
# 8. FEATURE IMPORTANCE (Random Forest)
# ----------------------------------------------------------------------------
rf_pipe = fitted_pipes["Random Forest"]
feature_names = X.columns.tolist()
importances = rf_pipe.named_steps["clf"].feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False).head(15)
fi_df.to_csv(f"{RESULTS_DIR}/feature_importance.csv", index=False)

fi_palette = sns.light_palette(OXBLOOD, n_colors=len(fi_df) + 4, reverse=True)[2:2 + len(fi_df)]
plt.figure(figsize=(8, 7))
sns.barplot(x="importance", y="feature", data=fi_df, palette=fi_palette)
plt.title("Top 15 Feature Importances (Random Forest)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/feature_importance.png", dpi=120)
plt.close()

print("\nAll figures and result tables saved to /results")
