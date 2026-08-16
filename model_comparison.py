import time
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


TRAIN_FILE = "data/train_final.csv"
TEST_FILE = "data/test.csv"

THRESHOLD = 0.10


print("=" * 80)
print("FRAUD DETECTION - MODEL COMPETITION")
print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

X_train = train_df.drop("Class", axis=1)
y_train = train_df["Class"]

X_test = test_df.drop("Class", axis=1)
y_test = test_df["Class"]


print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=42
    )
}


# ============================================================
# TRAIN + EVALUATE
# ============================================================

results = []


for name, model in models.items():

    print("\n" + "-" * 80)
    print(f"Training: {name}")

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Training Time (s)": training_time
    })

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Time:      {training_time:.2f}s")


# ============================================================
# COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format,
            "PR-AUC": "{:.4f}".format,
            "Training Time (s)": "{:.2f}".format
        }
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best = results_df.loc[
    results_df["PR-AUC"].idxmax()
]


print("\n" + "=" * 80)
print("🏆 BEST MODEL BY PR-AUC")
print("=" * 80)

print(f"\nModel:    {best['Model']}")
print(f"PR-AUC:   {best['PR-AUC']:.4f}")
print(f"Precision:{best['Precision']:.4f}")
print(f"Recall:   {best['Recall']:.4f}")
print(f"F1:       {best['F1']:.4f}")

print("\n" + "=" * 80)