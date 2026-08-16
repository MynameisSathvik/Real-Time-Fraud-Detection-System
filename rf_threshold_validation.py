import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)


TRAIN_FILE = "data/train_final.csv"
VALIDATION_FILE = "data/validation.csv"


print("=" * 80)
print("RANDOM FOREST - VALIDATION THRESHOLD OPTIMIZATION")
print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

X_train = train_df.drop("Class", axis=1)
y_train = train_df["Class"]

X_validation = validation_df.drop("Class", axis=1)
y_validation = validation_df["Class"]


print(f"\nTraining samples:   {len(X_train)}")
print(f"Validation samples: {len(X_validation)}")


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)


print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Training complete!")


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    X_validation
)[:, 1]


# ============================================================
# VALIDATION PR-AUC
# ============================================================

pr_auc = average_precision_score(
    y_validation,
    probabilities
)

print(f"\nValidation PR-AUC: {pr_auc:.4f}")


# ============================================================
# THRESHOLD SEARCH
# ============================================================

thresholds = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]


results = []


print("\n" + "=" * 80)
print("RANDOM FOREST THRESHOLD RESULTS")
print("=" * 80)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
    )


# ============================================================
# BEST F1 THRESHOLD
# ============================================================

results_df = pd.DataFrame(results)

best = results_df.loc[
    results_df["f1"].idxmax()
]


print("\n" + "=" * 80)
print("BEST RANDOM FOREST THRESHOLD")
print("=" * 80)

print(
    f"\nThreshold: {best['threshold']:.2f}"
)

print(
    f"Precision: {best['precision']:.4f}"
)

print(
    f"Recall:    {best['recall']:.4f}"
)

print(
    f"F1:        {best['f1']:.4f}"
)

print("\n" + "=" * 80)
print("Random Forest threshold optimization completed.")
print("=" * 80)