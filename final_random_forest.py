import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


TRAIN_FILE = "data/train_final.csv"
TEST_FILE = "data/test.csv"

FINAL_THRESHOLD = 0.40


print("=" * 80)
print("RANDOM FOREST - FINAL UNTOUCHED TEST")
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
print(f"Final test samples: {len(X_test)}")

print(f"\nLocked threshold: {FINAL_THRESHOLD}")


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)


print("\nTraining final Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Training complete!")


# ============================================================
# PREDICTIONS
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= FINAL_THRESHOLD
).astype(int)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 80)
print("FINAL CLASSIFICATION REPORT")
print("=" * 80)

print(
    classification_report(
        y_test,
        predictions,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nCONFUSION MATRIX")

matrix = confusion_matrix(
    y_test,
    predictions
)

print(matrix)


# ============================================================
# AUC METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)


print("\n" + "=" * 80)
print("FINAL METRICS")
print("=" * 80)

print(f"\nROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")


# ============================================================
# FRAUD SUMMARY
# ============================================================

actual_fraud = (
    y_test == 1
).sum()

detected_fraud = (
    (y_test == 1) &
    (predictions == 1)
).sum()

missed_fraud = (
    (y_test == 1) &
    (predictions == 0)
).sum()

false_alarms = (
    (y_test == 0) &
    (predictions == 1)
).sum()


print("\n" + "=" * 80)
print("FRAUD DETECTION SUMMARY")
print("=" * 80)

print(f"\nActual fraud transactions: {actual_fraud}")
print(f"Fraud detected:            {detected_fraud}")
print(f"Fraud missed:              {missed_fraud}")
print(f"False alarms:              {false_alarms}")


print("\n" + "=" * 80)
print("FINAL RANDOM FOREST TEST COMPLETED")
print("=" * 80)