import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"


print("=" * 80)
print("FRAUD DETECTION - THRESHOLD ANALYSIS")
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


# ============================================================
# BASELINE MODEL
# ============================================================

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


print("\nTraining baseline Logistic Regression...")

model.fit(X_train, y_train)

print("Training complete!")


# ============================================================
# PROBABILITIES
# ============================================================

probabilities = model.predict_proba(X_test)[:, 1]


# ============================================================
# THRESHOLD TESTING
# ============================================================

thresholds = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50
]


results = []


print("\n" + "=" * 80)
print("THRESHOLD RESULTS")
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

    results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    )

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
    )


# ============================================================
# BEST F1
# ============================================================

results_df = pd.DataFrame(results)

best = results_df.loc[
    results_df["f1"].idxmax()
]


print("\n" + "=" * 80)
print("BEST F1 THRESHOLD")
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
print("Threshold analysis completed.")
print("=" * 80)