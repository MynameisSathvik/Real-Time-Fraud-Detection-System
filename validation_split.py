import pandas as pd

from sklearn.model_selection import train_test_split


INPUT_FILE = "data/train.csv"

TRAIN_FILE = "data/train_final.csv"
VALIDATION_FILE = "data/validation.csv"


print("=" * 70)
print("FRAUD DETECTION - TRAIN / VALIDATION SPLIT")
print("=" * 70)


# ============================================================
# LOAD EXISTING TRAINING DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

X = df.drop("Class", axis=1)
y = df["Class"]

print(f"\nOriginal training data: {len(df)}")


print("\nOriginal class distribution:")
print(y.value_counts())


# ============================================================
# SPLIT
# ============================================================

X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# ============================================================
# RECONSTRUCT DATASETS
# ============================================================

train_final = X_train.copy()
train_final["Class"] = y_train

validation = X_validation.copy()
validation["Class"] = y_validation


# ============================================================
# SAVE
# ============================================================

train_final.to_csv(
    TRAIN_FILE,
    index=False
)

validation.to_csv(
    VALIDATION_FILE,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SPLIT RESULTS")
print("=" * 70)

print(f"\nFinal training samples: {len(train_final)}")
print(f"Validation samples:     {len(validation)}")

print("\nFinal training class distribution:")
print(y_train.value_counts())

print("\nValidation class distribution:")
print(y_validation.value_counts())

print("\nTraining fraud percentage:")
print(f"{y_train.mean() * 100:.4f}%")

print("\nValidation fraud percentage:")
print(f"{y_validation.mean() * 100:.4f}%")


print("\n" + "=" * 70)
print("Validation split completed successfully.")
print("=" * 70)