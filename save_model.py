import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


TRAIN_FILE = "data/train_final.csv"
MODEL_FILE = "models/fraud_detection_model.pkl"

print("=" * 70)
print("FRAUD DETECTION - MODEL SAVING")
print("=" * 70)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)

X_train = train_df.drop("Class", axis=1)
y_train = train_df["Class"]

print(f"\nTraining samples: {len(X_train)}")
print(f"Features: {X_train.shape[1]}")


# ============================================================
# TRAIN CHAMPION MODEL
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
# SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)


# ============================================================
# VERIFY
# ============================================================

file_size = os.path.getsize(
    MODEL_FILE
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(f"\nModel file: {MODEL_FILE}")
print(f"File size: {file_size / (1024 * 1024):.2f} MB")

print("\nThreshold to use: 0.40")

print("\n" + "=" * 70)
print("Model saving completed successfully.")
print("=" * 70)