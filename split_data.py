import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/creditcard_clean.csv"

TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"

print("=" * 70)
print("FRAUD DETECTION - TRAIN / TEST SPLIT")
print("=" * 70)

# Load cleaned dataset
df = pd.read_csv(INPUT_FILE)

X = df.drop("Class", axis=1)
y = df["Class"]

print(f"\nDataset shape: {df.shape}")

print("\nOriginal class distribution:")
print(y.value_counts())

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Reconstruct datasets
train_df = X_train.copy()
train_df["Class"] = y_train

test_df = X_test.copy()
test_df["Class"] = y_test

# Save
train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print("\n" + "=" * 70)
print("SPLIT RESULTS")
print("=" * 70)

print(f"\nTraining samples: {len(train_df)}")
print(f"Testing samples:  {len(test_df)}")

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\nTraining fraud percentage:")
print(f"{y_train.mean() * 100:.4f}%")

print("\nTesting fraud percentage:")
print(f"{y_test.mean() * 100:.4f}%")

print("\n" + "=" * 70)
print("Train/test split completed successfully.")
print("=" * 70)