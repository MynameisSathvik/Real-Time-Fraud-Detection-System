import pandas as pd

INPUT_FILE = "data/creditcard.csv"
OUTPUT_FILE = "data/creditcard_clean.csv"

print("=" * 70)
print("FRAUD DETECTION - DATA PREPARATION")
print("=" * 70)

# Load dataset
df = pd.read_csv(INPUT_FILE)

print(f"\nOriginal dataset shape: {df.shape}")
print(f"Original duplicates: {df.duplicated().sum()}")

# Check label distribution before cleaning
print("\nClass distribution BEFORE cleaning:")
print(df["Class"].value_counts())

# Remove exact duplicate rows
df_clean = df.drop_duplicates().reset_index(drop=True)

print(f"\nCleaned dataset shape: {df_clean.shape}")
print(f"Rows removed: {len(df) - len(df_clean)}")

# Verify no duplicates remain
print(f"Remaining duplicates: {df_clean.duplicated().sum()}")

# Check label distribution after cleaning
print("\nClass distribution AFTER cleaning:")
print(df_clean["Class"].value_counts())

# Class percentages
print("\nClass percentage AFTER cleaning:")
print(df_clean["Class"].value_counts(normalize=True) * 100)

# Save cleaned dataset
df_clean.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print(f"Cleaned dataset saved to: {OUTPUT_FILE}")
print("=" * 70)