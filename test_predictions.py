import joblib
import pandas as pd


MODEL_FILE = "models/fraud_detection_model.pkl"
TEST_FILE = "data/test.csv"

THRESHOLD = 0.40


print("=" * 70)
print("REAL-TIME FRAUD DETECTION - SAMPLE TEST")
print("=" * 70)


# Load model
model = joblib.load(MODEL_FILE)

# Load test data
df = pd.read_csv(TEST_FILE)

X = df.drop("Class", axis=1)
y = df["Class"]


# Select examples
fraud_indices = df.index[df["Class"] == 1][:5]
legitimate_indices = df.index[df["Class"] == 0][:5]

sample_indices = list(legitimate_indices) + list(fraud_indices)


print("\nTesting 10 real transactions...")
print("=" * 70)


for index in sample_indices:

    features = X.loc[[index]]

    actual = y.loc[index]

    probability = model.predict_proba(
        features
    )[0][1]

    prediction = int(
        probability >= THRESHOLD
    )

    predicted_label = (
        "FRAUD"
        if prediction == 1
        else "LEGITIMATE"
    )

    actual_label = (
        "FRAUD"
        if actual == 1
        else "LEGITIMATE"
    )

    result = (
        "✅"
        if prediction == actual
        else "❌"
    )

    print(
        f"{result} Transaction {index}"
    )

    print(
        f"   Actual:     {actual_label}"
    )

    print(
        f"   Prediction: {predicted_label}"
    )

    print(
        f"   Probability: {probability * 100:.2f}%"
    )

    print("-" * 70)


print("\nTesting completed.")