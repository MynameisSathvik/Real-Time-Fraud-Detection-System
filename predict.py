import joblib
import numpy as np


MODEL_FILE = "models/fraud_detection_model.pkl"
THRESHOLD = 0.40


print("=" * 70)
print("REAL-TIME FRAUD DETECTION")
print("=" * 70)

# Load trained model
model = joblib.load(MODEL_FILE)

print("\nModel loaded successfully.")
print(f"Fraud threshold: {THRESHOLD}")

print("\nEnter 30 transaction features.")
print("Order:")
print("Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,")
print("V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,")
print("V21, V22, V23, V24, V25, V26, V27, V28, Amount")

print("\nType 'exit' to stop.")


while True:

    user_input = input("\nTransaction features: ")

    if user_input.lower() == "exit":
        print("\nGoodbye! 👋")
        break

    try:

        values = [
            float(x.strip())
            for x in user_input.split(",")
        ]

        if len(values) != 30:
            print(
                f"❌ Expected 30 values, but received {len(values)}."
            )
            continue

        features = np.array(values).reshape(1, -1)

        probability = model.predict_proba(
            features
        )[0][1]

        prediction = (
            probability >= THRESHOLD
        )

        print("\n" + "-" * 70)

        print(
            f"Fraud probability: {probability * 100:.2f}%"
        )

        if prediction:

            print("🚨 FRAUD DETECTED")

            print(
                "⚠️ Transaction should be reviewed."
            )

        else:

            print("✅ LEGITIMATE TRANSACTION")

            print(
                "Transaction appears normal."
            )

        print("-" * 70)

    except ValueError:

        print(
            "❌ Invalid input. Enter numeric values separated by commas."
        )