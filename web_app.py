import streamlit as st
import joblib
import pandas as pd
import shap

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "models/fraud_detection_model.pkl"
TEST_DATA = "data/test.csv"
THRESHOLD = 0.40

FEATURE_NAMES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Real-Time Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND TEST DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_test_data():
    return pd.read_csv(TEST_DATA)


model = load_model()
df = load_test_data()


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ Real-Time Fraud Detection System")

st.write(
    "Machine-learning based transaction fraud detection "
    "using a Random Forest classifier."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Model Information")

st.sidebar.write("**Algorithm:** Random Forest")
st.sidebar.write("**Features:** 30")
st.sidebar.write("**Decision Threshold:** 0.40")
st.sidebar.write("**PR-AUC:** 0.7869")
st.sidebar.write("**Fraud F1:** 0.8304")

st.sidebar.divider()

st.sidebar.info(
    "A transaction is classified as fraud when "
    "its predicted fraud probability is ≥ 40%."
)


# ============================================================
# TRANSACTION SECTION
# ============================================================

st.subheader("🧪 Test a Real Transaction")

fraud_rows = df[df["Class"] == 1]
legitimate_rows = df[df["Class"] == 0]

sample_type = st.radio(
    "Choose transaction type",
    [
        "🟢 Legitimate sample",
        "🔴 Fraud sample",
        "✏️ Manual input"
    ],
    horizontal=True
)


# ============================================================
# SAMPLE SELECTION
# ============================================================

if sample_type == "🟢 Legitimate sample":

    if len(legitimate_rows) == 0:
        st.error("No legitimate transaction exists in the test dataset.")
        st.stop()

    sample = legitimate_rows.iloc[0]

    st.success(
        "A real legitimate transaction from the test dataset "
        "has been loaded."
    )

elif sample_type == "🔴 Fraud sample":

    if len(fraud_rows) == 0:
        st.error("No fraud transaction exists in the test dataset.")
        st.stop()

    sample = fraud_rows.iloc[0]

    st.error(
        "A real fraud transaction from the test dataset "
        "has been loaded."
    )

else:

    sample = None


# ============================================================
# INPUT
# ============================================================

if sample_type == "✏️ Manual input":

    st.info(
        "Enter the 30 numerical features used by the trained model."
    )

    values = []

    cols = st.columns(3)

    for i, feature in enumerate(FEATURE_NAMES):

        with cols[i % 3]:

            value = st.number_input(
                feature,
                value=0.0,
                format="%.6f",
                key=f"manual_{feature}"
            )

            values.append(value)

    input_data = pd.DataFrame(
        [values],
        columns=FEATURE_NAMES
    )

else:

    input_data = pd.DataFrame(
        [[sample[feature] for feature in FEATURE_NAMES]],
        columns=FEATURE_NAMES
    )

    with st.expander("🔎 View transaction features"):

        st.dataframe(
            input_data,
            use_container_width=True
        )


# ============================================================
# ANALYSIS
# ============================================================

st.divider()

if st.button(
    "🔍 Analyze Transaction",
    type="primary",
    use_container_width=True
):

    probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = probability >= THRESHOLD


    # ========================================================
    # DETECTION RESULT
    # ========================================================

    st.subheader("📊 Detection Result")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Decision Threshold",
            f"{THRESHOLD * 100:.0f}%"
        )

    with col3:

        st.metric(
            "Prediction",
            "FRAUD" if prediction else "LEGITIMATE"
        )


    if prediction:

        st.error("🚨 FRAUD DETECTED")

        st.warning(
            "The predicted fraud probability is at or above "
            "the configured decision threshold. Further "
            "review is recommended."
        )

    else:

        st.success("✅ TRANSACTION APPEARS LEGITIMATE")

        st.info(
            "The predicted fraud probability is below "
            "the configured decision threshold."
        )


    # ========================================================
    # RISK BAR
    # ========================================================

    st.write("### Fraud Risk")

    st.progress(
        min(float(probability), 1.0),
        text=f"{probability * 100:.2f}% fraud probability"
    )


    # ========================================================
    # DATASET VERIFICATION
    # ========================================================

    if sample_type != "✏️ Manual input":

        actual = int(sample["Class"])

        st.divider()

        st.subheader("🧾 Dataset Verification")

        actual_text = (
            "FRAUD"
            if actual == 1
            else "LEGITIMATE"
        )

        st.write(
            f"**Actual dataset label:** `{actual_text}`"
        )

        if prediction == bool(actual):

            st.success(
                "✅ Model prediction matches the actual label."
            )

        else:

            st.warning(
                "⚠️ Model prediction does not match the actual label."
            )


# ============================================================
# MODEL EXPLAINABILITY
# ============================================================

st.divider()

st.subheader("🧠 Model Explainability")

st.write(
    "This section explains which features are most influential "
    "for the Random Forest model and for the selected transaction."
)


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

st.write("### 🌍 Global Feature Importance")

st.caption(
    "These features have the greatest overall importance "
    "across the trained Random Forest model."
)

importance_df = pd.DataFrame({
    "Feature": FEATURE_NAMES,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

top_features = importance_df.head(10)

st.bar_chart(
    top_features.set_index("Feature")
)

st.dataframe(
    top_features,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# INDIVIDUAL SHAP EXPLANATION
# ============================================================

st.write("### 🔬 Individual Transaction Explanation")

st.caption(
    "Positive SHAP values push the model toward fraud; "
    "negative SHAP values push the model toward legitimate."
)

try:

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        input_data
    )

    # --------------------------------------------------------
    # HANDLE DIFFERENT SHAP OUTPUT FORMATS
    # --------------------------------------------------------

    if isinstance(shap_values, list):

        if len(shap_values) > 1:
            individual_values = shap_values[1][0]
        else:
            individual_values = shap_values[0][0]

    else:

        individual_values = shap_values

        if len(individual_values.shape) == 3:

            individual_values = individual_values[0, :, 1]

        elif len(individual_values.shape) == 2:

            individual_values = individual_values[0]


    individual_values = individual_values.flatten()


    shap_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "SHAP Value": individual_values
    })

    shap_df["Absolute Impact"] = (
        shap_df["SHAP Value"].abs()
    )

    shap_df = shap_df.sort_values(
        "Absolute Impact",
        ascending=False
    )

    top_shap = shap_df.head(10)


    # --------------------------------------------------------
    # TOP SHAP FACTORS
    # --------------------------------------------------------

    st.write("#### Top factors for this transaction")

    st.dataframe(
        top_shap[
            ["Feature", "SHAP Value"]
        ],
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # SHAP CHART
    # --------------------------------------------------------

    st.write("#### SHAP Contribution")

    chart_df = (
        top_shap
        .set_index("Feature")["SHAP Value"]
        .sort_values()
    )

    st.bar_chart(chart_df)


    # --------------------------------------------------------
    # POSITIVE / NEGATIVE CONTRIBUTORS
    # --------------------------------------------------------

    positive = shap_df[
        shap_df["SHAP Value"] > 0
    ].sort_values(
        "SHAP Value",
        ascending=False
    ).head(5)

    negative = shap_df[
        shap_df["SHAP Value"] < 0
    ].sort_values(
        "SHAP Value"
    ).head(5)


    col1, col2 = st.columns(2)


    with col1:

        st.write("#### 🚨 Pushes toward fraud")

        if len(positive) > 0:

            for _, row in positive.iterrows():

                st.write(
                    f"**{row['Feature']}** "
                    f"`+{row['SHAP Value']:.5f}`"
                )

        else:

            st.write(
                "No strong positive contributors."
            )


    with col2:

        st.write("#### ✅ Pushes toward legitimate")

        if len(negative) > 0:

            for _, row in negative.iterrows():

                st.write(
                    f"**{row['Feature']}** "
                    f"`{row['SHAP Value']:.5f}`"
                )

        else:

            st.write(
                "No strong negative contributors."
            )


    st.info(
        "SHAP values describe the model's contribution for "
        "this prediction. They should not be interpreted as "
        "proof that a particular feature caused fraud."
    )


except Exception as e:

    st.warning(
        f"SHAP explanation could not be generated: {e}"
    )


# ============================================================
# MODEL PERFORMANCE DASHBOARD
# ============================================================

st.divider()

st.subheader("📊 Model Performance Dashboard")

st.write(
    "Final Random Forest performance evaluated on the "
    "test dataset."
)


# ============================================================
# TEST SET PREDICTIONS
# ============================================================

X_test = df[FEATURE_NAMES]
y_test = df["Class"]

test_probabilities = model.predict_proba(
    X_test
)[:, 1]

test_predictions = (
    test_probabilities >= THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

pr_auc = average_precision_score(
    y_test,
    test_probabilities
)

report = classification_report(
    y_test,
    test_predictions,
    output_dict=True
)

fraud_precision = report["1"]["precision"]
fraud_recall = report["1"]["recall"]
fraud_f1 = report["1"]["f1-score"]

cm = confusion_matrix(
    y_test,
    test_predictions
)

true_negative = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_positive = cm[1][1]


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "ROC-AUC",
        f"{roc_auc:.4f}"
    )

with col2:

    st.metric(
        "PR-AUC",
        f"{pr_auc:.4f}"
    )

with col3:

    st.metric(
        "Fraud Precision",
        f"{fraud_precision:.4f}"
    )

with col4:

    st.metric(
        "Fraud Recall",
        f"{fraud_recall:.4f}"
    )


col5, col6, col7, col8 = st.columns(4)

with col5:

    st.metric(
        "Fraud F1",
        f"{fraud_f1:.4f}"
    )

with col6:

    st.metric(
        "Fraud Detected",
        f"{true_positive}"
    )

with col7:

    st.metric(
        "Fraud Missed",
        f"{false_negative}"
    )

with col8:

    st.metric(
        "False Alarms",
        f"{false_positive}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.write("### 🎯 Confusion Matrix")

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual Legitimate",
        "Actual Fraud"
    ],
    columns=[
        "Predicted Legitimate",
        "Predicted Fraud"
    ]
)

st.dataframe(
    cm_df,
    use_container_width=True
)


# ============================================================
# CONFUSION MATRIX INTERPRETATION
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.success(
        f"✅ Correct legitimate transactions: "
        f"{true_negative}"
    )

    st.success(
        f"🚨 Correctly detected fraud: "
        f"{true_positive}"
    )


with col2:

    st.warning(
        f"⚠️ False alarms: "
        f"{false_positive}"
    )

    st.warning(
        f"❌ Missed fraud transactions: "
        f"{false_negative}"
    )


# ============================================================
# CLASS PERFORMANCE
# ============================================================

st.write("### 📋 Classification Performance")

performance_df = pd.DataFrame({
    "Metric": [
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Fraud Class": [
        fraud_precision,
        fraud_recall,
        fraud_f1
    ]
})

st.dataframe(
    performance_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODEL SUMMARY
# ============================================================

st.write("### 🏆 Final Model Summary")

summary = pd.DataFrame({
    "Property": [
        "Algorithm",
        "Features",
        "Decision Threshold",
        "ROC-AUC",
        "PR-AUC",
        "Fraud Precision",
        "Fraud Recall",
        "Fraud F1"
    ],
    "Value": [
        "Random Forest",
        "30",
        "0.40",
        f"{roc_auc:.4f}",
        f"{pr_auc:.4f}",
        f"{fraud_precision:.4f}",
        f"{fraud_recall:.4f}",
        f"{fraud_f1:.4f}"
    ]
})

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# BATCH TRANSACTION ANALYSIS
# ============================================================

st.divider()

st.subheader("📂 Batch Transaction Analysis")

st.write(
    "Upload a CSV containing transaction features and analyze "
    "multiple transactions using the trained Random Forest model."
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        batch_df = pd.read_csv(
            uploaded_file
        )

        st.write(
            f"Loaded **{len(batch_df):,} transactions**."
        )


        # ----------------------------------------------------
        # CHECK FEATURES
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in FEATURE_NAMES
            if feature not in batch_df.columns
        ]


        if missing_features:

            st.error(
                "Missing required features: "
                + ", ".join(missing_features)
            )


        else:

            batch_input = batch_df[
                FEATURE_NAMES
            ].copy()


            # ------------------------------------------------
            # PREDICTIONS
            # ------------------------------------------------

            batch_probabilities = model.predict_proba(
                batch_input
            )[:, 1]

            batch_predictions = (
                batch_probabilities >= THRESHOLD
            ).astype(int)


            results_df = batch_df.copy()


            results_df["Fraud Probability"] = (
                batch_probabilities
            )


            results_df["Prediction"] = [
                "FRAUD"
                if prediction == 1
                else "LEGITIMATE"
                for prediction in batch_predictions
            ]


            # ------------------------------------------------
            # RISK LEVEL
            # ------------------------------------------------

            def get_risk_level(probability):

                if probability >= 0.70:

                    return "🔴 CRITICAL"

                elif probability >= 0.40:

                    return "🟠 HIGH"

                elif probability >= 0.10:

                    return "🟡 MEDIUM"

                else:

                    return "🟢 LOW"


            results_df["Risk Level"] = [
                get_risk_level(probability)
                for probability in batch_probabilities
            ]


            # ------------------------------------------------
            # SORT HIGHEST RISK FIRST
            # ------------------------------------------------

            results_df = results_df.sort_values(
                "Fraud Probability",
                ascending=False
            ).reset_index(
                drop=True
            )


            # ------------------------------------------------
            # BATCH METRICS
            # ------------------------------------------------

            total_transactions = len(
                results_df
            )

            fraud_count = int(
                batch_predictions.sum()
            )

            legitimate_count = (
                total_transactions - fraud_count
            )

            fraud_percentage = (
                fraud_count
                / total_transactions
                * 100
                if total_transactions > 0
                else 0
            )

            average_probability = (
                batch_probabilities.mean()
                * 100
                if total_transactions > 0
                else 0
            )


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "Transactions",
                    f"{total_transactions:,}"
                )


            with col2:

                st.metric(
                    "Fraud Detected",
                    f"{fraud_count:,}"
                )


            with col3:

                st.metric(
                    "Legitimate",
                    f"{legitimate_count:,}"
                )


            with col4:

                st.metric(
                    "Fraud Rate",
                    f"{fraud_percentage:.2f}%"
                )


            st.metric(
                "Average Fraud Probability",
                f"{average_probability:.2f}%"
            )


            # ------------------------------------------------
            # RISK DISTRIBUTION
            # ------------------------------------------------

            st.write(
                "### 🚨 Risk Distribution"
            )

            risk_counts = (
                results_df["Risk Level"]
                .value_counts()
            )

            risk_df = pd.DataFrame({
                "Risk Level": risk_counts.index,
                "Transactions": risk_counts.values
            })

            st.dataframe(
                risk_df,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # ALL PREDICTIONS
            # ------------------------------------------------

            st.write(
                "### 🔎 Prediction Results"
            )

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # FRAUD TRANSACTIONS
            # ------------------------------------------------

            fraud_results = results_df[
                results_df["Prediction"] == "FRAUD"
            ]


            st.write(
                "### 🚨 Detected Fraud Transactions"
            )


            if len(fraud_results) > 0:

                st.dataframe(
                    fraud_results,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.success(
                    "No transactions crossed "
                    "the fraud threshold."
                )


            # ------------------------------------------------
            # PROBABILITY DISTRIBUTION
            # ------------------------------------------------

            st.write(
                "### 📈 Fraud Probability Distribution"
            )

            probability_chart = pd.DataFrame({
                "Fraud Probability (%)":
                    batch_probabilities * 100
            })

            st.bar_chart(
                probability_chart
            )


            # ------------------------------------------------
            # DOWNLOAD RESULTS
            # ------------------------------------------------

            csv_data = (
                results_df
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                label="⬇️ Download Prediction Results",
                data=csv_data,
                file_name="fraud_detection_results.csv",
                mime="text/csv",
                use_container_width=True
            )


    except Exception as e:

        st.error(
            f"Could not process the uploaded file: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Real-Time Fraud Detection System | "
    "Random Forest | "
    "30 Features | "
    "Threshold = 0.40"
)