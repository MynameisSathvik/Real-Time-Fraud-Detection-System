\# 🔐 Real-Time Fraud Detection System



A machine-learning based fraud detection system that analyzes financial transactions and predicts whether a transaction is legitimate or fraudulent.



The system uses a \*\*Random Forest classifier\*\* with a tuned decision threshold and provides a Streamlit web dashboard for real-time and batch transaction analysis.



\---



\## 🚀 Features



\- Random Forest fraud detection

\- Real-time transaction prediction

\- Batch transaction analysis

\- Fraud probability estimation

\- Configurable decision threshold

\- Risk-level classification

\- SHAP-based model explainability

\- Feature importance visualization

\- Dataset verification

\- Prediction results download

\- Streamlit web dashboard



\---



\## 🤖 Machine Learning Model



\### Final Model



\*\*Algorithm:\*\* Random Forest



\*\*Number of Features:\*\* 30



\*\*Decision Threshold:\*\* 0.40



A transaction is classified as fraudulent when its predicted fraud probability is greater than or equal to 40%.



\---



\## 📊 Final Model Performance



The final model was evaluated on an untouched test dataset.



| Metric | Score |

|---|---:|

| Precision | 0.9342 |

| Recall | 0.7474 |

| F1-Score | 0.8304 |

| ROC-AUC | 0.9654 |

| PR-AUC | 0.7869 |

| Accuracy | 0.9995 |



\### Final Test Results



\- Actual fraud transactions: \*\*95\*\*

\- Fraud detected: \*\*71\*\*

\- Fraud missed: \*\*24\*\*

\- False alarms: \*\*5\*\*



\### Confusion Matrix



```text

&#x20;                Predicted

&#x20;              Legit   Fraud



Actual Legit   56646      5

Actual Fraud      24     71

