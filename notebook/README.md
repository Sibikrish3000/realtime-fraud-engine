# 🛡️ Behavioral Fraud Detection System (XGBoost + SHAP)

**A production-grade machine learning pipeline identifying fraudulent credit card transactions with 97% precision and explainable AI.**

## 💼 Executive Summary & Business Impact

Financial fraud detection is not just about accuracy; it's about the **Precision-Recall trade-off**. A model that flags too many legitimate transactions (False Positives) causes customer churn, while missing fraud (False Negatives) causes direct financial loss.

This project implements a cost-sensitive **XGBoost** classifier engineered to minimize **operational friction**. By optimizing the decision threshold to **0.895**, the system achieved:

| Metric | Performance | Business Value |
| --- | --- | --- |
| **Precision** | **97%** | Only 3% of alerts are false alarms, drastically reducing manual review costs. |
| **Recall** | **80%** | Captures 80% of all fraud attempts (approx. $810k in prevented loss). |
| **Net Savings** | **$810,470** | Calculated ROI on the test set (Loss Prevented - Operational Costs). |
| **Latency** | **<50ms** | Inference speed optimized via `scikit-learn` Pipeline serialization. |

---

## 🏗️ Technical Architecture

The solution moves beyond basic "fit-predict" workflows by implementing a robust preprocessing pipeline designed to prevent **data leakage** and handle extreme class imbalance (0.5% fraud rate).

### 1. Advanced Feature Engineering

Raw transaction data is insufficient for modern fraud. I engineered **14 behavioral features** to capture context:

* **Velocity Metrics (`trans_count_24h`, `amt_to_avg_ratio_24h`)**: Detects "burst" behavior where a card is used rapidly or for amounts exceeding the user's historical norm.
* **Geospatial Analysis (`distance_km`)**: Calculates the Haversine distance between the cardholder's home and the merchant.
* **Cyclical Temporal Encoding (`hour_sin`, `hour_cos`)**: Captures high-risk time windows (e.g., 3 AM surges) while preserving the 24-hour cycle continuity.
* **Risk Profiling (`WOEEncoder`)**: Replaces high-cardinality categorical features (Merchant, Job) with their "Weight of Evidence" - a measure of how much a specific category supports the "Fraud" hypothesis.

### 2. The Pipeline

To ensure production stability, all steps are wrapped in a single Scikit-Learn `Pipeline`:

```python
pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(transformers=[
        ('cat', WOEEncoder(), ['job', 'category']),
        ('num', RobustScaler(), numerical_features)
    ])),
    ('classifier', XGBClassifier(scale_pos_weight=imbalance_ratio, ...))
])

```

---

## 📊 Model Performance

### Precision-Recall Strategy

Instead of optimizing for ROC-AUC (which can be misleading in imbalanced datasets), I optimized for **PR-AUC (0.998)**.

* **Default Threshold (0.50):** Precision was 65%. Too many false alarms.
* **Optimized Threshold (0.895):** Precision increased to **97%**, with minimal loss in Recall.

*(Insert your Precision-Recall Curve image here)*

---

## 🔍 Explainability & "The Why"

Black-box models are dangerous in finance. I implemented **SHAP (SHapley Additive exPlanations)** to provide reason codes for every decision.

### The "Smoking Gun" (Fraud Example)

For a transaction flagged with **99.9% confidence**, the SHAP Waterfall plot reveals the exact drivers:

1. **`amt_log` (+8.83)**: The transaction amount was significantly higher than normal.
2. **`hour_sin` (+2.46)**: The transaction occurred during a high-risk time window (late night).
3. **`job` (+1.72)**: The cardholder's profession falls into a statistically higher-risk segment.
4. **`amt_to_avg_ratio_24h` (+1.24)**: The amount was an outlier *specifically* for this user's 24-hour history.

*(Insert your SHAP Waterfall Plot image here)*

---

## 🚀 How to Run

### Prerequisites

```bash
pip install pandas xgboost category_encoders shap scikit-learn

```

### Inference

The model is serialized as a `.pkl` file. You can load it to predict on new data immediately without re-training.

```python
import joblib
import pandas as pd

# Load the production pipeline
model = joblib.load('fraud_detection_model_v1.pkl')

# Define a new transaction (Example)
new_transaction = pd.DataFrame([{
    'amt_log': 5.2,
    'distance_km': 120.5,
    'trans_count_24h': 12,
    'amt_to_avg_ratio_24h': 4.5,
    # ... include all 14 features
}])

# Get prediction (Probability of Fraud)
fraud_prob = model.predict_proba(new_transaction)[:, 1]
is_fraud = (fraud_prob >= 0.895).astype(int)

print(f"Fraud Probability: {fraud_prob[0]:.4f}")
print(f"Action: {'BLOCK' if is_fraud[0] else 'APPROVE'}")

```

---

### **Author**

**Sibi Krishnamoorthy** *Machine Learning Engineer | Fintech & Risk Analytics*