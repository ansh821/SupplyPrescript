import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# SUPPLYPRESCRIPT - MODEL COMPARISON
# ============================================================

print("=" * 70)
print("SUPPLYPRESCRIPT MODEL COMPARISON")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD TEST DATA
# ------------------------------------------------------------

TEST_PATH = "data/model_grouped/test.csv"

test_df = pd.read_csv(TEST_PATH)

TARGET = "Late_delivery_risk"

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print("\nTest dataset loaded.")
print(f"Testing rows: {len(test_df):,}")


# ------------------------------------------------------------
# 2. LOAD MODELS
# ------------------------------------------------------------

models = {
    "Random Forest": "models/supplyprescript_random_forest.joblib",
    "XGBoost": "models/supplyprescript_xgboost.joblib"
}


# ------------------------------------------------------------
# 3. EVALUATE MODELS
# ------------------------------------------------------------

results = []

for model_name, model_path in models.items():

    print("\n" + "-" * 70)
    print(f"Evaluating: {model_name}")
    print("-" * 70)

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })


# ------------------------------------------------------------
# 4. CREATE COMPARISON TABLE
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ------------------------------------------------------------
# 5. SAVE RESULTS
# ------------------------------------------------------------

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\nComparison saved to:")
print("models/model_comparison.csv")

print("\nModel comparison completed successfully.")