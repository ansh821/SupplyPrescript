import os
import joblib
import pandas as pd


# ============================================================
# SUPPLYPRESCRIPT - FINAL XGBOOST RISK PREDICTION
# ============================================================

MODEL_PATH = "models/supplyprescript_xgboost.joblib"
TEST_PATH = "data/model_grouped/test.csv"
OUTPUT_PATH = "data/model_grouped/risk_predictions.csv"


print("=" * 70)
print("SUPPLYPRESCRIPT - XGBOOST RISK PREDICTION")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load final XGBoost model
# ------------------------------------------------------------

print("\n[1] Loading XGBoost model...")

model = joblib.load(MODEL_PATH)

print("XGBoost model loaded successfully.")


# ------------------------------------------------------------
# 2. Load grouped test data
# ------------------------------------------------------------

print("\n[2] Loading grouped test data...")

df = pd.read_csv(TEST_PATH)

target_column = "Late_delivery_risk"

X = df.drop(columns=[target_column])

print(f"Dataset shape: {df.shape}")
print(f"Feature shape: {X.shape}")


# ------------------------------------------------------------
# 3. Generate predictions
# ------------------------------------------------------------

print("\n[3] Generating predictions...")

predictions = model.predict(X)

probabilities = model.predict_proba(X)[:, 1]


# ------------------------------------------------------------
# 4. Convert probability into risk level
# ------------------------------------------------------------

def get_risk_level(probability):

    if probability >= 0.75:
        return "HIGH"

    elif probability >= 0.50:
        return "MEDIUM"

    else:
        return "LOW"


risk_levels = [
    get_risk_level(probability)
    for probability in probabilities
]


# ------------------------------------------------------------
# 5. Add predictions to dataset
# ------------------------------------------------------------

df["Predicted_Late_Delivery_Risk"] = predictions

df["Risk_Probability"] = probabilities

df["Risk_Level"] = risk_levels


# ------------------------------------------------------------
# 6. Save predictions
# ------------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ------------------------------------------------------------
# 7. Display summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RISK LEVEL SUMMARY")
print("=" * 70)

print(
    df["Risk_Level"]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# 8. Display probability statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RISK PROBABILITY STATISTICS")
print("=" * 70)

print(
    df["Risk_Probability"]
    .describe()
    .to_string()
)


# ------------------------------------------------------------
# 9. Display sample predictions
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE XGBOOST PREDICTIONS")
print("=" * 70)

sample_columns = [
    target_column,
    "Predicted_Late_Delivery_Risk",
    "Risk_Probability",
    "Risk_Level"
]

print(
    df[sample_columns]
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 10. Final status
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("XGBOOST RISK PREDICTION COMPLETED")
print("=" * 70)

print(f"Prediction file:")
print(OUTPUT_PATH)