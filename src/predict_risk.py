import joblib
import pandas as pd


# ============================================================
# SUPPLYPRESCRIPT - RISK PREDICTION
# ============================================================

print("=" * 60)
print("SUPPLYPRESCRIPT RISK PREDICTION")
print("=" * 60)


# ------------------------------------------------------------
# 1. LOAD MODEL
# ------------------------------------------------------------

MODEL_PATH = "models/supplyprescript_random_forest.joblib"
TEST_PATH = "data/model_grouped/test.csv"

model = joblib.load(MODEL_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\nModel and test data loaded successfully.")


# ------------------------------------------------------------
# 2. TARGET
# ------------------------------------------------------------

TARGET = "Late_delivery_risk"

X_test = test_df.drop(columns=[TARGET])


# ------------------------------------------------------------
# 3. PREDICT
# ------------------------------------------------------------

predicted_class = model.predict(X_test)

risk_probability = model.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# 4. CREATE RESULTS
# ------------------------------------------------------------

results = test_df.copy()

results["Predicted_Risk"] = predicted_class

results["Risk_Probability"] = risk_probability


# ------------------------------------------------------------
# 5. RISK LEVEL
# ------------------------------------------------------------

def get_risk_level(probability):

    if probability >= 0.75:
        return "HIGH"

    elif probability >= 0.50:
        return "MEDIUM"

    else:
        return "LOW"


results["Risk_Level"] = results["Risk_Probability"].apply(
    get_risk_level
)


# ------------------------------------------------------------
# 6. PRESCRIPTION
# ------------------------------------------------------------

def generate_prescription(row):

    risk = row["Risk_Level"]

    shipping_mode = row.get("Shipping_Mode", "")

    order_region = row.get("Order_Region", "")

    if risk == "HIGH":

        if shipping_mode == "Standard Class":
            return (
                "Consider upgrading shipping priority "
                "and closely monitor fulfillment."
            )

        return (
            "Prioritize fulfillment and monitor shipment "
            "status closely."
        )

    elif risk == "MEDIUM":

        return (
            "Monitor fulfillment progress and review "
            "shipment scheduling."
        )

    else:

        return (
            "Continue normal fulfillment monitoring."
        )


results["Recommended_Action"] = results.apply(
    generate_prescription,
    axis=1
)


# ------------------------------------------------------------
# 7. DISPLAY SAMPLE RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE RISK PREDICTIONS")
print("=" * 60)

display_columns = [
    "Risk_Probability",
    "Risk_Level",
    "Recommended_Action"
]

print(
    results[display_columns]
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 8. RISK DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RISK DISTRIBUTION")
print("=" * 60)

print(
    results["Risk_Level"]
    .value_counts()
)


# ------------------------------------------------------------
# 9. SAVE RESULTS
# ------------------------------------------------------------

OUTPUT_PATH = "data/model_grouped/risk_predictions.csv"

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("PREDICTIONS SAVED")
print("=" * 60)

print(f"\nOutput file: {OUTPUT_PATH}")

print("\nRisk prediction completed successfully.")