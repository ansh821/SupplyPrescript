import pandas as pd
import numpy as np
import os


# ============================================================
# SUPPLYPRESCRIPT - RISK THRESHOLD ANALYSIS
# ============================================================

INPUT_PATH = "data/model_grouped/risk_predictions.csv"
OUTPUT_PATH = "models/risk_threshold_analysis.csv"


print("=" * 70)
print("SUPPLYPRESCRIPT - RISK THRESHOLD ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load predictions
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print(f"\nPrediction rows: {len(df)}")


# ------------------------------------------------------------
# 2. Create probability bands
# ------------------------------------------------------------

df["Probability_Band"] = pd.cut(
    df["Risk_Probability"],
    bins=[
        -np.inf,
        0.25,
        0.50,
        0.75,
        1.00
    ],
    labels=[
        "< 0.25",
        "0.25 - 0.49",
        "0.50 - 0.74",
        ">= 0.75"
    ],
    right=False
)


# ------------------------------------------------------------
# 3. Calculate actual late-delivery rate
# ------------------------------------------------------------

summary = (
    df.groupby("Probability_Band", observed=False)
    .agg(
        Orders=("Late_delivery_risk", "count"),
        Actual_Late_Deliveries=("Late_delivery_risk", "sum"),
        Average_Risk_Probability=("Risk_Probability", "mean")
    )
    .reset_index()
)

summary["Actual_Late_Delivery_Rate"] = (
    summary["Actual_Late_Deliveries"]
    / summary["Orders"]
)


# ------------------------------------------------------------
# 4. Current risk-level distribution
# ------------------------------------------------------------

risk_distribution = (
    df["Risk_Level"]
    .value_counts()
    .rename_axis("Risk_Level")
    .reset_index(name="Orders")
)

risk_distribution["Percentage"] = (
    risk_distribution["Orders"]
    / len(df)
    * 100
)


# ------------------------------------------------------------
# 5. Display probability-band analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PROBABILITY BAND ANALYSIS")
print("=" * 70)

print(
    summary.to_string(index=False)
)


# ------------------------------------------------------------
# 6. Display current risk distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CURRENT RISK LEVEL DISTRIBUTION")
print("=" * 70)

print(
    risk_distribution.to_string(index=False)
)


# ------------------------------------------------------------
# 7. Save analysis
# ------------------------------------------------------------

os.makedirs("models", exist_ok=True)

summary.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS COMPLETED")
print("=" * 70)

print(f"Saved to:")
print(OUTPUT_PATH)