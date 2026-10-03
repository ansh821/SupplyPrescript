import os
import pandas as pd


# ============================================================
# SUPPLYPRESCRIPT - PRESCRIPTION ENGINE
# ============================================================

INPUT_PATH = "data/model_grouped/risk_predictions.csv"
OUTPUT_PATH = "data/model_grouped/prescriptions.csv"


print("=" * 70)
print("SUPPLYPRESCRIPT - PRESCRIPTION ENGINE")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load risk predictions
# ------------------------------------------------------------

print("\n[1] Loading risk predictions...")

df = pd.read_csv(INPUT_PATH)

print(f"Rows loaded: {len(df)}")


# ------------------------------------------------------------
# 2. Inspect available columns
# ------------------------------------------------------------

print("\nAvailable columns:")

for column in df.columns:
    print(f" - {column}")


# ------------------------------------------------------------
# 3. Generate operational prescriptions
# ------------------------------------------------------------

def generate_prescription(row):

    risk_level = str(row.get("Risk_Level", "")).upper()

    probability = row.get("Risk_Probability", 0)

    try:
        probability = float(probability)
    except (ValueError, TypeError):
        probability = 0.0


    # --------------------------------------------------------
    # HIGH RISK
    # --------------------------------------------------------

    if risk_level == "HIGH":

        if probability >= 0.90:
            action = "Escalate order immediately to logistics team"
            priority = "CRITICAL"

        else:
            action = "Prioritize shipment and monitor carrier status"
            priority = "HIGH"


    # --------------------------------------------------------
    # MEDIUM RISK
    # --------------------------------------------------------

    elif risk_level == "MEDIUM":

        action = "Monitor shipment closely and verify carrier schedule"
        priority = "MEDIUM"


    # --------------------------------------------------------
    # LOW RISK
    # --------------------------------------------------------

    else:

        action = "Continue standard shipment monitoring"
        priority = "LOW"


    return pd.Series({
        "Priority": priority,
        "Recommended_Action": action
    })


# ------------------------------------------------------------
# 4. Apply prescription engine
# ------------------------------------------------------------

print("\n[2] Generating prescriptions...")

prescriptions = df.apply(
    generate_prescription,
    axis=1
)

df["Priority"] = prescriptions["Priority"]

df["Recommended_Action"] = prescriptions["Recommended_Action"]


# ------------------------------------------------------------
# 5. Save results
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
# 6. Display summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PRESCRIPTION SUMMARY")
print("=" * 70)

print(
    df["Priority"]
    .value_counts()
    .to_string()
)


print("\nRecommended actions:")

print(
    df["Recommended_Action"]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# 7. Display sample
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE PRESCRIPTIONS")
print("=" * 70)

display_columns = [
    column
    for column in [
        "Risk_Probability",
        "Risk_Level",
        "Priority",
        "Recommended_Action"
    ]
    if column in df.columns
]

print(
    df[display_columns]
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 8. Final status
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PRESCRIPTION ENGINE COMPLETED")
print("=" * 70)

print(f"Output saved to:")
print(OUTPUT_PATH)