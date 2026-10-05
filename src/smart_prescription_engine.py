import os
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

EXPLANATION_PATH = (
    "models/order_level_explanations.csv"
)

OUTPUT_DIR = "models"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "smart_prescriptions.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading order-level explanations...")

df = pd.read_csv(
    EXPLANATION_PATH
)

print(
    f"Loaded {len(df)} order explanations."
)


# --------------------------------------------------
# PRESCRIPTION FUNCTION
# --------------------------------------------------

def generate_prescription(row):

    risk_level = row["Risk_Level"]
    probability = row["Risk_Probability"]

    factors = str(
        row["Top_Risk_Factors"]
    )


    # ----------------------------------------------
    # HIGH RISK
    # ----------------------------------------------

    if risk_level == "HIGH":

        if probability >= 0.90:

            priority = "CRITICAL"

            action = (
                "Escalate order immediately to "
                "the logistics team and prioritize "
                "shipment monitoring."
            )

        else:

            priority = "HIGH"

            action = (
                "Prioritize shipment and closely "
                "monitor carrier status."
            )


    # ----------------------------------------------
    # MEDIUM RISK
    # ----------------------------------------------

    elif risk_level == "MEDIUM":

        priority = "MEDIUM"

        action = (
            "Monitor shipment closely and verify "
            "carrier schedule before delay occurs."
        )


    # ----------------------------------------------
    # LOW RISK
    # ----------------------------------------------

    else:

        priority = "LOW"

        action = (
            "Continue standard shipment monitoring."
        )


    # ----------------------------------------------
    # EXPLANATION
    # ----------------------------------------------

    if factors:

        explanation = (
            "Prediction driven by: "
            + factors
        )

    else:

        explanation = (
            "No major explanatory factors available."
        )


    return pd.Series({
        "Priority": priority,
        "Recommended_Action": action,
        "Prediction_Explanation": explanation
    })


# --------------------------------------------------
# GENERATE PRESCRIPTIONS
# --------------------------------------------------

print(
    "\nGenerating smart prescriptions..."
)

prescriptions = df.apply(
    generate_prescription,
    axis=1
)


# --------------------------------------------------
# COMBINE RESULTS
# --------------------------------------------------

result = pd.concat(
    [
        df,
        prescriptions
    ],
    axis=1
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

result.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("SMART PRESCRIPTION SUMMARY")
print("=" * 70)

print(
    result["Priority"]
    .value_counts()
)


print("\nRecommended Actions:")

print(
    result[
        [
            "Priority",
            "Recommended_Action"
        ]
    ]
    .drop_duplicates()
    .to_string(index=False)
)


# --------------------------------------------------
# SAMPLE
# --------------------------------------------------

print("\n" + "=" * 70)
print("SMART PRESCRIPTION SAMPLE")
print("=" * 70)

print(
    result[
        [
            "Order_Index",
            "Risk_Probability",
            "Risk_Level",
            "Priority",
            "Recommended_Action",
            "Prediction_Explanation"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print("\n" + "=" * 70)
print("SMART PRESCRIPTION ENGINE COMPLETED")
print("=" * 70)

print(
    f"\nSaved to: {OUTPUT_PATH}"
)