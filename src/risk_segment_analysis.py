import os
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PREDICTIONS_PATH = "data/model_grouped/risk_predictions.csv"
RAW_DATA_PATH = "data/validated/validated_dataset.csv"

OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading prediction data...")
predictions = pd.read_csv(PREDICTIONS_PATH)

print("Loading cleaned dataset...")
raw_data = pd.read_csv(RAW_DATA_PATH)

print(f"Prediction data shape: {predictions.shape}")
print(f"Cleaned data shape: {raw_data.shape}")


# --------------------------------------------------
# IDENTIFY COMMON COLUMNS
# --------------------------------------------------

common_columns = [
    column
    for column in raw_data.columns
    if column in predictions.columns
]

print("\nCommon columns:")
print(common_columns)


# --------------------------------------------------
# BUILD ANALYSIS DATASET
# --------------------------------------------------

analysis_columns = [
    "Order_Region",
    "Order_State",
    "Shipping_Mode",
    "Category_Name",
    "Late_delivery_risk"
]

available_columns = [
    column
    for column in analysis_columns
    if column in predictions.columns
]

analysis_df = predictions[available_columns].copy()


# --------------------------------------------------
# RISK SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("OVERALL RISK SUMMARY")
print("=" * 60)

risk_summary = (
    predictions["Risk_Level"]
    .value_counts()
    .rename_axis("Risk_Level")
    .reset_index(name="Orders")
)

risk_summary["Percentage"] = (
    risk_summary["Orders"]
    / len(predictions)
    * 100
)

print(risk_summary)

risk_summary.to_csv(
    os.path.join(OUTPUT_DIR, "risk_segment_summary.csv"),
    index=False
)


# --------------------------------------------------
# REGION ANALYSIS
# --------------------------------------------------

if "Order_Region" in predictions.columns:

    print("\n" + "=" * 60)
    print("RISK BY REGION")
    print("=" * 60)

    region_analysis = (
        predictions
        .groupby("Order_Region")
        .agg(
            Orders=("Risk_Level", "size"),
            Average_Risk_Probability=(
                "Risk_Probability",
                "mean"
            ),
            High_Risk_Orders=(
                "Risk_Level",
                lambda x: (x == "HIGH").sum()
            ),
            Medium_Risk_Orders=(
                "Risk_Level",
                lambda x: (x == "MEDIUM").sum()
            )
        )
        .reset_index()
    )

    region_analysis["High_Risk_Rate"] = (
        region_analysis["High_Risk_Orders"]
        / region_analysis["Orders"]
    )

    region_analysis = region_analysis.sort_values(
        "High_Risk_Rate",
        ascending=False
    )

    print(region_analysis)

    region_analysis.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "risk_by_region.csv"
        ),
        index=False
    )


# --------------------------------------------------
# SHIPPING MODE ANALYSIS
# --------------------------------------------------

if "Shipping_Mode" in predictions.columns:

    print("\n" + "=" * 60)
    print("RISK BY SHIPPING MODE")
    print("=" * 60)

    shipping_analysis = (
        predictions
        .groupby("Shipping_Mode")
        .agg(
            Orders=("Risk_Level", "size"),
            Average_Risk_Probability=(
                "Risk_Probability",
                "mean"
            ),
            High_Risk_Orders=(
                "Risk_Level",
                lambda x: (x == "HIGH").sum()
            )
        )
        .reset_index()
    )

    shipping_analysis["High_Risk_Rate"] = (
        shipping_analysis["High_Risk_Orders"]
        / shipping_analysis["Orders"]
    )

    shipping_analysis = shipping_analysis.sort_values(
        "High_Risk_Rate",
        ascending=False
    )

    print(shipping_analysis)

    shipping_analysis.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "risk_by_shipping_mode.csv"
        ),
        index=False
    )


# --------------------------------------------------
# CATEGORY ANALYSIS
# --------------------------------------------------

if "Category_Name" in predictions.columns:

    print("\n" + "=" * 60)
    print("RISK BY PRODUCT CATEGORY")
    print("=" * 60)

    category_analysis = (
        predictions
        .groupby("Category_Name")
        .agg(
            Orders=("Risk_Level", "size"),
            Average_Risk_Probability=(
                "Risk_Probability",
                "mean"
            ),
            High_Risk_Orders=(
                "Risk_Level",
                lambda x: (x == "HIGH").sum()
            )
        )
        .reset_index()
    )

    category_analysis["High_Risk_Rate"] = (
        category_analysis["High_Risk_Orders"]
        / category_analysis["Orders"]
    )

    category_analysis = category_analysis.sort_values(
        "High_Risk_Rate",
        ascending=False
    )

    print(category_analysis)

    category_analysis.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "risk_by_category.csv"
        ),
        index=False
    )


# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print("\n" + "=" * 60)
print("RISK SEGMENT ANALYSIS COMPLETED")
print("=" * 60)

print("\nGenerated files:")

for file in [
    "risk_segment_summary.csv",
    "risk_by_region.csv",
    "risk_by_shipping_mode.csv",
    "risk_by_category.csv"
]:
    path = os.path.join(OUTPUT_DIR, file)

    if os.path.exists(path):
        print(f"✓ {path}")