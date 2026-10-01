import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SUPPLYPRESCRIPT - FEATURE IMPORTANCE
# ============================================================

print("=" * 60)
print("SUPPLYPRESCRIPT FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# 1. LOAD MODEL
# ------------------------------------------------------------

MODEL_PATH = "models/supplyprescript_xgboost.joblib"

model_pipeline = joblib.load(MODEL_PATH)

print("\nXGBoost model loaded successfully.")


# ------------------------------------------------------------
# 2. GET PIPELINE COMPONENTS
# ------------------------------------------------------------

preprocessor = model_pipeline.named_steps["preprocessor"]
model = model_pipeline.named_steps["model"]


# ------------------------------------------------------------
# 3. GET ORIGINAL FEATURE NAMES
# ------------------------------------------------------------

numeric_features = preprocessor.transformers_[0][2]
categorical_features = preprocessor.transformers_[1][2]

feature_names = []


# Numeric features
feature_names.extend(numeric_features)


# Categorical features
categorical_pipeline = preprocessor.transformers_[1][1]

encoder = categorical_pipeline.named_steps["encoder"]

encoded_names = encoder.get_feature_names_out(
    categorical_features
)

feature_names.extend(encoded_names)


# ------------------------------------------------------------
# 4. GET FEATURE IMPORTANCE
# ------------------------------------------------------------

importance_values = model.feature_importances_


# Safety check
if len(feature_names) != len(importance_values):

    print("\nERROR:")
    print(
        f"Feature names: {len(feature_names)}"
    )

    print(
        f"Importance values: {len(importance_values)}"
    )

    raise ValueError(
        "Feature count does not match XGBoost importance count."
    )


# ------------------------------------------------------------
# 5. CREATE DATAFRAME
# ------------------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# ------------------------------------------------------------
# 6. DISPLAY TOP 20
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TOP 20 IMPORTANT FEATURES")
print("=" * 60)

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 7. SAVE CSV
# ------------------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

CSV_PATH = "models/xgboost_feature_importance.csv"

importance_df.to_csv(
    CSV_PATH,
    index=False
)

print(
    f"\nFeature importance saved to:"
)

print(CSV_PATH)


# ------------------------------------------------------------
# 8. CREATE CHART
# ------------------------------------------------------------

top_features = (
    importance_df
    .head(20)
    .sort_values(
        by="Importance",
        ascending=True
    )
)


plt.figure(
    figsize=(10, 8)
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "SupplyPrescript - XGBoost Feature Importance"
)

plt.tight_layout()


# ------------------------------------------------------------
# 9. SAVE CHART
# ------------------------------------------------------------

FIGURE_PATH = "models/xgboost_feature_importance.png"

plt.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"\nFeature importance chart saved to:"
)

print(FIGURE_PATH)


print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
print("=" * 60)