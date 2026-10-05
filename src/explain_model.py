import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt


# ============================================================
# SUPPLYPRESCRIPT - XGBOOST MODEL EXPLAINABILITY
# ============================================================

MODEL_PATH = "models/supplyprescript_xgboost.joblib"
TEST_PATH = "data/model_grouped/test.csv"

OUTPUT_CSV = "models/shap_feature_importance.csv"
OUTPUT_PNG = "models/shap_summary.png"

print("=" * 70)
print("SUPPLYPRESCRIPT - XGBOOST SHAP EXPLAINABILITY")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load model
# ------------------------------------------------------------

print("\n[1] Loading XGBoost model...")

pipeline = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ------------------------------------------------------------
# 2. Load grouped test data
# ------------------------------------------------------------

print("\n[2] Loading grouped test data...")

test_df = pd.read_csv(TEST_PATH)

print(f"Test dataset shape: {test_df.shape}")


# ------------------------------------------------------------
# 3. Separate features and target
# ------------------------------------------------------------

target_column = "Late_delivery_risk"

X_test = test_df.drop(columns=[target_column])

print(f"Feature count: {X_test.shape[1]}")


# ------------------------------------------------------------
# 4. Get preprocessing and model
# ------------------------------------------------------------

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]


# ------------------------------------------------------------
# 5. Transform test data
# ------------------------------------------------------------

print("\n[3] Transforming test data...")

X_transformed = preprocessor.transform(X_test)

print(f"Transformed shape: {X_transformed.shape}")


# ------------------------------------------------------------
# 6. Build feature names
# ------------------------------------------------------------

print("\n[4] Building transformed feature names...")

numeric_features = preprocessor.transformers_[0][2]

categorical_features = preprocessor.transformers_[1][2]

categorical_pipeline = preprocessor.transformers_[1][1]

encoder = categorical_pipeline.named_steps["encoder"]

encoded_categorical_names = encoder.get_feature_names_out(
    categorical_features
)

feature_names = list(numeric_features) + list(encoded_categorical_names)

print(f"Total transformed features: {len(feature_names)}")


# ------------------------------------------------------------
# 7. Sample data for SHAP
# ------------------------------------------------------------

# The full test set can be expensive on an 8 GB RAM machine.
# Use a representative sample.

sample_size = min(3000, X_transformed.shape[0])

np.random.seed(42)

sample_indices = np.random.choice(
    X_transformed.shape[0],
    size=sample_size,
    replace=False
)

X_sample = X_transformed[sample_indices]

print(f"SHAP sample size: {sample_size}")


# ------------------------------------------------------------
# 8. Create SHAP Tree Explainer
# ------------------------------------------------------------

print("\n[5] Creating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)

print("SHAP explainer created.")


# ------------------------------------------------------------
# 9. Calculate SHAP values
# ------------------------------------------------------------

print("\n[6] Calculating SHAP values...")
print("This may take some time...")

shap_values = explainer.shap_values(X_sample)

print("SHAP calculation completed.")


# ------------------------------------------------------------
# 10. Handle binary classification output
# ------------------------------------------------------------

if isinstance(shap_values, list):
    shap_values_for_class_1 = shap_values[1]
else:
    shap_values_for_class_1 = shap_values


# ------------------------------------------------------------
# 11. Calculate mean absolute SHAP importance
# ------------------------------------------------------------

mean_abs_shap = np.abs(shap_values_for_class_1).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": mean_abs_shap
})

importance_df = importance_df.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
).reset_index(drop=True)


# ------------------------------------------------------------
# 12. Save SHAP importance CSV
# ------------------------------------------------------------

os.makedirs("models", exist_ok=True)

importance_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(f"\nSHAP importance saved to:")
print(OUTPUT_CSV)


# ------------------------------------------------------------
# 13. Print top 20 features
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 20 FEATURES BY SHAP IMPORTANCE")
print("=" * 70)

print(
    importance_df.head(20).to_string(index=False)
)


# ------------------------------------------------------------
# 14. Generate SHAP summary plot
# ------------------------------------------------------------

print("\n[7] Generating SHAP summary plot...")

plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values_for_class_1,
    X_sample,
    feature_names=feature_names,
    max_display=20,
    show=False
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"SHAP summary plot saved to:")
print(OUTPUT_PNG)


# ------------------------------------------------------------
# 15. Final status
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SHAP EXPLAINABILITY COMPLETED")
print("=" * 70)

print(f"CSV : {OUTPUT_CSV}")
print(f"PNG : {OUTPUT_PNG}")