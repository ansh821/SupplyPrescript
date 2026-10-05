import os
import pandas as pd
import numpy as np
import joblib
import shap


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = "models/supplyprescript_xgboost.joblib"
TEST_PATH = "data/model_grouped/test.csv"

OUTPUT_DIR = "models"
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "order_level_explanations.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading XGBoost model...")

pipeline = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

print("\nLoading grouped test data...")

test_df = pd.read_csv(TEST_PATH)

TARGET = "Late_delivery_risk"

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print(f"Test dataset shape: {test_df.shape}")
print(f"Feature shape: {X_test.shape}")


# --------------------------------------------------
# GET PIPELINE COMPONENTS
# --------------------------------------------------

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]


# --------------------------------------------------
# GET TRANSFORMER INFORMATION
# --------------------------------------------------

numeric_transformer = preprocessor.named_transformers_["numeric"]
categorical_transformer = preprocessor.named_transformers_["categorical"]

numeric_features = (
    preprocessor
    .transformers_[0][2]
)

categorical_features = (
    preprocessor
    .transformers_[1][2]
)


# --------------------------------------------------
# GET ENCODER
# --------------------------------------------------

encoder = categorical_transformer.named_steps["encoder"]


# --------------------------------------------------
# TRANSFORM TEST DATA
# --------------------------------------------------

print("\nTransforming test data...")

X_transformed = preprocessor.transform(X_test)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

print(
    f"Transformed feature shape: "
    f"{X_transformed.shape}"
)


# --------------------------------------------------
# BUILD FEATURE NAMES
# --------------------------------------------------

feature_names = []

# Numeric features
feature_names.extend(
    numeric_features
)

# One-hot encoded categorical features
encoded_names = encoder.get_feature_names_out(
    categorical_features
)

feature_names.extend(
    encoded_names
)

print(
    f"Total feature names: "
    f"{len(feature_names)}"
)


# --------------------------------------------------
# VALIDATE FEATURE COUNT
# --------------------------------------------------

if len(feature_names) != X_transformed.shape[1]:

    raise ValueError(
        "Feature name count does not match "
        "transformed feature count."
    )


# --------------------------------------------------
# SHAP EXPLAINER
# --------------------------------------------------

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X_transformed
)


# --------------------------------------------------
# HANDLE SHAP OUTPUT
# --------------------------------------------------

if isinstance(shap_values, list):

    shap_values = shap_values[1]

shap_values = np.asarray(
    shap_values
)

print(
    f"SHAP matrix shape: "
    f"{shap_values.shape}"
)


# --------------------------------------------------
# MODEL PREDICTIONS
# --------------------------------------------------

print("\nGenerating predictions...")

predictions = pipeline.predict(
    X_test
)

probabilities = pipeline.predict_proba(
    X_test
)[:, 1]


# --------------------------------------------------
# RISK LEVEL FUNCTION
# --------------------------------------------------

def get_risk_level(probability):

    if probability >= 0.75:
        return "HIGH"

    elif probability >= 0.50:
        return "MEDIUM"

    else:
        return "LOW"


# --------------------------------------------------
# TOP FEATURE FUNCTION
# --------------------------------------------------

def get_top_features(row_index, top_n=5):

    values = shap_values[row_index]

    top_indices = np.argsort(
        np.abs(values)
    )[::-1][:top_n]

    explanations = []

    for index in top_indices:

        feature = feature_names[index]
        contribution = values[index]

        if contribution > 0:

            direction = "increases risk"

        else:

            direction = "reduces risk"

        explanations.append(
            f"{feature}: "
            f"{contribution:.4f} "
            f"({direction})"
        )

    return " | ".join(
        explanations
    )


# --------------------------------------------------
# GENERATE EXPLANATIONS
# --------------------------------------------------

print("\nGenerating order-level explanations...")

results = []

for i in range(len(X_test)):

    probability = probabilities[i]

    risk_level = get_risk_level(
        probability
    )

    explanation = get_top_features(
        i,
        top_n=5
    )

    results.append({

        "Order_Index": i,

        "Actual_Late_Delivery": int(
            y_test.iloc[i]
        ),

        "Predicted_Late_Delivery": int(
            predictions[i]
        ),

        "Risk_Probability": round(
            float(probability),
            6
        ),

        "Risk_Level": risk_level,

        "Top_Risk_Factors": explanation

    })


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

explanation_df = pd.DataFrame(
    results
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

explanation_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# DISPLAY SAMPLE
# --------------------------------------------------

print("\n" + "=" * 80)
print("ORDER-LEVEL EXPLANATION SAMPLE")
print("=" * 80)

print(
    explanation_df[
        [
            "Order_Index",
            "Risk_Probability",
            "Risk_Level",
            "Top_Risk_Factors"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print("\n" + "=" * 80)
print("ORDER-LEVEL EXPLANATION COMPLETED")
print("=" * 80)

print(
    f"\nSaved to: {OUTPUT_PATH}"
)

print(
    f"Total explanations generated: "
    f"{len(explanation_df)}"
)