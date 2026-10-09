
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ==================================================
# PATHS AND CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "supplyprescript_xgboost.joblib"

TEST_DATA_PATH = (
    BASE_DIR / "data" / "model_grouped" / "test.csv"
)


# ==================================================
# LOAD TRAINED MODEL PIPELINE
# ==================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

pipeline = joblib.load(MODEL_PATH)

# The complete pipeline contains preprocessing and XGBoost.
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]


# ==================================================
# FEATURE SCHEMA
# ==================================================

NUMERIC_FEATURES = list(
    preprocessor.transformers_[0][2]
)

CATEGORICAL_FEATURES = list(
    preprocessor.transformers_[1][2]
)

REQUIRED_FEATURES = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
)

# Keep these names for the existing API endpoints.
numeric_features = NUMERIC_FEATURES
categorical_features = CATEGORICAL_FEATURES


# ==================================================
# ENCODED FEATURE NAMES
# ==================================================

categorical_transformer = (
    preprocessor.named_transformers_["categorical"]
)

encoder = categorical_transformer.named_steps["encoder"]

encoded_feature_names = encoder.get_feature_names_out(
    CATEGORICAL_FEATURES
)

feature_names = (
    NUMERIC_FEATURES + list(encoded_feature_names)
)


# ==================================================
# SHAP EXPLAINER
# ==================================================

explainer = shap.TreeExplainer(model)


# ==================================================
# FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="SupplyPrescript API",
    description=(
        "AI-powered supply chain late-delivery "
        "risk prediction and prescription API."
    ),
    version="1.0.0"
)


# ==================================================
# CORS CONFIGURATION
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# HELPER: RISK CLASSIFICATION
# ==================================================

def classify_risk(probability: float) -> str:
    if probability >= 0.75:
        return "HIGH"

    if probability >= 0.50:
        return "MEDIUM"

    return "LOW"


# ==================================================
# HELPER: PRIORITY AND RECOMMENDED ACTION
# ==================================================

def get_prescription(risk_level: str, probability: float):
    if risk_level == "HIGH" and probability >= 0.90:
        return (
            "CRITICAL",
            "Escalate the order immediately to the "
            "logistics team and prioritize shipment monitoring."
        )

    if risk_level == "HIGH":
        return (
            "HIGH",
            "Prioritize shipment and closely monitor carrier status."
        )

    if risk_level == "MEDIUM":
        return (
            "MEDIUM",
            "Monitor shipment closely and verify the "
            "carrier schedule before a delay occurs."
        )

    return (
        "LOW",
        "Continue standard shipment monitoring."
    )


# ==================================================
# ROOT ENDPOINT
# ==================================================

@app.get("/")
def root():
    return {
        "application": "SupplyPrescript",
        "status": "running",
        "version": "1.0.0",
        "service": "Late Delivery Risk Prediction API"
    }


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "numeric_features": len(NUMERIC_FEATURES),
        "categorical_features": len(CATEGORICAL_FEATURES),
        "total_input_features": len(REQUIRED_FEATURES),
        "transformed_features": len(feature_names)
    }


# ==================================================
# FEATURE INFORMATION
# ==================================================

@app.get("/features")
def get_features():
    return {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "total_input_features": len(REQUIRED_FEATURES)
    }


# ==================================================
# SAMPLE ORDER ENDPOINT
# ==================================================

@app.get("/sample-order")
def get_sample_order():
    """
    Return a real order from the grouped test dataset.

    Missing feature values are returned as null.
    The trained preprocessing pipeline handles imputation.
    """

    try:
        if not TEST_DATA_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Test dataset not found: {TEST_DATA_PATH}"
            )

        df = pd.read_csv(TEST_DATA_PATH)

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="The test dataset is empty."
            )

        missing_columns = [
            feature
            for feature in REQUIRED_FEATURES
            if feature not in df.columns
        ]

        if missing_columns:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Test dataset is missing required features.",
                    "missing_features": missing_columns
                }
            )

        # Select a real, reproducible sample.
        sample = df.iloc[0]

        order = {}

        for feature in REQUIRED_FEATURES:
            value = sample[feature]

            if pd.isna(value):
                order[feature] = None

            elif feature in NUMERIC_FEATURES:
                order[feature] = float(value)

            else:
                order[feature] = str(value)

        return {
            "success": True,
            "message": "Sample order loaded successfully.",
            "order": order
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not load sample order: "
                f"{type(error).__name__}: {str(error)}"
            )
        )


# ==================================================
# REAL-TIME PREDICTION ENDPOINT
# ==================================================

@app.post("/predict")
def predict(order: Dict[str, Any]):

    # ----------------------------------------------
    # VALIDATE REQUIRED FEATURES
    # ----------------------------------------------

    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in order
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required features.",
                "missing_features": missing_features
            }
        )

    # ----------------------------------------------
    # VALIDATE AND PREPARE INPUT
    # ----------------------------------------------

    input_data = {}

    for feature in REQUIRED_FEATURES:
        value = order[feature]

        if value is not None and isinstance(
            value, (dict, list, tuple)
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value for feature: {feature}"
            )

        if feature in NUMERIC_FEATURES:
            if value is None or value == "":
                input_data[feature] = np.nan
            else:
                try:
                    input_data[feature] = float(value)
                except (TypeError, ValueError):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Feature '{feature}' must be numeric."
                        )
                    )
        else:
            if value is None or value == "":
                input_data[feature] = np.nan
            else:
                input_data[feature] = str(value)

    X = pd.DataFrame(
        [input_data],
        columns=REQUIRED_FEATURES
    )

    # ----------------------------------------------
    # PREDICTION
    # ----------------------------------------------

    try:
        probability = float(
            pipeline.predict_proba(X)[0][1]
        )

        prediction = int(
            pipeline.predict(X)[0]
        )

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unable to process input data.",
                "error": str(error)
            }
        )

    # ----------------------------------------------
    # RISK LEVEL
    # ----------------------------------------------

    risk_level = classify_risk(probability)

    # ----------------------------------------------
    # PRIORITY AND ACTION
    # ----------------------------------------------

    priority, action = get_prescription(
        risk_level,
        probability
    )

    # ----------------------------------------------
    # TRANSFORM INPUT FOR SHAP
    # ----------------------------------------------

    try:
        X_transformed = preprocessor.transform(X)

        # SHAP explanation for this single order.
        shap_values = explainer.shap_values(
            X_transformed
        )

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = np.asarray(shap_values)

        # Support binary-classifier SHAP output formats.
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]

        row_values = shap_values[0]

        top_indices = np.argsort(
            np.abs(row_values)
        )[::-1][:5]

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Prediction succeeded, but explanation failed.",
                "error": str(error)
            }
        )

    # ----------------------------------------------
    # BUILD TOP FACTORS
    # ----------------------------------------------

    top_factors = []

    for index in top_indices:
        contribution = float(row_values[index])

        feature = feature_names[index]

        direction = (
            "increases risk"
            if contribution > 0
            else "reduces risk"
        )

        top_factors.append({
            "feature": feature,
            "contribution": round(contribution, 4),
            "direction": direction
        })

    # ----------------------------------------------
    # API RESPONSE
    # ----------------------------------------------

    return {
        "prediction": {
            "late_delivery": prediction,
            "risk_probability": round(probability, 6),
            "risk_level": risk_level
        },
        "decision": {
            "priority": priority,
            "recommended_action": action
        },
        "explanation": {
            "top_factors": top_factors
        }
    }