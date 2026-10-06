from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import pandas as pd
import numpy as np
import joblib
import shap
import os


# --------------------------------------------------
# APP
# --------------------------------------------------

app = FastAPI(
    title="SupplyPrescript API",
    description="AI-powered supply chain late-delivery risk prediction API",
    version="1.0.0"
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = "models/supplyprescript_xgboost.joblib"


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

pipeline = joblib.load(MODEL_PATH)

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]


# --------------------------------------------------
# FEATURE INFORMATION
# --------------------------------------------------

numeric_features = preprocessor.transformers_[0][2]

categorical_features = preprocessor.transformers_[1][2]

categorical_transformer = (
    preprocessor.named_transformers_["categorical"]
)

encoder = categorical_transformer.named_steps["encoder"]

encoded_feature_names = (
    encoder.get_feature_names_out(
        categorical_features
    )
)

feature_names = (
    list(numeric_features)
    + list(encoded_feature_names)
)


# --------------------------------------------------
# SHAP EXPLAINER
# --------------------------------------------------

explainer = shap.TreeExplainer(model)


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "application": "SupplyPrescript",
        "status": "running",
        "version": "1.0.0",
        "service": "Late Delivery Risk Prediction API"
    }


# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "numeric_features": len(numeric_features),
        "categorical_features": len(categorical_features),
        "transformed_features": len(feature_names)
    }


# --------------------------------------------------
# FEATURE SCHEMA
# --------------------------------------------------

@app.get("/features")
def features():

    return {
        "numeric_features": list(
            numeric_features
        ),
        "categorical_features": list(
            categorical_features
        ),
        "total_input_features": (
            len(numeric_features)
            + len(categorical_features)
        )
    }


# --------------------------------------------------
# REAL-TIME PREDICTION
# --------------------------------------------------

@app.post("/predict")
def predict(order: Dict[str, Any]):

    # ----------------------------------------------
    # VALIDATE INPUT
    # ----------------------------------------------

    required_features = (
        list(numeric_features)
        + list(categorical_features)
    )

    missing_features = [
        feature
        for feature in required_features
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
    # CREATE DATAFRAME
    # ----------------------------------------------

    input_data = {
        feature: order[feature]
        for feature in required_features
    }

    X = pd.DataFrame(
        [input_data]
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

    if probability >= 0.75:

        risk_level = "HIGH"

    elif probability >= 0.50:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # ----------------------------------------------
    # PRIORITY + ACTION
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

    elif risk_level == "MEDIUM":

        priority = "MEDIUM"

        action = (
            "Monitor shipment closely and verify "
            "carrier schedule before delay occurs."
        )

    else:

        priority = "LOW"

        action = (
            "Continue standard shipment monitoring."
        )


    # ----------------------------------------------
    # TRANSFORM FOR SHAP
    # ----------------------------------------------

    X_transformed = preprocessor.transform(X)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()


    # ----------------------------------------------
    # SHAP EXPLANATION
    # ----------------------------------------------

    shap_values = explainer.shap_values(
        X_transformed
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.asarray(
        shap_values
    )

    row_values = shap_values[0]

    top_indices = np.argsort(
        np.abs(row_values)
    )[::-1][:5]


    # ----------------------------------------------
    # BUILD EXPLANATIONS
    # ----------------------------------------------

    top_factors = []

    for index in top_indices:

        contribution = float(
            row_values[index]
        )

        feature = feature_names[index]

        if contribution > 0:

            direction = "increases risk"

        else:

            direction = "reduces risk"

        top_factors.append({
            "feature": feature,
            "contribution": round(
                contribution,
                4
            ),
            "direction": direction
        })


    # ----------------------------------------------
    # RESPONSE
    # ----------------------------------------------

    return {

        "prediction": {
            "late_delivery": prediction,
            "risk_probability": round(
                probability,
                6
            ),
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