from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
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
TEST_PATH = "data/model_grouped/test.csv"
PRESCRIPTION_PATH = "models/smart_prescriptions.csv"


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        f"Test data not found: {TEST_PATH}"
    )

test_df = pd.read_csv(TEST_PATH)


# --------------------------------------------------
# LOAD PRESCRIPTIONS
# --------------------------------------------------

prescriptions_df = None

if os.path.exists(PRESCRIPTION_PATH):
    prescriptions_df = pd.read_csv(
        PRESCRIPTION_PATH
    )


# --------------------------------------------------
# ROOT ENDPOINT
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
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "test_records": len(test_df)
    }


# --------------------------------------------------
# SAMPLE PREDICTION
# --------------------------------------------------

@app.get("/predict/{order_index}")
def predict(order_index: int):

    if order_index < 0:
        raise HTTPException(
            status_code=400,
            detail="Order index must be non-negative."
        )

    if order_index >= len(test_df):
        raise HTTPException(
            status_code=404,
            detail="Order index not found."
        )


    # ----------------------------------------------
    # GET ORDER
    # ----------------------------------------------

    row = test_df.iloc[
        order_index
    ]

    target = "Late_delivery_risk"

    X = pd.DataFrame(
        [row.drop(labels=[target])]
    )


    # ----------------------------------------------
    # PREDICTION
    # ----------------------------------------------

    probability = float(
        model.predict_proba(X)[0][1]
    )

    prediction = int(
        model.predict(X)[0]
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
    # DEFAULT PRESCRIPTION
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
    # RESPONSE
    # ----------------------------------------------

    return {
        "order_index": order_index,
        "risk_probability": round(
            probability,
            6
        ),
        "predicted_late_delivery": prediction,
        "risk_level": risk_level,
        "priority": priority,
        "recommended_action": action
    }