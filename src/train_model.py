import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SUPPLYPRESCRIPT - BASELINE MODEL TRAINING
# ============================================================

print("=" * 60)
print("SUPPLYPRESCRIPT BASELINE MODEL TRAINING")
print("=" * 60)


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

TRAIN_PATH = "data/model/train.csv"
TEST_PATH = "data/model/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\nDatasets loaded successfully.")

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows: {len(test_df):,}")


# ------------------------------------------------------------
# 2. DEFINE TARGET
# ------------------------------------------------------------

TARGET = "Late_delivery_risk"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


print(f"\nTarget column: {TARGET}")
print(f"Number of features: {X_train.shape[1]}")


# ------------------------------------------------------------
# 3. IDENTIFY FEATURE TYPES
# ------------------------------------------------------------

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nFeature types:")
print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")


# ------------------------------------------------------------
# 4. NUMERIC PIPELINE
# ------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ------------------------------------------------------------
# 5. CATEGORICAL PIPELINE
# ------------------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse=True
            )
        )
    ]
)


# ------------------------------------------------------------
# 6. COMBINE PREPROCESSING
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ------------------------------------------------------------
# 7. CREATE BASELINE MODEL
# ------------------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)


# ------------------------------------------------------------
# 8. COMPLETE PIPELINE
# ------------------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ------------------------------------------------------------
# 9. TRAIN MODEL
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

pipeline.fit(X_train, y_train)

print("\nModel training completed successfully.")


# ------------------------------------------------------------
# 10. PREDICTIONS
# ------------------------------------------------------------

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# 11. EVALUATION
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ------------------------------------------------------------
# 12. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Late Delivery Risk",
            "Late Delivery Risk"
        ],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 13. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\n                Predicted")
print("              0         1")
print(f"Actual 0   {cm[0][0]:6d}   {cm[0][1]:6d}")
print(f"Actual 1   {cm[1][0]:6d}   {cm[1][1]:6d}")


# ------------------------------------------------------------
# 14. SAVE MODEL
# ------------------------------------------------------------

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "supplyprescript_logistic_regression.joblib"
)

joblib.dump(pipeline, MODEL_PATH)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(f"\nModel path: {MODEL_PATH}")

print("\nBaseline model training completed successfully.")