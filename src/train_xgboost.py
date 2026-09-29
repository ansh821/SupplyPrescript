import os
import joblib
import pandas as pd
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
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
# SUPPLYPRESCRIPT - XGBOOST MODEL TRAINING
# ============================================================

print("=" * 60)
print("SUPPLYPRESCRIPT XGBOOST TRAINING")
print("=" * 60)


# ------------------------------------------------------------
# 1. LOAD GROUPED DATA
# ------------------------------------------------------------

TRAIN_PATH = "data/model_grouped/train.csv"
TEST_PATH = "data/model_grouped/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\nDatasets loaded successfully.")
print(f"Training rows: {len(train_df):,}")
print(f"Testing rows : {len(test_df):,}")


# ------------------------------------------------------------
# 2. TARGET
# ------------------------------------------------------------

TARGET = "Late_delivery_risk"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print(f"\nTarget column: {TARGET}")
print(f"Number of features: {X_train.shape[1]}")


# ------------------------------------------------------------
# 3. FEATURE TYPES
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
# 4. PREPROCESSING
# ------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

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

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ------------------------------------------------------------
# 5. XGBOOST MODEL
# ------------------------------------------------------------

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=2
)


# ------------------------------------------------------------
# 6. COMPLETE PIPELINE
# ------------------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ------------------------------------------------------------
# 7. TRAIN
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)

pipeline.fit(
    X_train,
    y_train
)

print("\nXGBoost training completed successfully.")


# ------------------------------------------------------------
# 8. PREDICTIONS
# ------------------------------------------------------------

print("\nGenerating predictions...")

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# 9. EVALUATION
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

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


# ------------------------------------------------------------
# 10. DISPLAY PERFORMANCE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("XGBOOST MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ------------------------------------------------------------
# 11. CLASSIFICATION REPORT
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
# 12. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\n                Predicted")
print("              0         1")
print(f"Actual 0   {cm[0][0]:6d}   {cm[0][1]:6d}")
print(f"Actual 1   {cm[1][0]:6d}   {cm[1][1]:6d}")


# ------------------------------------------------------------
# 13. SAVE MODEL
# ------------------------------------------------------------

MODEL_DIR = "models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "supplyprescript_xgboost.joblib"
)

joblib.dump(
    pipeline,
    MODEL_PATH
)


# ------------------------------------------------------------
# 14. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("XGBOOST MODEL SAVED")
print("=" * 60)

print(f"\nModel path: {MODEL_PATH}")

print("\nXGBoost training completed successfully.")