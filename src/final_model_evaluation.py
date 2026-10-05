import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

import joblib


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = "models/supplyprescript_xgboost.joblib"
TEST_PATH = "data/model_grouped/test.csv"
OUTPUT_DIR = "models"

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

print("\nLoading grouped test dataset...")

test_df = pd.read_csv(TEST_PATH)

TARGET = "Late_delivery_risk"

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print(f"Test dataset shape: {test_df.shape}")
print(f"Features shape: {X_test.shape}")


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

print("\nGenerating predictions...")

y_pred = pipeline.predict(X_test)
y_probability = pipeline.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)
average_precision = average_precision_score(y_test, y_probability)


print("\n" + "=" * 60)
print("FINAL XGBOOST MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy           : {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"F1 Score           : {f1:.4f}")
print(f"ROC-AUC            : {roc_auc:.4f}")
print(f"Average Precision  : {average_precision:.4f}")


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks([0, 1], ["On Time", "Late"])
plt.yticks([0, 1], ["On Time", "Late"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=14
        )

plt.colorbar()
plt.tight_layout()

confusion_path = os.path.join(
    OUTPUT_DIR,
    "xgboost_confusion_matrix.png"
)

plt.savefig(confusion_path, dpi=300)
plt.close()

print(f"\nConfusion matrix saved to: {confusion_path}")


# --------------------------------------------------
# ROC CURVE
# --------------------------------------------------

fpr, tpr, _ = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("XGBoost ROC Curve")
plt.legend()
plt.grid(alpha=0.3)

roc_path = os.path.join(
    OUTPUT_DIR,
    "xgboost_roc_curve.png"
)

plt.savefig(roc_path, dpi=300)
plt.close()

print(f"ROC curve saved to: {roc_path}")


# --------------------------------------------------
# PRECISION-RECALL CURVE
# --------------------------------------------------

precision_values, recall_values, _ = precision_recall_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall_values,
    precision_values,
    label=f"XGBoost (AP = {average_precision:.4f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("XGBoost Precision-Recall Curve")
plt.legend()
plt.grid(alpha=0.3)

pr_path = os.path.join(
    OUTPUT_DIR,
    "xgboost_precision_recall_curve.png"
)

plt.savefig(pr_path, dpi=300)
plt.close()

print(
    f"Precision-Recall curve saved to: {pr_path}"
)


# --------------------------------------------------
# SAVE METRICS
# --------------------------------------------------

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Average Precision"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        average_precision
    ]
})

metrics_path = os.path.join(
    OUTPUT_DIR,
    "xgboost_final_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)

print(
    f"Final metrics saved to: {metrics_path}"
)


# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION COMPLETED")
print("=" * 60)