"""evaluate_model.py

Job 3 — Evaluate the trained XGBoost pipeline and enforce an F1 quality gate.

Differentiation from original lab:
  - Uses the real held-out test split saved by the train job (not synthetic data)
  - Evaluates accuracy, precision, recall, F1, and ROC-AUC (not just F1)
  - Hard quality gate: exits with code 1 if F1 < F1_THRESHOLD, failing the CI run
  - Saves a rich metrics JSON with all scores + timestamp + gate result
"""

import argparse
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.abspath(".."))

# CI will fail if F1 on the held-out test set drops below this value
F1_THRESHOLD = 0.60


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()

    print(f"[evaluate] timestamp={args.timestamp}")

    # Load model saved by the train job
    model_path = f"models/model_{args.timestamp}.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    pipeline = joblib.load(model_path)
    print(f"[evaluate] loaded model from {model_path}")

    # Load the held-out test split saved by the train job
    test_data_path = "data/test_split.parquet"
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test split not found: {test_data_path}")
    df_test = pd.read_parquet(test_data_path)
    y_true = df_test["label"].astype(int)
    X_test = df_test.drop(columns=["label"])
    print(f"[evaluate] test set size: {len(X_test)} rows")

    # Predict
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    f1 = round(f1_score(y_true, y_pred), 4)
    gate_passed = bool(f1 >= F1_THRESHOLD)

    metrics = {
        "model_version": args.timestamp,
        "test_set_size": len(X_test),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_score": f1,
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "f1_threshold": F1_THRESHOLD,
        "gate_passed": gate_passed,
    }

    print(f"[evaluate] metrics:\n{json.dumps(metrics, indent=2)}")

    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    metrics_path = f"metrics/{args.timestamp}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] metrics saved → {metrics_path}")

    # Enforce quality gate
    if not gate_passed:
        print(
            f"\n[evaluate] GATE FAILED: F1={f1} < threshold={F1_THRESHOLD}"
        )
        sys.exit(1)

    print(f"\n[evaluate] GATE PASSED: F1={f1} >= threshold={F1_THRESHOLD}")
