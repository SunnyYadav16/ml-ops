"""train_model.py

Job 1 — Train an XGBoost classifier on the IBM Telco Customer Churn dataset.

Differentiation from original lab:
  - Real dataset (IBM Telco Churn CSV) instead of make_classification synthetic data
  - XGBoost instead of Random Forest / LightGBM
  - Proper sklearn ColumnTransformer preprocessing pipeline
  - Saves model artifact + feature schema for downstream jobs
  - Saves a held-out test split as parquet for the evaluate job
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath(".."))


def load_and_preprocess(data_path: str):
    """Load Telco Churn CSV and return feature matrix X and label vector y."""
    df = pd.read_csv(data_path)

    # Drop customer ID — not a feature
    df = df.drop(columns=["customerID"], errors="ignore")

    # TotalCharges is sometimes stored as string with blank spaces
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    y = (df["Churn"].str.strip() == "Yes").astype(int)
    X = df.drop(columns=["Churn"])

    return X, y


def build_pipeline(X: pd.DataFrame, scale_pos_weight: float = 1.0) -> Pipeline:
    """Build a ColumnTransformer + XGBoost pipeline."""
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
                cat_cols,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
        ]
    )
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timestamp", type=str, required=True,
        help="Version timestamp injected by GitHub Actions"
    )
    parser.add_argument(
        "--data", type=str,
        default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to Telco Churn CSV"
    )
    args = parser.parse_args()

    print(f"[train] timestamp={args.timestamp}")

    X, y = load_and_preprocess(args.data)
    print(f"[train] dataset shape: {X.shape}, churn rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compute class weight ratio from actual training labels
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = round(n_neg / n_pos, 3)
    print(f"[train] class balance — neg:{n_neg}, pos:{n_pos}, scale_pos_weight={scale_pos_weight}")

    pipeline = build_pipeline(X_train, scale_pos_weight=scale_pos_weight)
    pipeline.fit(X_train, y_train)
    print("[train] model fitted successfully")

    # --- Persist model ---
    os.makedirs("models", exist_ok=True)
    model_path = f"models/model_{args.timestamp}.joblib"
    joblib.dump(pipeline, model_path)
    print(f"[train] model saved → {model_path}")

    # --- Persist feature schema (used by API and tests) ---
    schema = {
        "feature_columns": X.columns.tolist(),
        "cat_cols": X.select_dtypes(include=["object"]).columns.tolist(),
        "num_cols": X.select_dtypes(include=[np.number]).columns.tolist(),
        "model_version": args.timestamp,
    }
    with open("models/feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    print("[train] feature schema saved → models/feature_schema.json")

    # --- Save held-out test split for the evaluate job ---
    os.makedirs("data", exist_ok=True)
    X_test_save = X_test.copy()
    X_test_save["label"] = y_test.values
    X_test_save.to_parquet("data/test_split.parquet", index=False)
    print(f"[train] test split saved → data/test_split.parquet  ({len(X_test)} rows)")
