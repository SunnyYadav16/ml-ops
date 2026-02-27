import json
import os
import warnings
from datetime import datetime

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_and_preprocess() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load Titanic from the bundled CSV and apply basic feature engineering.
    Returns (X, y) ready for modeling.
    """
    csv_path = "titanic.csv"
    if not os.path.exists(csv_path):
        print("      Downloading Titanic dataset...")
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        import urllib.request
        urllib.request.urlretrieve(url, csv_path)
    df = pd.read_csv(csv_path)

    # Target
    y = df["Survived"]

    # ── Feature engineering ──────────────────────────────────────────────
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)

    # Family size & solo traveler flag
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Encode categorical features
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone"]
    X = df[features]

    return X, y


def train_model(X_train, y_train) -> XGBClassifier:
    """Train an XGBoost classifier."""
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Run full evaluation and return metrics dict."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": {
            "true_negative": int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive": int(cm[1][1]),
        },
    }
    return metrics


def save_artifacts(model, metrics: dict, feature_names: list):
    """Persist model and metrics to OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save model
    model_path = os.path.join(OUTPUT_DIR, "titanic_xgb_model.pkl")
    joblib.dump(model, model_path)

    # Save metrics with metadata
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": "XGBClassifier",
        "dataset": "Titanic",
        "features": feature_names,
        "metrics": metrics,
    }
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    return model_path, metrics_path


if __name__ == "__main__":
    print("=" * 60)
    print("  Titanic Survival Prediction — XGBoost in Docker")
    print("=" * 60)

    # 1 ─ Load & preprocess
    print("\n[1/4] Loading and preprocessing Titanic dataset...")
    X, y = load_and_preprocess()
    print(f"      Dataset shape: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"      Survival rate: {y.mean():.2%}")

    # 2 ─ Split
    print("\n[2/4] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # 3 ─ Train
    print("\n[3/4] Training XGBoost classifier...")
    model = train_model(X_train, y_train)
    print("      Training complete.")

    # 4 ─ Evaluate
    print("\n[4/4] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    print(f"\n      Accuracy:  {metrics['accuracy']}")
    print(f"      Precision: {metrics['precision']}")
    print(f"      Recall:    {metrics['recall']}")
    print(f"      F1 Score:  {metrics['f1_score']}")
    print(f"      ROC AUC:   {metrics['roc_auc']}")
    cm = metrics["confusion_matrix"]
    print(f"\n      Confusion Matrix:")
    print(f"        TN={cm['true_negative']}  FP={cm['false_positive']}")
    print(f"        FN={cm['false_negative']}  TP={cm['true_positive']}")

    # 5 ─ Save
    model_path, metrics_path = save_artifacts(model, metrics, list(X.columns))
    print(f"\n  Model saved to:   {model_path}")
    print(f"  Metrics saved to: {metrics_path}")
    print("\n" + "=" * 60)
    print("  Done! Check your mounted volume for artifacts.")
    print("=" * 60)
