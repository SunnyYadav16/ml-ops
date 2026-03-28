"""test_pipeline.py — Unit + integration tests run by Job 2 in GitHub Actions.

Differentiation from original lab (which has an empty test/ folder):
  - Tests data schema integrity
  - Tests preprocessing logic
  - Tests model artifact presence after training
  - Tests FastAPI endpoint responses using TestClient
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath("."))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_ROW = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 1685.65,
}

EXPECTED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


@pytest.fixture
def sample_df():
    return pd.DataFrame([SAMPLE_ROW])


@pytest.fixture
def multi_row_df():
    rows = [SAMPLE_ROW.copy() for _ in range(5)]
    rows[1]["Contract"] = "One year"
    rows[2]["tenure"] = 60
    rows[3]["MonthlyCharges"] = 20.0
    rows[4]["InternetService"] = "DSL"
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data schema tests
# ---------------------------------------------------------------------------

class TestDataSchema:
    def test_sample_row_has_all_expected_columns(self, sample_df):
        assert list(sample_df.columns) == EXPECTED_COLUMNS

    def test_numeric_columns_are_numeric(self, sample_df):
        for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
            assert pd.api.types.is_numeric_dtype(sample_df[col]), \
                f"{col} should be numeric"

    def test_no_missing_values_in_sample(self, sample_df):
        assert sample_df.isnull().sum().sum() == 0

    def test_total_charges_non_negative(self, sample_df):
        assert (sample_df["TotalCharges"] >= 0).all()

    def test_monthly_charges_non_negative(self, sample_df):
        assert (sample_df["MonthlyCharges"] >= 0).all()

    def test_tenure_non_negative(self, sample_df):
        assert (sample_df["tenure"] >= 0).all()

    def test_senior_citizen_binary(self, sample_df):
        assert sample_df["SeniorCitizen"].isin([0, 1]).all()

    def test_gender_valid_values(self, sample_df):
        assert sample_df["gender"].isin(["Male", "Female"]).all()

    def test_contract_valid_values(self, multi_row_df):
        valid = {"Month-to-month", "One year", "Two year"}
        assert set(multi_row_df["Contract"].unique()).issubset(valid)


# ---------------------------------------------------------------------------
# Preprocessing logic tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_total_charges_coercion(self):
        """TotalCharges with blank string should be coerced to NaN and dropped."""
        df = pd.DataFrame([{**SAMPLE_ROW, "TotalCharges": " "}])
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        assert df["TotalCharges"].isna().any()

    def test_churn_label_encoding(self):
        df = pd.DataFrame({"Churn": ["Yes", "No", "Yes", "No"]})
        labels = (df["Churn"].str.strip() == "Yes").astype(int)
        assert list(labels) == [1, 0, 1, 0]

    def test_feature_count(self, sample_df):
        assert len(sample_df.columns) == 19, \
            f"Expected 19 feature columns, got {len(sample_df.columns)}"

    def test_multi_row_no_missing(self, multi_row_df):
        assert multi_row_df.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Model artifact tests (run after train job uploads artifact)
# ---------------------------------------------------------------------------

class TestModelArtifact:
    def test_models_directory_exists(self):
        assert os.path.isdir("models"), \
            "models/ directory missing — train job may not have run"

    def test_at_least_one_model_file(self):
        import glob
        model_files = glob.glob("models/model_*.joblib")
        assert len(model_files) >= 1, \
            "No model_*.joblib files found in models/"

    def test_feature_schema_exists(self):
        assert os.path.exists("models/feature_schema.json"), \
            "feature_schema.json missing from models/"

    def test_feature_schema_has_required_keys(self):
        with open("models/feature_schema.json") as f:
            schema = json.load(f)
        for key in ["feature_columns", "cat_cols", "num_cols", "model_version"]:
            assert key in schema, f"Missing key '{key}' in feature_schema.json"

    def test_feature_schema_column_count(self):
        with open("models/feature_schema.json") as f:
            schema = json.load(f)
        assert len(schema["feature_columns"]) == 19

    def test_model_loads_and_predicts(self):
        import glob
        import joblib
        model_files = sorted(glob.glob("models/model_*.joblib"))
        pipeline = joblib.load(model_files[-1])
        df = pd.DataFrame([SAMPLE_ROW])
        proba = pipeline.predict_proba(df)
        assert proba.shape == (1, 2), "Expected shape (1, 2) for binary classification"
        assert 0.0 <= proba[0, 1] <= 1.0, "Probability must be in [0, 1]"

    def test_model_output_is_binary(self):
        import glob
        import joblib
        model_files = sorted(glob.glob("models/model_*.joblib"))
        pipeline = joblib.load(model_files[-1])
        df = pd.DataFrame([SAMPLE_ROW] * 10)
        preds = pipeline.predict(df)
        assert set(preds).issubset({0, 1}), "Predictions must be 0 or 1"


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------

class TestAPIEndpoints:
    @pytest.fixture
    def client(self):
        import glob
        import joblib
        from fastapi.testclient import TestClient
        from src import app as app_module

        model_files = sorted(glob.glob("models/model_*.joblib"))
        if not model_files:
            pytest.skip("No model available — skipping API tests")

        # Inject the model directly into the app module before TestClient starts
        app_module._pipeline = joblib.load(model_files[-1])
        app_module._model_version = os.path.basename(model_files[-1]).replace(".joblib", "")

        return TestClient(app_module.app)

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_predict_returns_probability(self, client):
        response = client.post("/predict", json=SAMPLE_ROW)
        assert response.status_code == 200
        data = response.json()
        assert "churn_probability" in data
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_predict_churn_prediction_is_binary(self, client):
        response = client.post("/predict", json=SAMPLE_ROW)
        data = response.json()
        assert data["churn_prediction"] in [0, 1]

    def test_predict_includes_model_version(self, client):
        response = client.post("/predict", json=SAMPLE_ROW)
        data = response.json()
        assert "model_version" in data
        assert data["model_version"] != ""

    def test_batch_predict_returns_correct_count(self, client):
        payload = [SAMPLE_ROW, SAMPLE_ROW, SAMPLE_ROW]
        response = client.post("/batch_predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
        assert len(data["predictions"]) == 3

    def test_batch_predict_empty_list_returns_400(self, client):
        response = client.post("/batch_predict", json=[])
        assert response.status_code == 400
