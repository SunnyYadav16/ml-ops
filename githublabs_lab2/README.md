# GitHub Actions Lab 2 — Customer Churn CI/CD Pipeline

> **MLOps Course — Northeastern University (Prof. Ramin Mohammadi)**
> Differentiated version using **IBM Telco Customer Churn** dataset, **XGBoost**, a **3-job GitHub Actions workflow**, and a **FastAPI inference endpoint**.

---

## What This Lab Demonstrates

| Concept | Implementation |
|---|---|
| Model training in CI | XGBoost trained on real Telco Churn data on every push |
| Multi-job pipeline | `train → (test ∥ evaluate)` with `needs:` dependencies |
| Artifact passing | Model + test split uploaded from `train`, downloaded by `test` and `evaluate` |
| Quality gate | `evaluate` job exits with code 1 if F1 < 0.60, blocking the commit |
| Model versioning | Each model saved as `model_<timestamp>.joblib` |
| Unit testing | pytest suite with schema, preprocessing, artifact, and API tests |
| Inference endpoint | FastAPI app with `/predict`, `/batch_predict`, `/health` |

---

## Differentiation vs. Original Lab

| | **Original Lab** | **This Version** |
|---|---|---|
| Dataset | `make_classification` synthetic | IBM Telco Customer Churn (real) |
| Model | Random Forest | XGBoost with preprocessing pipeline |
| Workflow jobs | 1 job (flat steps) | 3 jobs: train → test ∥ evaluate |
| Quality gate | None | F1 ≥ 0.60 required to commit |
| Test coverage | Empty `test/` folder | 17 pytest tests across 4 test classes |
| Inference | None | FastAPI with single + batch predict |
| Metrics | F1 only | Accuracy, F1, Precision, Recall, ROC-AUC |

---

## Project Structure

```
githublabs_lab2_sunny/
├── .github/
│   └── workflows/
│       └── churn_cicd.yml        # 3-job CI/CD workflow
├── src/
│   ├── train_model.py            # Job 1: Train XGBoost + save artifacts
│   ├── evaluate_model.py         # Job 3: Evaluate + enforce F1 gate
│   └── app.py                    # FastAPI inference endpoint
├── test/
│   └── test_pipeline.py          # pytest: schema, preprocessing, model, API
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # ← add this file (see below)
├── models/                       # populated by CI
├── metrics/                      # populated by CI
└── requirements.txt
```

---

## Dataset Setup

Download the IBM Telco Customer Churn dataset and place it at:

```
githublabs_lab2_sunny/data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

**Source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset has **7,043 rows** and **21 columns** (tenure, contract type, monthly charges, etc.) with a binary churn label.

---

## GitHub Actions Workflow

The workflow file is at `.github/workflows/churn_cicd.yml`.

> **Important:** The workflow file must live at `.github/workflows/churn_cicd.yml` in the **root** of your repository, not inside `githublabs_lab2_sunny/`. GitHub Actions only reads workflows from `.github/workflows/` at the repo root.

### Job Dependency Graph

```
push to main
     │
     ▼
┌─────────┐
│  train  │  ← fits XGBoost, uploads model + test_split.parquet
└────┬────┘
     │  needs: train
     ├──────────────────────────┐
     ▼                          ▼
┌─────────┐              ┌──────────┐
│  test   │              │ evaluate │  ← F1 gate: fails if F1 < 0.60
│ pytest  │              │          │  ← commits metrics + model if gate passes
└─────────┘              └──────────┘
```

`test` and `evaluate` run **in parallel** after `train` completes.

### Triggering the Workflow

```bash
git add .
git commit -m "feat: add telco churn model"
git push origin main
```

---

## Running Locally

### 1. Install dependencies

```bash
cd githublabs_lab2_sunny
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train_model.py \
  --timestamp $(date '+%Y%m%d%H%M%S') \
  --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 3. Evaluate the model

```bash
python src/evaluate_model.py --timestamp <same_timestamp_as_above>
```

### 4. Run tests

```bash
pytest test/test_pipeline.py -v
```

### 5. Start the FastAPI server

```bash
uvicorn src.app:app --reload --port 8000
```

Then open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Reference

### `GET /health`
Returns model version and readiness.

```json
{
  "status": "ok",
  "model_version": "model_20240315120000",
  "model_loaded": true
}
```

### `POST /predict`
Accepts a single customer record, returns churn probability.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 6,
    "Contract": "Month-to-month",
    "MonthlyCharges": 70.0,
    "TotalCharges": 420.0,
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "PaperlessBilling": "Yes"
  }'
```

```json
{
  "churn_probability": 0.7812,
  "churn_prediction": 1,
  "model_version": "model_20240315120000"
}
```

### `POST /batch_predict`
Accepts a list of records, returns predictions for all.

---

## Quality Gate

The `evaluate` job enforces a minimum F1 score. If the model doesn't meet the threshold:

- The job exits with code `1`
- The workflow is marked **failed** (red ✗)
- The model is **not committed** to the repository

To change the threshold, edit `F1_THRESHOLD` in `src/evaluate_model.py`:

```python
F1_THRESHOLD = 0.60  # adjust as needed
```

---

## Acknowledgements

- Dataset: IBM Telco Customer Churn via Kaggle
- Course: MLOps — Northeastern University (Prof. Ramin Mohammadi)
- Original lab structure: [raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps)
