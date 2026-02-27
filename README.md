# MLOps Labs

Hands-on lab projects from [Prof. Ramin Mohammadi's MLOps course](https://www.mlwithramin.com/) at Northeastern University. Each lab is a self-contained project covering a different stage of the ML engineering lifecycle — from model serving to pipeline orchestration.

---

## Project Structure

```
MlOps/
├── fastapi_lab/       # Lab 1 — ML model serving with FastAPI
├── airflow_lab2/      # Lab 2 — ML pipeline orchestration with Airflow 3.x
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Lab 1: FastAPI — California Housing Price Prediction API

**Directory:** [`fastapi_lab/`](./fastapi_lab/)

Trains a **Random Forest Regressor** on the California Housing dataset and serves predictions through a FastAPI REST API with Pydantic request validation.

| Aspect | Details |
|---|---|
| Model | Random Forest Regressor (scikit-learn) |
| Dataset | California Housing — 8 features, median house value target |
| Serving | FastAPI with Pydantic schema validation |
| Key Files | `src/main.py` (API), `src/train_model.py` (training), `src/predict.py` (inference), `src/data.py` (data loading) |

**Run it:**

```bash
cd fastapi_lab
pip install -r requirements.txt
python src/train_model.py       # Train and save the model
uvicorn src.main:app --reload   # Start the API at localhost:8000
```

---

## Lab 2: Airflow — ML Pipeline Orchestration with Email Notifications & FastAPI Monitoring

**Directory:** [`airflow_lab2/`](./airflow_lab2/)

An end-to-end ML pipeline running inside **Docker Compose** with **Apache Airflow 3.0.2**. The pipeline trains a Logistic Regression classifier on the Breast Cancer Wisconsin dataset, sends email notifications via Gmail SMTP on completion, and exposes a FastAPI monitoring dashboard that queries the Airflow REST API using JWT authentication.

This is a modified version of the [original Airflow Lab 2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Airflow_Labs/Lab_2). Key changes from the original: Docker Compose replaces bare `pip install` (which causes SIGSEGV on macOS), Airflow 3.x replaces 2.x (with FAB auth + JWT), and FastAPI replaces Flask for the monitoring API.

| Aspect | Details |
|---|---|
| Model | Logistic Regression (scikit-learn) |
| Dataset | Breast Cancer Wisconsin — 569 samples, 30 features, binary classification |
| Orchestration | Apache Airflow 3.0.2 with 7-task DAG (load → preprocess → split → train → evaluate → email) |
| Monitoring | FastAPI dashboard with JWT-authenticated Airflow REST API queries |
| Notifications | Gmail SMTP via Airflow SmtpOperator |
| Infrastructure | Docker Compose — PostgreSQL, Airflow (API server, scheduler, DAG processor, triggerer), FastAPI |

**Run it:**

```bash
cd airflow_lab2
cp .env.example .env            # Edit with your credentials
mkdir -p logs plugins config
docker compose build
docker compose up airflow-init  # Exits with code 0 — that's expected
docker compose up -d
```

| Service | URL |
|---|---|
| Airflow UI | http://localhost:8080 (login: airflow / airflow) |
| FastAPI Monitor | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |

See [`airflow_lab2/README.md`](./airflow_lab2/README.md) for full setup instructions, architecture diagrams, Airflow 3.x migration notes, and troubleshooting.

---

## Lab 3: Docker — Titanic Survival Prediction with XGBoost

**Directory:** [`docker_lab1/`](./docker_lab1/)

Containerizes an ML training pipeline that downloads & preprocesses the Titanic dataset, trains an XGBoost classifier with feature engineering, evaluates with full metrics, and persists the model + metrics JSON to a volume-mounted host directory.

This is a modified version of the [original Docker Lab 1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Docker_Labs/Lab_1). Key changes from the original: Titanic dataset replaces Iris, XGBoost replaces Random Forest, full evaluation metrics added, volume mount for artifact persistence, and `python:3.10-slim` base image.

| Aspect | Details                                                                                           |
|---|---------------------------------------------------------------------------------------------------|
| Model | XGBClassifier with tuned hyperparameters                                                          |
| Dataset | Titanic: survival prediction with feature engineering (FamilySize, IsAlone, categorical encoding) |
| Evaluation | Accuracy, Precision, Recall, F1, ROC AUC, Confusion Matrix                                        |
| Infrastructure | Docker with volume mounts for model + metrics persistence                                         |
| Key Files | `src/main.py` (training script), `Dockerfile`, `src/requirements.txt`                             |

**Run it:**

```bash
cd docker_lab1
docker build -t titanic-xgb:v1 .
mkdir -p ./output
docker run -v $(pwd)/output:/app/output titanic-xgb:v1
```

---

## Technologies

| Technology | Lab 1 | Lab 2 | Lab 3 |
|---|---|---|---|
| Python 3.12 | ✅ | ✅ | ✅ |
| FastAPI | ✅ serving | ✅ monitoring | — |
| scikit-learn | ✅ Random Forest | ✅ Logistic Regression | — |
| XGBoost | — | — | ✅ |
| Apache Airflow 3.x | — | ✅ | — |
| Docker | — | ✅ | ✅ |
| Docker Compose | — | ✅ | — |
| PostgreSQL | — | ✅ | — |
| Gmail SMTP | — | ✅ | — |
| JWT Authentication | — | ✅ | — |
```
---

## Credits

- **Course:** [Prof. Ramin Mohammadi](https://www.mlwithramin.com/) — MLOps, Northeastern University
- **Original Labs:** [MLOps GitHub Repo](https://github.com/raminmohammadi/MLOps/)
- **Modified by:** Sunny Yadav