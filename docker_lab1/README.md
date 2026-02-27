# Docker Lab 1 — Titanic Survival Prediction with XGBoost

## Overview

Containerize an ML training pipeline that:
- Downloads & preprocesses the Titanic dataset
- Trains an XGBoost classifier with feature engineering
- Evaluates with full metrics (accuracy, precision, recall, F1, ROC AUC, confusion matrix)
- Persists the model + metrics JSON to a **volume-mounted** directory on your host

## Project Structure

```
docker_lab1/
├── Dockerfile
├── README.md
└── src/
    ├── main.py            # Training script
    └── requirements.txt   # Python dependencies
```

## Prerequisites

- Docker Desktop installed and running
- Terminal / command line access

## Steps

### 1. Build the Docker image

```bash
docker build -t titanic-xgb:v1 .
```

**What's happening:** Docker reads the `Dockerfile`, pulls the `python:3.10-slim` base image,
installs dependencies from `requirements.txt`, and copies your code into the image.

### 2. Run the container (basic — no volume mount)

```bash
docker run titanic-xgb:v1
```

You should see the training output and metrics printed to the console.
The model is saved *inside* the container (ephemeral — lost when container is removed).

### 3. Run with a volume mount (persist model to your machine)

```bash
# Create a local directory for outputs
mkdir -p ./output

# Run with volume mount: -v <host_path>:<container_path>
docker run -v $(pwd)/output:/app/output titanic-xgb:v1
```

After this runs, check your local `./output/` folder — you'll find:
- `titanic_xgb_model.pkl` — the trained XGBoost model
- `metrics.json` — evaluation metrics with timestamp

### 4. Inspect the saved metrics

```bash
cat ./output/metrics.json | python -m json.tool
```

### 5. (Optional) Save the image as a tar archive

```bash
docker save titanic-xgb:v1 > titanic_xgb_image.tar
```

### 6. Cleanup

```bash
# Remove stopped containers
docker container prune

# Remove the image
docker rmi titanic-xgb:v1
```

## Key Docker Concepts Practiced

| Concept | How it's used here |
|---|---|
| `FROM` | Base image — `python:3.10-slim` (smaller than full `python:3.10`) |
| `WORKDIR` | Sets `/app` as the working directory inside the container |
| `COPY` | Copies `requirements.txt` first (layer caching), then code |
| `RUN` | Installs pip dependencies at build time |
| `CMD` | Defines the default command when the container starts |
| Volume mount (`-v`) | Maps host directory to container directory to persist artifacts |
| Layer caching | Requirements copied before code — rebuild is fast when only code changes |

## What's Different from the Original Lab

| Original Lab 1 | This Modified Version |
|---|---|
| Iris dataset (sklearn built-in) | Titanic dataset (downloaded CSV) |
| RandomForestClassifier | XGBClassifier with tuned hyperparameters |
| No evaluation | Full metrics: accuracy, precision, recall, F1, AUC, confusion matrix |
| Model saved inside container (lost) | Volume mount persists model + metrics to host |
| `python:3.10` base image | `python:3.10-slim` (smaller, production-ready) |
| No feature engineering | FamilySize, IsAlone, categorical encoding |