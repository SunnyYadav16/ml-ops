import os
import logging
from datetime import datetime

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration
AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL", "http://airflow-apiserver:8080")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "airflow")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "airflow")
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


# JWT Token helper
def get_jwt_token() -> str:
    """Obtain a JWT token from Airflow's /auth/token endpoint (FAB auth)."""
    try:
        resp = requests.post(
            f"{AIRFLOW_API_URL}/auth/token",
            json={"username": AIRFLOW_USERNAME, "password": AIRFLOW_PASSWORD},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        token = resp.json().get("access_token")
        logger.info("JWT token obtained successfully")
        return token
    except Exception as e:
        logger.error(f"Failed to get JWT token: {e}")
        return ""


def airflow_api_get(endpoint: str) -> dict:
    """Make an authenticated GET request to the Airflow REST API."""
    token = get_jwt_token()
    if not token:
        return {"error": "Could not authenticate with Airflow API"}

    try:
        resp = requests.get(
            f"{AIRFLOW_API_URL}{endpoint}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Airflow API request failed: {e}")
        return {"error": str(e)}


# Check pipeline status via Airflow API
def check_dag_status() -> dict:
    """
    Query the most recent DAG run of 'Airflow_Lab2' via the REST API.
    Returns a dict with status info.
    """
    data = airflow_api_get("/api/v2/dags/Airflow_Lab2/dagRuns?limit=1&order_by=-start_date")

    if "error" in data:
        return {"success": False, "state": "unknown", "detail": data["error"]}

    dag_runs = data.get("dag_runs", [])
    if not dag_runs:
        return {"success": False, "state": "no_runs", "detail": "No runs found for Airflow_Lab2"}

    latest = dag_runs[0]
    state = latest.get("state", "unknown")
    return {
        "success": state == "success",
        "state": state,
        "dag_run_id": latest.get("dag_run_id"),
        "start_date": latest.get("start_date"),
        "end_date": latest.get("end_date"),
    }


# FastAPI Application
app = FastAPI(
    title="Airflow Lab2 — ML Pipeline Monitor",
    description="FastAPI service to monitor the Airflow ML pipeline status via JWT-authenticated API calls",
    version="2.0.0",
)


def _read_template(name: str) -> str:
    path = os.path.join(TEMPLATES_DIR, name)
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return f"<h1>Template '{name}' not found</h1>"


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <head><title>ML Pipeline Monitor</title></head>
    <body style="font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 20px;">
        <h1>🚀 Airflow Lab2 — ML Pipeline Monitor</h1>
        <p>Welcome to the FastAPI monitoring service (Airflow 3.x + JWT auth).</p>
        <ul>
            <li><a href="/api">/api</a> — JSON status of the last pipeline run</li>
            <li><a href="/status">/status</a> — HTML status page</li>
            <li><a href="/health">/health</a> — Health check</li>
            <li><a href="/docs">/docs</a> — Interactive Swagger UI</li>
        </ul>
    </body>
    </html>
    """


@app.get("/api")
async def api_status():
    """Return JSON status of the last Airflow_Lab2 DAG run."""
    status = check_dag_status()
    return JSONResponse(
        content={
            "dag_id": "Airflow_Lab2",
            "status": status["state"],
            "success": status["success"],
            "detail": status.get("detail", ""),
            "dag_run_id": status.get("dag_run_id"),
            "start_date": status.get("start_date"),
            "end_date": status.get("end_date"),
            "timestamp": datetime.utcnow().isoformat(),
        },
        status_code=200 if status["success"] else 500,
    )


@app.get("/status", response_class=HTMLResponse)
async def status_page():
    """Render an HTML page based on the last pipeline run status."""
    status = check_dag_status()
    template = "success.html" if status["success"] else "failure.html"
    return HTMLResponse(content=_read_template(template))


@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "airflow-lab2-fastapi-monitor"}


# Entry point (when run as a standalone script in Docker)
if __name__ == "__main__":
    logger.info("Starting FastAPI monitoring server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")