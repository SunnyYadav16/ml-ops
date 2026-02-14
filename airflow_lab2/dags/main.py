import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator

# Import ML pipeline functions
from src.model_development import (
    load_data,
    data_preprocessing,
    separate_data_outputs,
    build_model,
    load_model,
)

# ← UPDATE THIS with your actual email
NOTIFICATION_EMAIL_TO = os.environ.get("NOTIFICATION_EMAIL_TO", "sunny@gmail.com")

# Default arguments
default_args = {
    "owner": "Sunny",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# DAG definition
with DAG(
    dag_id="Airflow_Lab2",
    default_args=default_args,
    description="ML pipeline: Logistic Regression on advertising data with email alerts",
    schedule=None,
    catchup=False,
    tags=["mlops", "lab2", "ml-pipeline"],
) as dag:

    # ── Task 1: Owner task (BashOperator) ──
    owner_task = BashOperator(
        task_id="owner_task",
        bash_command='echo "Pipeline owned by: Sunny | Airflow Lab 2"',
    )

    # ── Task 2: Load data ──
    load_data_task = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
    )

    # ── Task 3: Preprocess data ──
    preprocessing_task = PythonOperator(
        task_id="data_preprocessing",
        python_callable=data_preprocessing,
    )

    # ── Task 4: Split data ──
    split_data_task = PythonOperator(
        task_id="separate_data_outputs",
        python_callable=separate_data_outputs,
    )

    # ── Task 5: Train model ──
    build_model_task = PythonOperator(
        task_id="build_model",
        python_callable=build_model,
    )

    # ── Task 6: Evaluate model ──
    load_model_task = PythonOperator(
        task_id="load_model",
        python_callable=load_model,
    )

    # ── Task 7: Send success email ──
    send_email = EmailOperator(
        task_id="send_email",
        to=NOTIFICATION_EMAIL_TO,
        subject="Airflow Lab2 - Pipeline Run Complete",
        html_content="""
        <h3>Airflow Lab2 Pipeline Notification</h3>
        <p>The ML pipeline has completed its run.</p>
        <p>Check the Airflow UI or FastAPI monitoring endpoint for details.</p>
        """,
    )

    # ── Task dependencies ──
    (
        owner_task
        >> load_data_task
        >> preprocessing_task
        >> split_data_task
        >> build_model_task
        >> load_model_task
        >> send_email
    )