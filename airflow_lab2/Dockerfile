FROM apache/airflow:3.0.2

# Install ML and FastAPI dependencies
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt