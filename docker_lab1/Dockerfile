# Base image: Python 3.10 slim (smaller than full python:3.10)
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first
COPY src/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ .

# Create output directory for model artifacts
# This will be the mount point for the volume
RUN mkdir -p /app/output

# Run the training script when the container launches
CMD ["python", "main.py"]