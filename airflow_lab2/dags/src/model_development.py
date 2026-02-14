import os
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Paths — inside the Docker container, dags are at /opt/airflow/dags
DAGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(DAGS_DIR, "data", "breast_cancer.csv")
MODEL_DIR = os.path.join(DAGS_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Feature columns (all 30 features from the breast cancer dataset)
FEATURE_COLS = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]
TARGET_COL = "target"

logger = logging.getLogger(__name__)


# Task 1: Load Data
def load_data(**kwargs):
    """Load the breast cancer CSV and push it to XCom."""
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns[:5])}... ({len(df.columns)} total)")
    logger.info(f"Target distribution:\n{df[TARGET_COL].value_counts()}")
    logger.info(f"First 5 rows:\n{df.head()}")

    kwargs["ti"].xcom_push(key="raw_data", value=df.to_json())
    return "Data loaded successfully"


# Task 2: Data Preprocessing
def data_preprocessing(**kwargs):
    """
    Preprocess the data:
    - Check for nulls
    - Scale all 30 features using StandardScaler
    - Save the scaler for inference
    """
    ti = kwargs["ti"]
    raw_json = ti.xcom_pull(key="raw_data", task_ids="load_data")
    df = pd.read_json(raw_json)

    logger.info("Checking for missing values...")
    null_counts = df.isnull().sum()
    logger.info(f"Total nulls: {null_counts.sum()}")

    # Drop any rows with missing values
    df = df.dropna()
    logger.info(f"Shape after dropping nulls: {df.shape}")

    # Feature scaling
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    logger.info("Features scaled with StandardScaler")

    # Save scaler for inference
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {SCALER_PATH}")

    ti.xcom_push(key="processed_data", value=df.to_json())
    return "Data preprocessing completed"


# Task 3: Separate Data into Train / Test
def separate_data_outputs(**kwargs):
    """Split data into training and testing sets (80/20, stratified)."""
    ti = kwargs["ti"]
    processed_json = ti.xcom_pull(key="processed_data", task_ids="data_preprocessing")
    df = pd.read_json(processed_json)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train set size: {X_train.shape[0]} samples")
    logger.info(f"Test set size:  {X_test.shape[0]} samples")
    logger.info(f"Train target distribution:\n{y_train.value_counts()}")

    ti.xcom_push(key="X_train", value=X_train.to_json())
    ti.xcom_push(key="X_test", value=X_test.to_json())
    ti.xcom_push(key="y_train", value=y_train.to_json())
    ti.xcom_push(key="y_test", value=y_test.to_json())
    return "Data split completed"


# Task 4: Build Model
def build_model(**kwargs):
    """Train a Logistic Regression model and save it."""
    ti = kwargs["ti"]
    X_train = pd.read_json(ti.xcom_pull(key="X_train", task_ids="separate_data_outputs"))
    y_train = pd.read_json(
        ti.xcom_pull(key="y_train", task_ids="separate_data_outputs"), typ="series"
    )

    logger.info("Training Logistic Regression model...")
    logger.info(f"  Features: {X_train.shape[1]}")
    logger.info(f"  Samples:  {X_train.shape[0]}")

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {MODEL_PATH}")

    # Log training accuracy
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    logger.info(f"Training accuracy: {train_accuracy:.4f}")
    ti.xcom_push(key="train_accuracy", value=float(train_accuracy))

    return "Model training completed"


# Task 5: Load & Evaluate Model
def load_model(**kwargs):
    """Load the saved model and evaluate on the test set."""
    ti = kwargs["ti"]
    X_test = pd.read_json(ti.xcom_pull(key="X_test", task_ids="separate_data_outputs"))
    y_test = pd.read_json(
        ti.xcom_pull(key="y_test", task_ids="separate_data_outputs"), typ="series"
    )

    logger.info(f"Loading model from {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, target_names=["Malignant", "Benign"]
    )

    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    logger.info(f"Classification Report:\n{report}")

    ti.xcom_push(key="test_accuracy", value=float(test_accuracy))
    ti.xcom_push(key="classification_report", value=report)

    return "Model evaluation completed"