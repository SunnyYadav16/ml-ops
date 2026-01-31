# California Housing Price Prediction API

This project is a simple MLOps lab demonstrating how to build, train, and serve a Machine Learning model using **FastAPI** and **Scikit-Learn**. 

It uses the **California Housing Dataset** to train a Random Forest Regressor that predicts house values based on 8 features (income, age, rooms, etc.).

## 📂 Project Structure

```text
fastapi_lab/
├── model/                  # Serialized model artifacts
│   └── housing_model.pkl   # (Generated after training)
├── src/                    # Source code
│   ├── __init__.py
│   ├── data.py             # Data ingestion and splitting
│   ├── schema.py           # Pydantic models
│   ├── main.py             # FastAPI application & endpoints
│   ├── predict.py          # Model loading and inference logic
│   └── train.py            # Training script
├── README.md               # Project documentation
└── requirements.txt        # Dependencies