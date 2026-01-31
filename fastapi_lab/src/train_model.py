import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from data import load_and_split_data

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/housing_model.pkl")

def train():
    # 1. Getting the Data
    X_train, X_test, y_train, y_test = load_and_split_data()

    # 2. Train Model
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # 3. Evaluate
    score = r2_score(y_test, model.predict(X_test))
    print(f"Model R^2 Score: {score:.2f}")

    # 4. Save Model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()