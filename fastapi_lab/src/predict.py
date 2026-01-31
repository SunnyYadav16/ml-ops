import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/housing_model.pkl")

class HousingPredictor:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Loads the model if it exists."""
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
        else:
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")

    def predict(self, features: list):
        """
        Expects a list of features: [MedInc, HouseAge, AveRooms, ...]
        """
        if not self.model:
            self.load_model()

        return self.model.predict([features])[0]


predictor = HousingPredictor()