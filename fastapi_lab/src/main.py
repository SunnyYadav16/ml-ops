from fastapi import FastAPI, HTTPException

from predict import predictor
from schema import HousingFeatures, PredictionResponse

app = FastAPI(title="California Housing Price Predictor")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Housing Price Prediction API"}


@app.post("/predict", response_model=PredictionResponse)
def predict_price(features: HousingFeatures):
    try:
        feature_list = [
            features.MedInc,
            features.HouseAge,
            features.AveRooms,
            features.AveBedrooms,
            features.Population,
            features.AveOccup,
            features.Latitude,
            features.Longitude
        ]

        prediction = float(predictor.predict(feature_list))

        return {
            "predicted_value_100k": prediction,
            "estimated_price_usd": prediction * 100000
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))