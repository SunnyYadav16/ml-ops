from pydantic import BaseModel, Field

class HousingFeatures(BaseModel):
    MedInc: float = Field(..., gt=0, description="Median Income in block group")
    HouseAge: float = Field(..., gt=0, description="Median House Age in block group")
    AveRooms: float = Field(..., gt=0, description="Average number of rooms per household")
    AveBedrooms: float = Field(..., gt=0, description="Average number of bedrooms per household")
    Population: float = Field(..., gt=0, description="Block group population")
    AveOccup: float = Field(..., gt=0, description="Average number of household members")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")

class PredictionResponse(BaseModel):
    predicted_value_100k: float
    estimated_price_usd: float