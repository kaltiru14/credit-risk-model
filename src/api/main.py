from fastapi import FastAPI
from pydantic import BaseModel
from ..predict import make_prediction

app = FastAPI()

class PredictRequest(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float

@app.post("/predict")
def predict(request: PredictRequest):
    features = [
        request.feature_1,
        request.feature_2,
        request.feature_3,
        request.feature_4,
        request.feature_5
    ]
    return make_prediction(features)
