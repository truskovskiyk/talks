from fastapi import FastAPI
from pydantic import BaseModel
from qqp_inference.model import PythonPredictor


class Payload(BaseModel):
    q1: str
    q2: str


class Prediction(BaseModel):
    is_duplicate: bool
    model_version: str
    score: float


app = FastAPI()
predictor = PythonPredictor.create_for_demo()


@app.post("/predict", response_model=Prediction)
def predict(payload: Payload):
    prediction = predictor.predict(payload=payload.dict())
    return prediction
