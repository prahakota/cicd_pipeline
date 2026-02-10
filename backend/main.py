from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np 

app = FastAPI(title="Linear Regression API")

#import os
#model_PATH = path(os.getenv("Model_PATH","model/model.pkl"))

MODEL_PATH = "models/model.pkl"
try:
    with open(MODEL_PATH,"rb") as f:
        model = pickle.load(f)

except Exception as e:

    raise RuntimeError(f"failed to load model from {MODEL_PATH}: {e}")

#input schema
class InoutData(BaseModel):
    area: float
    bedrooms: int

#prediction endpoint
@app.post("/predict")
def predict(data: InoutData):
    x = np.array([[data.area, data.bedrooms]])
    prediction = model.predict(x)[0]
    return {"predicted_price": float(prediction)}