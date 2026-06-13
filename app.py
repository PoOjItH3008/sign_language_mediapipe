from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI()

model = joblib.load("sign_model.pkl")

class LandmarkInput(BaseModel):
    landmarks: list

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/predict")
def predict(data: LandmarkInput):

    prediction = model.predict([data.landmarks])[0]

    confidence = float(
        np.max(
            model.predict_proba([data.landmarks])
        )
    )

    return {
        "label": str(prediction),
        "confidence": confidence
    }