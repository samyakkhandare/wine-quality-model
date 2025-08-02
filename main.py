from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import pickle

# Load model
with open("wine_quality_model.pkl", "rb") as f:
    model = pickle.load(f)

# FastAPI app
app = FastAPI(title="Wine Quality Prediction API")

# Templates setup
templates = Jinja2Templates(directory="templates")

# Input Schema
class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# Root Route (HTML Page)
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction API Route
@app.post("/predict")
def predict_wine_quality(data: WineInput):
    input_data = np.array([[data.fixed_acidity, data.volatile_acidity, data.citric_acid,
                            data.residual_sugar, data.chlorides, data.free_sulfur_dioxide,
                            data.total_sulfur_dioxide, data.density, data.pH,
                            data.sulphates, data.alcohol]])
    
    prediction = model.predict(input_data)[0]
    return {"predicted_quality_score": round(prediction, 2)}
