from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI Server
app = FastAPI(
    title="Regression Prediction API",
    description="A FastAPI server to serve trained regression models and provide predictions.",
    version="1.0.0"
)

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models
model_paths = {
    "RandomForest": "RandomForest.pkl",
    "SVR": "SVR.pkl",
    "KNN": "KNN.pkl"
}
try:
    models = {name: joblib.load(path) for name, path in model_paths.items()}
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise HTTPException(status_code=500, detail="Error loading models")

class InputData(BaseModel):
    features: List[float]

@app.post("/predict/{model_name}", summary="Get prediction from a specified model")
def predict(model_name: str, data: InputData):
    """
    Get a prediction from the specified model.

    - model_name: Name of the model to use for prediction (RandomForest, SVR, KNN)
    - data: Input data containing features for prediction
    """
    logger.info(f"Received prediction request for model: {model_name} with data: {data.features}")
    if model_name not in models:
        logger.error(f"Model {model_name} not found")
        raise HTTPException(status_code=404, detail="Model not found")
    model = models[model_name]
    try:
        # Validate input data
        if not all(isinstance(feature, (int, float)) for feature in data.features):
            raise ValueError("All features must be numeric (int or float)")
        
        input_array = np.array(data.features).reshape(1, -1)
        logger.info(f"Input array for prediction: {input_array}")
        prediction = model.predict(input_array)
        logger.info(f"Prediction result: {prediction}")
        return {"prediction": prediction.tolist()}
    except ValueError as ve:
        logger.error(f"ValueError during prediction: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid input data: {ve}")
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")

@app.get("/models", summary="List all available models")
def list_models():
    """
    List all available models for prediction.
    """
    return {"models": list(models.keys())}

@app.get("/", summary="Home endpoint")
def home():
    """
    Home endpoint to check if the server is running.
    """
    return {"message": "Welcome to the FastAPI Regression Prediction Server"}
