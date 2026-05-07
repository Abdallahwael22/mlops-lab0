import os
import mlflow
import dagshub
import pandas as pd
from src.logger import ExecutorLogger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
from control import BatchPredictionRequest
ml_component = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/abdallahwael082/mlops-lab0.mlflow"
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    os.environ["MLFLOW_TRACKING_USERNAME"] = "abdallahwael082"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "0ce0281adf60193dd06e4ca2190e4624021a1d69"
    
    try:
        ml_component["model"] = mlflow.pyfunc.load_model("models:/lr@Production")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
    yield
    print("Shutting down")
    ml_component.clear()
    
app = FastAPI(lifespan=lifespan)

@app.post("/predict")

async def predict(batch_request: BatchPredictionRequest):
    if "model" not in ml_component:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        passengers_df = pd.DataFrame([p.model_dump() for p in batch_request.passengers])
        predictions = ml_component["model"].predict(passengers_df)
        result=[]
        for passenger, pred in zip(batch_request.passengers, predictions):
            result.append({
                "name": passenger.Name,
                "Survived": int(pred)
            })
        return {"predictions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    