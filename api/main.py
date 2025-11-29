from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import warnings
import joblib
import sys
import os



# Suppress warnings to keep logs clean
warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*LightGBM binary classifier.*")

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from churn.preprocessing import DataCleaner

app = FastAPI(title="Telco Churn Prediction API")

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "lightgbm_pipeline.joblib")
CLEANER_PATH = os.path.join(MODELS_DIR, "data_cleaner.joblib")

# Load models
try:
    pipeline = joblib.load(PIPELINE_PATH)
    cleaner = joblib.load(CLEANER_PATH)
except FileNotFoundError as e:
    print(f"Error: Model files not found at {MODELS_DIR}")
    raise e

# Data scheme
class CustomerData(BaseModel):
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str

# Endpoints

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(customer: CustomerData):
    try:
        input_data = customer.model_dump()
        df = pd.DataFrame([input_data])

        # Preprocessing
        df_clean = cleaner.transform(df, y=None)

        # Prediction
        pred_proba = pipeline.predict_proba(df_clean)[:, 1][0]
        pred_label = int(pred_proba > 0.5)

        return {
            "churn_prediction": pred_label,
            "churn_probability": round(float(pred_proba), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))