from fastapi.testclient import TestClient
import pytest
import sys
import os

# Add api folder to path to import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../api')))
from main import app 

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}

@pytest.mark.filterwarnings("ignore:X does not have valid feature names") # Suppress to keep logs clean
def test_prediction_churn():
    # Data for a customer who is likely to churn (High monthly charges, fiber optic, low tenure)
    payload = {
        "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.0, "TotalCharges": "90.0"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "churn_probability" in response.json()
    # Expect high probability for this profile
    assert response.json()["churn_probability"] > 0.1