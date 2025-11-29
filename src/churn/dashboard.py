import streamlit as st
import requests
import os

st.title("Customer Churn Predictor")

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 120.0, 70.0)
    total_charges = st.text_input("Total Charges ($)", str(monthly_charges * tenure))

with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

if st.button("Predict Churn Risk"):
    # Gather data (add defaults for other fields not in UI to keep it simple)
    payload = {
        "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": tenure, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": internet, "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": contract, "PaperlessBilling": "Yes",
        "PaymentMethod": payment, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
    }
    
    # Call local API
    api_url = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

    try:
        response = requests.post(api_url, json=payload)
        prob = response.json()['churn_probability']
        
        if prob > 0.5:
            st.error(f"High Churn Risk: {prob:.1%}")
        else:
            st.success(f"Customer Safe: {prob:.1%}")
    except:
        st.warning("Make sure API is running! (uvicorn api.main:app --reload)")