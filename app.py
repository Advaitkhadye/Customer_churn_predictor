import streamlit as st
import joblib
import numpy as np

# Cache model + scaler so they are loaded only once
@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

scaler = load_scaler()
model = load_model()

# App Title
st.title("Churn Prediction App")

st.divider()

st.write("Please enter the values and hit the predict button for getting a prediction.")

st.divider()

# Inputs
age = st.number_input("Enter age", min_value=10, max_value=100, value=30)
tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)
monthlycharge = st.number_input("Enter Monthly Charge", min_value=30, max_value=150, value=70)
gender = st.selectbox("Enter the Gender", ["Male", "Female"])

st.divider()

# Prediction
if st.button("Predict!"):
    gender_selected = 1 if gender == "Female" else 0

    X = np.array([age, gender_selected, tenure, monthlycharge]).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    predicted = "Yes" if prediction == 1 else "No"

    st.success(f"Predicted: {predicted}")
    st.balloons()
else:
    st.info("Please enter the values and use predict button")
