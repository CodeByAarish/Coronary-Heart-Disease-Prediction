import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ====== Load model and scaler ======
model_path = "logistic_model.pkl"
scaler_path = "scaler.pkl"

model, scaler = None, None

if os.path.exists(model_path) and os.path.exists(scaler_path):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        st.error(f"❌ Error loading model/scaler: {e}")
        st.stop()
else:
    st.error("❌ Model or Scaler file not found! Please check paths.")
    st.stop()

# ====== Streamlit UI ======
st.title("💓 10-Year Coronary Heart Disease Risk Predictor")
st.markdown("""
<style>
/* 1. Page background */
.stApp {
    background-color: black;
    color: yellow;
}

/* 2. Title */
h1 {
    color: red !important;
}

/* 3. Welcome text */
h3, h4, h5, h6 {
    color: red !important;
}

/* 4. Label Texts */
label, .css-1v0mbdj, .css-1p05t8e {
    color: yellow !important;
}

/* 5. Radio buttons */
div[role="radiogroup"] > * > * {
    color: red !important;
}

/* 6. Sliders */
.stSlider > div {
    color: red !important;
}
.stSlider .css-1c5rx0a {
    background-color: red !important;
}

/* 7. Number Inputs */
.stNumberInput > div > input {
    color: white !important;
    background-color: #111111;
}

/* 8. Buttons */
.stButton > button {
    background-color: red !important;
    color: yellow !important;
    border: none;
}

/* 9. Success Box */
.stAlert-success {
    background-color: green !important;
    color: white !important;
}

/* 9. Error Box */
.stAlert-error {
    background-color: darkred !important;
    color: white !important;
}

/* 10. Sidebar */
section[data-testid="stSidebar"] {
    background-color: yellow !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("### 👋 Hey! **Aarish** this side and Welcome to my Machine Learning Model")
st.write("Please select your details below:")

# Gender
male = st.radio("Gender", options=[1, 0], format_func=lambda x: 'Male' if x == 1 else 'Female')

# Age
age = st.slider("Age", 20, 90, 50)

# Smoker
currentSmoker = st.radio("Are you a current smoker?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
cigsPerDay = st.slider("Cigarettes per Day", 0, 50, 0) if currentSmoker == 1 else 0

# BP Meds
BPMeds = st.radio("On BP Medication?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

# Stroke
prevalentStroke = st.radio("Any prior stroke?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

# Hypertension
prevalentHyp = st.radio("Has Hypertension?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

# Diabetes
diabetes = st.radio("Has Diabetes?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

# Other inputs
totChol = st.number_input("Total Cholesterol", value=200.0)
sysBP = st.number_input("Systolic BP", value=130.0)
diaBP = st.number_input("Diastolic BP", value=80.0)
BMI = st.number_input("BMI", value=25.0)
heartRate = st.number_input("Heart Rate", value=72.0)
glucose = st.number_input("Glucose Level", value=100.0)

# ====== Prepare input ======
input_data = pd.DataFrame([{
    'male': male,
    'age': age,
    'currentSmoker': currentSmoker,
    'cigsPerDay': cigsPerDay,
    'BPMeds': BPMeds,
    'prevalentStroke': prevalentStroke,
    'prevalentHyp': prevalentHyp,
    'diabetes': diabetes,
    'totChol': totChol,
    'sysBP': sysBP,
    'diaBP': diaBP,
    'BMI': BMI,
    'heartRate': heartRate,
    'glucose': glucose
}])

# ====== Predict ======
if st.button("Predict your CHD Risk"):
    try:
        input_scaled = scaler.transform(input_data)
        prob = model.predict_proba(input_scaled)[0][1]
        pred = model.predict(input_scaled)[0]

        st.subheader("🔍 Prediction Result:")
        st.write(f"**CHD Risk Probability (10 years):** {round(prob, 2)}")
        if pred == 1:
            st.error("⚠️ Heart Disease Risk Detected")
        else:
            st.success("✅ No Heart Disease Risk")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
