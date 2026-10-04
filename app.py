
import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("Simulated Drug Screening Model")
st.write("Educational synthetic machine-learning classification demo.")

# Input fields
molecular_weight = st.number_input("Molecular Weight", value=0.0)
binding_affinity = st.number_input("Binding Affinity", value=0.0)
solubility_score = st.number_input("Solubility Score", value=0.0)
toxicity_score = st.number_input("Toxicity Score", value=0.0)
feature_5 = st.number_input("Feature 5", value=0.0)
feature_6 = st.number_input("Feature 6", value=0.0)

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([[
        molecular_weight,
        binding_affinity,
        solubility_score,
        toxicity_score,
        feature_5,
        feature_6
    ]], columns=[
        "molecular_weight",
        "binding_affinity",
        "solubility_score",
        "toxicity_score",
        "feature_5",
        "feature_6"
    ])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("Prediction: Active")
    else:
        st.info("Prediction: Inactive")
