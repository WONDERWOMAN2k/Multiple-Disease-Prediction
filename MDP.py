import streamlit as st
import pickle
import numpy as np

# Load models
try:
    kidney_model = pickle.load(open('kidney_model.pkl', 'rb'))
    liver_model = pickle.load(open('liver_model.pkl', 'rb'))
    parkinson_model = pickle.load(open('parkinson_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("One or more model files are missing. Please upload them to your GitHub repo.")
    st.stop()

# App title
st.set_page_config(page_title="Multiple Disease Prediction", layout="centered")
st.title("🧠 Multiple Disease Prediction using ML")

# Sidebar for disease selection
selected_disease = st.sidebar.selectbox("Select Disease", 
    ["Kidney Disease", "Liver Disease", "Parkinson's Disease"])

# Kidney Disease Input
if selected_disease == "Kidney Disease":
    st.header("🔍 Kidney Disease Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        sg = st.number_input("Specific Gravity (e.g. 1.02)", min_value=1.00, max_value=1.05, step=0.01)
    with col2:
        al = st.number_input("Albumin (0-5)", min_value=0, max_value=5)
    with col3:
        sc = st.number_input("Serum Creatinine (e.g. 1.2)", min_value=0.0, max_value=15.0, step=0.1)

    with col1:
        hemo = st.number_input("Hemoglobin (e.g. 12.0)", min_value=3.0, max_value=17.0, step=0.1)
    with col2:
        pcv = st.number_input("Packed Cell Volume", min_value=20, max_value=60)
    with col3:
        htn = st.selectbox("Hypertension", ["yes", "no"])

    # Convert inputs to model format
    input_data = np.array([sg, al, sc, hemo, pcv, 1 if htn == "yes" else 0]).reshape(1, -1)
    
    if st.button("Predict Kidney Disease"):
        result = kidney_model.predict(input_data)
        if result[0] == 1:
            st.error("High risk of Kidney Disease 🚨")
        else:
            st.success("No signs of Kidney Disease ✅")

# Liver Disease Input
elif selected_disease == "Liver Disease":
    st.header("🩺 Liver Disease Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100)
    with col2:
        total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, max_value=75.0)
    with col3:
        alk_phosphate = st.number_input("Alkaline Phosphotase", min_value=0, max_value=2000)

    with col1:
        sgpt = st.number_input("SGPT", min_value=0, max_value=2000)
    with col2:
        sgpt_altt = st.number_input("SGOT", min_value=0, max_value=2000)
    with col3:
        total_protein = st.number_input("Total Protein", min_value=0.0, max_value=10.0)

    input_data = np.array([age, total_bilirubin, alk_phosphate, sgpt, sgpt_altt, total_protein]).reshape(1, -1)

    if st.button("Predict Liver Disease"):
        result = liver_model.predict(input_data)
        if result[0] == 1:
            st.error("Liver Disease Detected 🚨")
        else:
            st.success("Liver Function Normal ✅")

# Parkinson's Disease Input
elif selected_disease == "Parkinson's Disease":
    st.header("🧠 Parkinson's Disease Predict
