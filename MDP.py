import streamlit as st
import pickle
import numpy as np

# Load models
kidney_model = pickle.load(open('kidney_model.pkl', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.pkl', 'rb'))
liver_model = pickle.load(open('liver_model.pkl', 'rb'))

# App title
st.title("Multiple Disease Prediction System")

# Tabs for each disease
tab1, tab2, tab3 = st.tabs(["Kidney Disease", "Parkinson's Disease", "Liver Disease"])

with tab1:
    st.header("Kidney Disease Prediction")
    # Collect user inputs
    # Example:
    age = st.number_input("Age", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    # ... add other relevant inputs
    if st.button("Predict Kidney Disease"):
        # Prepare input data
        input_data = np.array([[age, bp, ...]])  # Replace '...' with other inputs
        prediction = kidney_model.predict(input_data)
        if prediction[0] == 1:
            st.error("Kidney Disease Detected")
        else:
            st.success("No Kidney Disease Detected")

with tab2:
    st.header("Parkinson's Disease Prediction")
    # Collect user inputs
    # Example:
    fo = st.number_input("MDVP:Fo(Hz)")
    fhi = st.number_input("MDVP:Fhi(Hz)")
    # ... add other relevant inputs
    if st.button("Predict Parkinson's Disease"):
        # Prepare input data
        input_data = np.array([[fo, fhi, ...]])  # Replace '...' with other inputs
        prediction = parkinsons_model.predict(input_data)
        if prediction[0] == 1:
            st.error("Parkinson's Disease Detected")
        else:
            st.success("No Parkinson's Disease Detected")

with tab3:
    st.header("Liver Disease Prediction")
    # Collect user inputs
    # Example:
    age = st.number_input("Age", min_value=0)
    total_bilirubin = st.number_input("Total Bilirubin")
    # ... add other relevant inputs
    if st.button("Predict Liver Disease"):
        # Prepare input data
        input_data = np.array([[age, total_bilirubin, ...]])  # Replace '...' with other inputs
        prediction = liver_model.predict(input_data)
        if prediction[0] == 1:
            st.error("Liver Disease Detected")
        else:
            st.success("No Liver Disease Detected")
