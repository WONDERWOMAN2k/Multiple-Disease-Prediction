import streamlit as st
import pickle
import numpy as np

# Load models (ensure these files are in your repo)
kidney_model = pickle.load(open('kidney_model.pkl', 'rb'))
liver_model = pickle.load(open('liver_model.pkl', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.pkl', 'rb'))
parkinsons_scaler = pickle.load(open('parkinsons_scaler.pkl', 'rb'))

st.title("Multiple Disease Prediction System")

tab1, tab2, tab3 = st.tabs(["Kidney Disease", "Parkinson's Disease", "Liver Disease"])

with tab1:
    st.header("Kidney Disease Prediction")

    # Replace these inputs with all features needed by your kidney_model
    age = st.number_input("Age", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    # Add all other features your model requires here as inputs

    if st.button("Predict Kidney Disease"):
        # Prepare input array with all features in correct order
        # Example: input_data = np.array([[age, bp, feature3, feature4, ...]])
        input_data = np.array([[age, bp]])  # expand with all features

        prediction = kidney_model.predict(input_data)
        if prediction[0] == 1:
            st.error("Kidney Disease Detected")
        else:
            st.success("No Kidney Disease Detected")

with tab2:
    st.header("Parkinson's Disease Prediction")

    # Add inputs for all features your parkinsons_model expects
    fo = st.number_input("MDVP:Fo(Hz)")
    fhi = st.number_input("MDVP:Fhi(Hz)")
    # Add all other Parkinson's features here...

    if st.button("Predict Parkinson's Disease"):
        # Scale inputs as done in training
        input_data = np.array([[fo, fhi]])  # expand with all features
        input_data_scaled = parkinsons_scaler.transform(input_data)

        prediction = parkinsons_model.predict(input_data_scaled)
        if prediction[0] == 1:
            st.error("Parkinson's Disease Detected")
        else:
            st.success("No Parkinson's Disease Detected")

with tab3:
    st.header("Liver Disease Prediction")

    # Add inputs for all liver disease features
    age = st.number_input("Age", min_value=0)
    total_bilirubin = st.number_input("Total Bilirubin")
    # Add other features here...

    if st.button("Predict Liver Disease"):
        input_data = np.array([[age, total_bilirubin]])  # expand with all features

        prediction = liver_model.predict(input_data)
        if prediction[0] == 1:
            st.error("Liver Disease Detected")
        else:
            st.success("No Liver Disease Detected")
