import streamlit as st
import pickle
import numpy as np
# save_models.py

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------
# KIDNEY MODEL
# --------------------------
kidney_data = pd.read_csv("kidney_disease.csv")
kidney_data.dropna(inplace=True)
kidney_data = kidney_data.drop(['id'], axis=1)
kidney_data = pd.get_dummies(kidney_data)

X_kidney = kidney_data.drop('classification_notckd', axis=1)
y_kidney = kidney_data['classification_notckd']

X_train_kidney, X_test_kidney, y_train_kidney, y_test_kidney = train_test_split(X_kidney, y_kidney, test_size=0.2, random_state=42)
kidney_model = RandomForestClassifier()
kidney_model.fit(X_train_kidney, y_train_kidney)

with open('kidney_model.pkl', 'wb') as f:
    pickle.dump(kidney_model, f)

# --------------------------
# LIVER MODEL
# --------------------------
liver_data = pd.read_csv("liver_disease.csv")
liver_data = liver_data.dropna()
liver_data['Dataset'] = liver_data['Dataset'].apply(lambda x: 1 if x == 2 else 0)

X_liver = liver_data.drop('Dataset', axis=1)
y_liver = liver_data['Dataset']

X_train_liver, X_test_liver, y_train_liver, y_test_liver = train_test_split(X_liver, y_liver, test_size=0.2, random_state=42)
liver_model = RandomForestClassifier()
liver_model.fit(X_train_liver, y_train_liver)

with open('liver_model.pkl', 'wb') as f:
    pickle.dump(liver_model, f)

# --------------------------
# PARKINSON'S MODEL
# --------------------------
parkinsons_data = pd.read_csv("parkinsons.csv")
X_parkinsons = parkinsons_data.drop(['status', 'name'], axis=1)
y_parkinsons = parkinsons_data['status']

scaler = StandardScaler()
X_parkinsons_scaled = scaler.fit_transform(X_parkinsons)

X_train_parkinsons, X_test_parkinsons, y_train_parkinsons, y_test_parkinsons = train_test_split(X_parkinsons_scaled, y_parkinsons, test_size=0.2, random_state=42)
parkinsons_model = RandomForestClassifier()
parkinsons_model.fit(X_train_parkinsons, y_train_parkinsons)

with open('parkinsons_model.pkl', 'wb') as f:
    pickle.dump(parkinsons_model, f)

print("✅ All 3 models trained and saved as .pkl files successfully!")

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
