import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Train Kidney Disease model (run only if pkl not found)
def train_kidney_model():
    df = pd.read_csv('kidney_disease.csv')
    X = df[['sg', 'al', 'sc', 'hemo', 'pcv', 'htn']].copy()
    X['htn'] = X['htn'].map({'yes':1, 'no':0})
    y = df['classification']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    with open('kidney_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Train Liver Disease model
def train_liver_model():
    df = pd.read_csv('liver_disease.csv')
    X = df[['age', 'total_bilirubin', 'alk_phosphate', 'sgpt', 'sgpt_altt', 'total_protein']]
    y = df['classification']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    with open('liver_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Train Parkinson's Disease model
def train_parkinson_model():
    df = pd.read_csv('parkinsons.csv')
    X = df.drop(columns=['name', 'status'])
    y = df['status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    with open('parkinson_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Train models if pickle files missing
if not os.path.exists('kidney_model.pkl'):
    train_kidney_model()

if not os.path.exists('liver_model.pkl'):
    train_liver_model()

if not os.path.exists('parkinson_model.pkl'):
    train_parkinson_model()

# Load models safely
try:
    kidney_model = pickle.load(open('kidney_model.pkl', 'rb'))
    liver_model = pickle.load(open('liver_model.pkl', 'rb'))
    parkinson_model = pickle.load(open('parkinson_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("One or more model files are missing. Please upload them.")
    st.stop()

# Streamlit App UI
st.set_page_config(page_title="Multiple Disease Prediction", layout="centered")
st.title("🧠 Multiple Disease Prediction using ML")

selected_disease = st.sidebar.selectbox("Select Disease", 
    ["Kidney Disease", "Liver Disease", "Parkinson's Disease"])

# Kidney Disease Form
if selected_disease == "Kidney Disease":
    st.header("🔍 Kidney Disease Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        sg = st.number_input("Specific Gravity (e.g. 1.02)", min_value=1.00, max_value=1.05, step=0.01)
    with col2:
        al = st.number_input("Albumin (0-5)", min_value=0, max_value=5)
    with col3:
        sc = st.number_input("Serum Creatinine", min_value=0.0, max_value=15.0, step=0.1)

    with col1:
        hemo = st.number_input("Hemoglobin", min_value=3.0, max_value=17.0, step=0.1)
    with col2:
        pcv = st.number_input("Packed Cell Volume", min_value=20, max_value=60)
    with col3:
        htn = st.selectbox("Hypertension", ["yes", "no"])

    input_data = np.array([sg, al, sc, hemo, pcv, 1 if htn == "yes" else 0]).reshape(1, -1)

    if st.button("Predict Kidney Disease"):
        result = kidney_model.predict(input_data)
        if result[0] == 1:
            st.error("High risk of Kidney Disease 🚨")
        else:
            st.success("No signs of Kidney Disease ✅")

# Liver Disease Form
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

# Parkinson's Disease Form
elif selected_disease == "Parkinson's Disease":
    st.header("🧠 Parkinson's Disease Prediction")

    col1, col2 = st.columns(2)
    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", min_value=50.0, max_value=300.0)
        jitter = st.number_input("MDVP:Jitter(%)", min_value=0.0, max_value=1.0)
        shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=1.0)
    with col2:
        rap = st.number_input("MDVP:RAP", min_value=0.0, max_value=0.1)
        ppq = st.number_input("MDVP:PPQ", min_value=0.0, max_value=0.1)
        dda = st.number_input("MDVP:DDP", min_value=0.0, max_value=0.1)

    input_data = np.array([fo, jitter, shimmer, rap, ppq, dda]).reshape(1, -1)

    if st.button("Predict Parkinson's Disease"):
        result = parkinson_model.predict(input_data)
        if result[0] == 1:
            st.error("Parkinson's Detected 🚨")
        else:
            st.success("No Parkinson's Symptoms ✅")
