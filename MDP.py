import streamlit as st
import pickle
import numpy as np

# Load models
kidney_model = pickle.load(open('kidney_model.pkl', 'rb'))
liver_model = pickle.load(open('liver_model.pkl', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.pkl', 'rb'))

# Load scaler for Parkinson's (if you have it saved, else skip scaling for now)
scaler = pickle.load(open('parkinsons_scaler.pkl', 'rb'))

st.title("Multiple Disease Prediction System")

tab1, tab2, tab3 = st.tabs(["Kidney Disease", "Parkinson's Disease", "Liver Disease"])

# ----------- Kidney Disease Tab -----------
with tab1:
    st.header("Kidney Disease Prediction")

    age = st.number_input("Age", min_value=0, max_value=120)
    bp = st.number_input("Blood Pressure (mmHg)", min_value=0)
    sg = st.number_input("Specific Gravity", min_value=1.0, max_value=1.05, format="%.3f")
    al = st.number_input("Albumin (0-5 scale)", min_value=0, max_value=5)
    su = st.number_input("Sugar (0-5 scale)", min_value=0, max_value=5)
    rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
    pc = st.selectbox("Pus Cell", ["Normal", "Abnormal"])
    pcc = st.selectbox("Pus Cell Clumps", ["Present", "Not Present"])
    bgr = st.number_input("Blood Glucose Random (mg/dl)", min_value=0)
    sc = st.number_input("Serum Creatinine (mg/dl)", min_value=0.0, format="%.2f")

    if st.button("Predict Kidney Disease"):
        rbc_val = 1 if rbc == "Normal" else 0
        pc_val = 1 if pc == "Normal" else 0
        pcc_val = 1 if pcc == "Present" else 0

        input_data = np.array([[age, bp, sg, al, su, rbc_val, pc_val, pcc_val, bgr, sc]])
        prediction = kidney_model.predict(input_data)

        if prediction[0] == 1:
            st.error("Kidney Disease Detected")
        else:
            st.success("No Kidney Disease Detected")

# ----------- Parkinson's Disease Tab -----------
with tab2:
    st.header("Parkinson's Disease Prediction")

    fo = st.number_input("MDVP:Fo(Hz)", format="%.6f")
    fhi = st.number_input("MDVP:Fhi(Hz)", format="%.6f")
    flo = st.number_input("MDVP:Flo(Hz)", format="%.6f")
    jitter_percent = st.number_input("MDVP:Jitter(%)", format="%.6f")
    jitter_abs = st.number_input("MDVP:Jitter(Abs)", format="%.6f")
    rap = st.number_input("MDVP:RAP", format="%.6f")
    ppq = st.number_input("MDVP:PPQ", format="%.6f")
    ddp = st.number_input("Jitter:DDP", format="%.6f")
    shimmer = st.number_input("MDVP:Shimmer", format="%.6f")
    shimmer_db = st.number_input("MDVP:Shimmer(dB)", format="%.6f")

    if st.button("Predict Parkinson's Disease"):
        input_data = np.array([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db]])
        input_data_scaled = scaler.transform(input_data)  # scale input data as model trained on scaled data
        prediction = parkinsons_model.predict(input_data_scaled)

        if prediction[0] == 1:
            st.error("Parkinson's Disease Detected")
        else:
            st.success("No Parkinson's Disease Detected")

# ----------- Liver Disease Tab -----------
with tab3:
    st.header("Liver Disease Prediction")

    age = st.number_input("Age", min_value=0, max_value=120)
    total_bilirubin = st.number_input("Total Bilirubin", format="%.3f")
    direct_bilirubin = st.number_input("Direct Bilirubin", format="%.3f")
    alkaline_phosphotase = st.number_input("Alkaline Phosphotase")
    alamine_aminotransferase = st.number_input("Alamine Aminotransferase")
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase")
    total_proteins = st.number_input("Total Proteins", format="%.3f")
    albumin = st.number_input("Albumin", format="%.3f")
    albumin_and_globulin_ratio = st.number_input("Albumin and Globulin Ratio", format="%.3f")

    if st.button("Predict Liver Disease"):
        input_data = np.array([[age, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                                alamine_aminotransferase, aspartate_aminotransferase,
                                total_proteins, albumin, albumin_and_globulin_ratio]])
        prediction = liver_model.predict(input_data)

        if prediction[0] == 1:
            st.error("Liver Disease Detected")
        else:
            st.success("No Liver Disease Detected")
