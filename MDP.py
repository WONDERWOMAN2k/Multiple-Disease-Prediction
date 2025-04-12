
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page config
st.set_page_config(page_title="Multiple Disease Prediction", layout="wide")

# Function for Preprocessing and Analysis
def preprocess_and_train(data, target_column, dataset_name):
    st.subheader(f"📊 Dataset Preview - {dataset_name}")
    st.write(data.head())

    # Show column names and missing values
    st.text(f"Columns in {dataset_name} dataset: {data.columns.tolist()}")
    st.write("Missing Values:")
    st.write(data.isnull().sum())

    # Handle missing values
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

    # Handle categorical columns
    non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
    label_encoder = LabelEncoder()
    for column in non_numeric_columns:
        data[column] = label_encoder.fit_transform(data[column])

    # Target check
    if target_column not in data.columns:
        st.error(f"Target column '{target_column}' not found in {dataset_name}.")
        return None

    # Features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    st.success(f"✅ Accuracy for {dataset_name}: {accuracy * 100:.2f}%")

    # Confusion matrix
    st.subheader(f"🔍 Confusion Matrix - {dataset_name}")
    fig1, ax1 = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig1)

    # Feature importance
    st.subheader(f"🌟 Feature Importance - {dataset_name}")
    feature_importances = model.feature_importances_
    features = X.columns
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=features)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    st.pyplot(fig2)

    return model


# Streamlit UI
st.title("🧠 Multiple Disease Prediction System")
st.markdown("Upload your datasets below for **Parkinson's**, **Liver Disease**, and **Kidney Disease** prediction.")

# Upload datasets
parkinsons_file = st.file_uploader("📤 Upload Parkinson's Disease CSV", type=["csv"])
liver_file = st.file_uploader("📤 Upload Liver Disease CSV", type=["csv"])
kidney_file = st.file_uploader("📤 Upload Kidney Disease CSV", type=["csv"])

if parkinsons_file:
    parkinsons_data = pd.read_csv(parkinsons_file)
    parkinsons_model = preprocess_and_train(parkinsons_data, target_column="status", dataset_name="Parkinson's Disease")

if liver_file:
    liver_data = pd.read_csv(liver_file)
    liver_model = preprocess_and_train(liver_data, target_column="Dataset", dataset_name="Liver Disease")

if kidney_file:
    kidney_data = pd.read_csv(kidney_file)
    kidney_model = preprocess_and_train(kidney_data, target_column="classification", dataset_name="Kidney Disease")
