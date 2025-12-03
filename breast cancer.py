# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ----------------------------
# Load dataset
# ----------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# ----------------------------
# Train-Test Split & Scaling
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Train Model
# ----------------------------
model = SVC(kernel='linear', probability=True)
model.fit(X_train_scaled, y_train)

# ----------------------------
# Streamlit App Interface
# ----------------------------
st.title("Breast Cancer Classification Web App")
st.write("Predict whether a tumor is **Benign** or **Malignant**")

# Create sliders for user input
def user_input_features():
    data_dict = {}
    for feature in X.columns:
        min_val = float(X[feature].min())
        max_val = float(X[feature].max())
        mean_val = float(X[feature].mean())
        data_dict[feature] = st.slider(feature, min_value=min_val, max_value=max_val, value=mean_val)
    features = pd.DataFrame(data_dict, index=[0])
    return features

input_df = user_input_features()

# Scale the user input
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Display Results
st.subheader("Prediction")
tumor_type = "Benign" if prediction[0] == 1 else "Malignant"
st.write(f"The tumor is **{tumor_type}**")

st.subheader("Prediction Probability")
st.write(f"Benign: {prediction_proba[0][1]*100:.2f}%")
st.write(f"Malignant: {prediction_proba[0][0]*100:.2f}%")
