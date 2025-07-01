
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load('iris_model.pkl')

st.title("🌼 Iris Flower Classifier")
st.write("Enter flower measurements:")

# Inputs
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)
probability = model.predict_proba(input_data)

classes = ['Setosa', 'Versicolor', 'Virginica']
st.subheader("Prediction")
st.success(f"Predicted species: {classes[prediction[0]]}")

st.subheader("Probabilities")
st.bar_chart(probability[0])
