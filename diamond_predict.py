import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import requests
from sklearn.preprocessing import MinMaxScaler

model = load_model('5073_diamond_price_prediction_model.keras')

# Sidebar Header
st.sidebar.header("Fitur Berlian")
st.sidebar.write("Atur slider untuk menentukan fitur berlian yang akan diprediksi.")

# Sidebar Inputs
carat = st.sidebar.slider("Berat Karat", 0.2, 5.0, 1.0, 0.1)
cut = st.sidebar.selectbox("Kualitas Potongan", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.sidebar.selectbox("Grade Warna", ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
clarity = st.sidebar.selectbox("Grade Kejernihan", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
depth = st.sidebar.slider("Persentase Kedalaman (%)", 50.0, 75.0, 62.0, 0.1)
table = st.sidebar.slider("Lebar Table (%)", 50.0, 80.0, 57.0, 0.1)
x = st.sidebar.slider("Panjang (mm)", 3.0, 10.0, 5.5, 0.1)
y = st.sidebar.slider("Lebar (mm)", 3.0, 10.0, 5.5, 0.1)
z = st.sidebar.slider("Kedalaman (mm)", 2.0, 7.0, 3.5, 0.1)

# Map categorical inputs to numeric values
cut_mapping = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
color_mapping = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity_mapping = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

cut = cut_mapping[cut]
color = color_mapping[color]
clarity = clarity_mapping[clarity]

# Input Data for Prediction
input_data = pd.DataFrame({
    'carat': [carat],
    'cut': [cut],
    'color': [color],
    'clarity': [clarity],
    'depth': [depth],
    'table': [table],
    'x': [x],
    'y': [y],
    'z': [z]
})

# Scale the input data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(input_data)

# Prediction
if st.sidebar.button("Prediksi Harga Berlian"):
    prediction = model.predict(scaled_data)
    st.success(f"Harga prediksi berlian adalah ${prediction[0][0]:,.2f}")

# Main UI
st.title("Aplikasi Prediksi Harga Berlian")
st.write("Gunakan aplikasi ini untuk memprediksi harga berlian berdasarkan fitur-fiturnya.")
st.write("### Fitur yang Dipilih:")
st.json(input_data.to_dict(orient='records')[0])
