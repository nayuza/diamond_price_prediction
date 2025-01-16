import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Fungsi untuk memuat model
model = load_model('5073_diamond_price_prediction_model.keras')

# Fungsi untuk skala data
def scale_input(data):
    scaler = MinMaxScaler()
    scaler.fit([[0.2, 0, 0, 0, 50, 50, 3, 3, 2], [5, 4, 6, 7, 75, 80, 10, 10, 7]])
    return scaler.transform(data)

# Judul aplikasi
st.markdown(
    """
    <style>
    .main-title {
        font-size: 35px;
        font-weight: bold;
        color: #007BFF;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-title {
        font-size: 20px;
        color: #333333;
        text-align: center;
        margin-bottom: 40px;
    }
    .footer {
        margin-top: 50px;
        text-align: center;
        font-size: 16px;
        color: #555555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Prediksi Harga Berlian</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Masukkan fitur berlian untuk mendapatkan prediksi harga.</div>',
    unsafe_allow_html=True,
)

# Input untuk fitur berlian
st.write("### Masukkan Fitur Berlian")

col1, col2, col3 = st.columns(3)

with col1:
    carat = st.slider("Berat Karat", 0.2, 5.0, 1.0, 0.1)
    depth = st.slider("Persentase Kedalaman (%)", 50.0, 75.0, 62.0, 0.1)
    x = st.slider("Panjang (mm)", 3.0, 10.0, 5.5, 0.1)

with col2:
    cut = st.selectbox("Kualitas Potongan", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    table = st.slider("Lebar Table (%)", 50.0, 80.0, 57.0, 0.1)
    y = st.slider("Lebar (mm)", 3.0, 10.0, 5.5, 0.1)

with col3:
    color = st.selectbox("Grade Warna", ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
    clarity = st.selectbox("Grade Kejernihan", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    z = st.slider("Kedalaman (mm)", 2.0, 7.0, 3.5, 0.1)

# Mapping kategori ke angka
cut_mapping = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
color_mapping = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity_mapping = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

cut = cut_mapping[cut]
color = color_mapping[color]
clarity = clarity_mapping[clarity]

# Membuat DataFrame dari input
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

st.write("### Data Fitur yang Dimasukkan:")
st.dataframe(input_data)

# Tombol prediksi
if st.button("Prediksi Harga Berlian"):
    try:
        # Skala data input
        scaled_data = scale_input(input_data)
        # Prediksi harga
        prediction = model.predict(scaled_data)
        # Tampilkan hasil prediksi
        st.success(f"Harga prediksi berlian adalah ${prediction[0][0]:,.2f}")
    except Exception as e:
        st.error(f"Error saat melakukan prediksi: {e}")

# Footer
st.markdown(
    """
    <div class="footer">
        Nama: M.Nasyid Yunitian Rizal<br>
        NIM: 22.11.5073<br>
        Aplikasi ini dibuat untuk tugas prediksi harga berlian.
    </div>
    """,
    unsafe_allow_html=True,
)
