import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# --- Title ---
st.title("🌡️ Temperature Predictor")
st.markdown("Predict the next day's temperature using an RNN model.")

# --- Load the model and scaler ---
@st.cache_resource
def load_model_and_scaler():
    model = load_model("temperature_rnn_model.keras")  # New Keras format
    scaler = joblib.load("temperature_scaler.save")
    return model, scaler

model, scaler = load_model_and_scaler()

# --- Load the dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("daily_minimum_temps.csv", parse_dates=["Date"], index_col="Date")
    df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")
    df.dropna(inplace=True)
    return df

df = load_data()

# --- Show recent temperature data ---
st.subheader("📈 Recent Temperature Data")
st.line_chart(df["Temp"].tail(100))

# --- Normalize and prepare last sequence ---
SEQ_LENGTH = 30
data_scaled = scaler.transform(df["Temp"].values.reshape(-1, 1))
last_sequence = data_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)

# --- Predict next temperature ---
pred_scaled = model.predict(last_sequence)
pred_scaled = np.clip(pred_scaled, 0, 1)
next_temp = scaler.inverse_transform(pred_scaled)

# --- Show prediction ---
st.subheader("🔮 Predicted Temperature for Next Day:")
st.success(f"{next_temp[0][0]:.2f} °C")
