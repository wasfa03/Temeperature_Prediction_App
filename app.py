import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("temperature_rnn_model.keras")
scaler = joblib.load("scaler.save")

# Input section
st.title("Next Day Temperature Prediction")
st.write("Enter the last 30 days' temperatures (one per line):")

temps_input = st.text_area("Temperatures (°C)", placeholder="12.3\n13.5\n14.8\n...")

if st.button("Predict Next Day Temperature"):
    try:
        # Parse input
        temps = [float(t) for t in temps_input.strip().split()]
        if len(temps) != 30:
            st.error("Please enter exactly 30 temperature values.")
        else:
            # Scale and reshape
            temps_scaled = scaler.transform(np.array(temps).reshape(-1, 1))
            input_seq = temps_scaled.reshape(1, 30, 1)

            # Predict and inverse transform
            pred_scaled = model.predict(input_seq)
            pred_temp = scaler.inverse_transform(pred_scaled)

            st.success(f"Predicted Next Day Temperature: {pred_temp[0][0]:.2f} °C")
    except Exception as e:
        st.error(f"Error: {e}")
