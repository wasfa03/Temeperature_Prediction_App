import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Temperature Prediction", layout="wide")

# App title
st.title("Daily Minimum Temperature Prediction using RNN")

# Load saved model and scaler
@st.cache_resource
def load_saved_components():
    try:
        model = load_model("temperature_rnn_model.h5")
        scaler = joblib.load("temperature_scaler.save")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        return None, None

model, scaler = load_saved_components()

# Load the dataset
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("daily_minimum_temps.csv", parse_dates=["Date"], index_col="Date")
        df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

df = load_dataset()

# Sidebar for user inputs
with st.sidebar:
    st.header("Prediction Settings")
    seq_length = st.number_input("Sequence Length", min_value=10, max_value=60, value=30, 
                               help="Number of previous days to use for prediction")
    
    if model is None:
        st.warning("No model loaded. Please check if model files exist.")
    else:
        st.success("Model and scaler loaded successfully!")
    
    if df is not None:
        st.info(f"Dataset loaded with {len(df)} records")

if df is not None and model is not None and scaler is not None:
    try:
        # Normalize the data using the loaded scaler
        data_scaled = scaler.transform(df["Temp"].values.reshape(-1, 1))
        
        # Function for creating sequences
        def create_sequences(data_scaled, seq_length):
            X, y = [], []
            for i in range(len(data_scaled) - seq_length):
                X.append(data_scaled[i:i + seq_length])
                y.append(data_scaled[i + seq_length])
            return np.array(X), np.array(y)
        
        # Create sequences
        X, y = create_sequences(data_scaled, seq_length)
        
        # Make predictions
        with st.spinner("Making predictions..."):
            y_pred_scaled = model.predict(X)
            y_pred_scaled = np.clip(y_pred_scaled, 0, 1)
            y_pred = scaler.inverse_transform(y_pred_scaled)
            y_actual = scaler.inverse_transform(y)
            
            # Predict next day temperature
            last_sequence = data_scaled[-seq_length:].reshape(1, seq_length, 1)
            next_temp_scaled = model.predict(last_sequence)
            next_temp_scaled = np.clip(next_temp_scaled, 0, 1)
            next_day_temp = scaler.inverse_transform(next_temp_scaled)
            
            # Get the last date from the data
            last_date = df.index[-1]
            next_date = last_date + timedelta(days=1)
        
        # Display results
        st.success("Predictions completed successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Next Day Prediction")
            st.metric(label=f"Predicted Temperature for {next_date.strftime('%Y-%m-%d')}", 
                     value=f"{next_day_temp[0][0]:.2f}°C")
            
            # Show recent temperatures
            st.subheader("Recent Temperatures")
            st.dataframe(df.tail(10))
            
            # Show dataset statistics
            with st.expander("Dataset Statistics"):
                st.write(df.describe())
                fig_dist = plt.figure(figsize=(8, 4))
                plt.hist(df["Temp"], bins=30)
                plt.title("Temperature Distribution")
                plt.xlabel("Temperature (°C)")
                plt.ylabel("Frequency")
                st.pyplot(fig_dist)
        
        with col2:
            st.subheader("Actual vs Predicted Values")
            fig_pred = plt.figure(figsize=(10, 5))
            plt.plot(y_actual, label="Actual Temperature")
            plt.plot(y_pred, label="Predicted Temperature")
            plt.title("Actual vs Predicted Temperatures")
            plt.xlabel("Time Step")
            plt.ylabel("Temperature (°C)")
            plt.legend()
            st.pyplot(fig_pred)
        
            # Show prediction statistics
            with st.expander("Prediction Performance"):
                errors = y_actual.flatten() - y_pred.flatten()
                mae = np.mean(np.abs(errors))
                rmse = np.sqrt(np.mean(errors**2))
                
                st.metric("Mean Absolute Error", f"{mae:.2f}°C")
                st.metric("Root Mean Square Error", f"{rmse:.2f}°C")
                
                fig_error = plt.figure(figsize=(8, 4))
                plt.hist(errors, bins=30)
                plt.title("Prediction Error Distribution")
                plt.xlabel("Error (°C)")
                plt.ylabel("Frequency")
                st.pyplot(fig_error)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
elif model is None or scaler is None:
    st.error("Required model files not found. Please ensure these files are present:")
    st.markdown("""
    - `temperature_rnn_model.h5` - The saved Keras model
    - `temperature_scaler.save` - The saved scaler object
    """)
elif df is None:
    st.error("Could not load the dataset. Please ensure 'daily_minimum_temps.csv' exists in the directory.")