import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("SmartRail — Compressor Failure Risk Predictor")
st.write("Enter current sensor readings to estimate failure risk")
st.caption("Note: assumes steady-state readings (no recent history) for simplicity.")

col1, col2 = st.columns(2)

with col1:
    tp2 = st.number_input("TP2 (Pressure)", value=0.0)
    tp3 = st.number_input("TP3 (Pressure)", value=9.0)
    h1 = st.number_input("H1 (Pressure)", value=9.0)
    dv_pressure = st.number_input("DV Pressure", value=0.0)

with col2:
    reservoirs = st.number_input("Reservoirs", value=9.0)
    oil_temp = st.number_input("Oil Temperature", value=60.0)
    motor_current = st.number_input("Motor Current", value=2.0)


if st.button("Predict"):

    # Load saved models
    scaler2 = joblib.load("models/scaler2.pkl")
    stack_model = joblib.load("models/stacking_model.pkl")
    kmeans = joblib.load("models/kmeans.pkl")

    # Sensor values
    sensors = {
        "TP2": tp2,
        "TP3": tp3,
        "H1": h1,
        "DV_pressure": dv_pressure,
        "Reservoirs": reservoirs,
        "Oil_temperature": oil_temp,
        "Motor_current": motor_current
    }

    # Create engineered features
    features = {}

    for name, val in sensors.items():
        features[f"{name}_roll_mean"] = val
        features[f"{name}_roll_std"] = 0.0
        features[f"{name}_roll_min"] = val
        features[f"{name}_roll_max"] = val

    for name, val in sensors.items():
        features[f"{name}_diff"] = 0.0
        features[f"{name}_lag1"] = val

    # Feature order used during training
    feature_order = [
        "TP2_roll_mean",
        "TP2_roll_std",
        "TP2_roll_min",
        "TP2_roll_max",

        "TP3_roll_mean",
        "TP3_roll_std",
        "TP3_roll_min",
        "TP3_roll_max",

        "H1_roll_mean",
        "H1_roll_std",
        "H1_roll_min",
        "H1_roll_max",

        "DV_pressure_roll_mean",
        "DV_pressure_roll_std",
        "DV_pressure_roll_min",
        "DV_pressure_roll_max",

        "Reservoirs_roll_mean",
        "Reservoirs_roll_std",
        "Reservoirs_roll_min",
        "Reservoirs_roll_max",

        "Oil_temperature_roll_mean",
        "Oil_temperature_roll_std",
        "Oil_temperature_roll_min",
        "Oil_temperature_roll_max",

        "Motor_current_roll_mean",
        "Motor_current_roll_std",
        "Motor_current_roll_min",
        "Motor_current_roll_max",

        "TP2_diff",
        "TP2_lag1",
        "TP3_diff",
        "TP3_lag1",
        "H1_diff",
        "H1_lag1",

        "DV_pressure_diff",
        "DV_pressure_lag1",
        "Reservoirs_diff",
        "Reservoirs_lag1",

        "Oil_temperature_diff",
        "Oil_temperature_lag1",
        "Motor_current_diff",
        "Motor_current_lag1"
    ]

    # Temporary values for the additional training features
    features["kmeans_cluster"] = 0
    features["pca_distance_score"] = 0.0

    # Final feature order
    final_order = feature_order + [
        "kmeans_cluster",
        "pca_distance_score"
    ]

    # Create input DataFrame
    X_row = pd.DataFrame(
        [[features[c] for c in final_order]],
        columns=final_order
    )

    # Scale exactly the same features used during training
    X_scaled = scaler2.transform(X_row)

    # Predict failure probability
    prob = stack_model.predict_proba(X_scaled)[0][1]

    # Display result
    st.metric("Failure Risk", f"{prob * 100:.1f}%")

    if prob > 0.5:
        st.error("⚠️ High risk — inspection recommended")
    else:
        st.success("✅ Normal operating range")