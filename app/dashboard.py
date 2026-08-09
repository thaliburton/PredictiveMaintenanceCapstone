import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title = "Predictive Maintenance Dashboard",
    page_icon = "🛠",
    layout = "wide",
)

st.title("Predictive Maintenance Dashboard")
st.write("This dashboard predicts the equipment failure risk using machine learning.")

MODEL_PATH = "models/failure_prediction_model.pkl"
FEATURES_PATH = "models/failure_model_features.pkl"
DATA_PATH = "data/maintenance_data.csv"

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)
data = pd.read_csv(DATA_PATH)

st.success("model and dataset loaded successfully!")

st.header("Enter Equipment Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    runtime_hours = st.number_input(
        "Runtime Hours",
        min_value=0,
        max_value=5000,
        value=1500
    )

    temperature = st.number_input(
        "Temperature",
        min_value=0.0,
        max_value=200.0,
        value=85.0
    )

    fault_count_30_days = st.number_input(
        "Fault Count Last 30 Days",
        min_value=0,
        max_value=50,
        value=3
    )

with col2:
    downtime_minutes_30_days = st.number_input(
        "Downtime Minutes Last 30 Days",
        min_value=0,
        max_value=1000,
        value=120
    )

    days_since_last_pm = st.number_input(
        "Days Since Last PM",
        min_value=0,
        max_value=365,
        value=30
    )

with col3:
    sensor_reading = st.number_input(
        "Sensor Reading",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    throughput_rate = st.number_input(
        "Throughput Rate",
        min_value=0,
        max_value=5000,
        value=1500
    )

    st.divider()

if st.button("Predict Failure Risk"):
    input_data = pd.DataFrame([{
        "runtime_hours": runtime_hours,
        "temperature": temperature,
        "fault_count_30_days": fault_count_30_days,
        "downtime_minutes_30_days": downtime_minutes_30_days,
        "days_since_last_pm": days_since_last_pm,
        "sensor_reading": sensor_reading,
        "throughput_rate": throughput_rate
    }])

    input_data = input_data[features]

    prediction = model.predict(input_data)[0]
    prediction_probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("Failure Risk: HIGH")
        st.write("Recommended Action: Schedule maintenance as soon as possible.")
    else:
        st.success("Failure Risk: LOW")
        st.write("Recommended Action: Continue normal monitoring.")

    st.write("Failure Probability:", round(prediction_probability * 100, 2), "%")

    st.divider()

st.header("Dataset Overview")

st.subheader("Failure Count")
failure_counts = data["failure_next_7_days"].value_counts()
st.bar_chart(failure_counts)

st.subheader("Average Temperature by Failure Status")
avg_temp = data.groupby("failure_next_7_days")["temperature"].mean()
st.bar_chart(avg_temp)

st.subheader("Fault Type Count")
fault_counts = data["fault_type"].value_counts()
st.bar_chart(fault_counts)

st.subheader("Full Dataset")
st.dataframe(data, use_container_width=True)
