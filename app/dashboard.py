import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🛠️",
    layout="wide"
)

st.title("Predictive Maintenance Dashboard")
st.write("This dashboard predicts equipment failure risk and likely fault type using machine learning.")

# File paths
FAILURE_MODEL_PATH = "models/failure_prediction_model.pkl"
FAILURE_FEATURES_PATH = "models/failure_model_features.pkl"

FAULT_MODEL_PATH = "models/fault_classification_model.pkl"
FAULT_FEATURES_PATH = "models/fault_model_features.pkl"

DATA_PATH = "data/maintenance_data.csv"

# Load models, feature lists, and dataset
failure_model = joblib.load(FAILURE_MODEL_PATH)
failure_features = joblib.load(FAILURE_FEATURES_PATH)

fault_model = joblib.load(FAULT_MODEL_PATH)
fault_features = joblib.load(FAULT_FEATURES_PATH)

data = pd.read_csv(DATA_PATH)

st.success("Models and dataset loaded successfully!")

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

if st.button("Predict Maintenance Risk"):
    input_data = pd.DataFrame([{
        "runtime_hours": runtime_hours,
        "temperature": temperature,
        "fault_count_30_days": fault_count_30_days,
        "downtime_minutes_30_days": downtime_minutes_30_days,
        "days_since_last_pm": days_since_last_pm,
        "sensor_reading": sensor_reading,
        "throughput_rate": throughput_rate
    }])

    failure_input = input_data[failure_features]
    fault_input = input_data[fault_features]

    failure_prediction = failure_model.predict(failure_input)[0]
    failure_probability = failure_model.predict_proba(failure_input)[0][1]

    fault_prediction = fault_model.predict(fault_input)[0]

    st.subheader("Prediction Results")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        if failure_prediction == 1:
            st.error("Failure Risk: HIGH")
        else:
            st.success("Failure Risk: LOW")

    with result_col2:
        st.metric(
            label="Failure Probability",
            value=f"{round(failure_probability * 100, 2)}%"
        )

    with result_col3:
        st.info(f"Likely Fault Type: {fault_prediction}")

    st.subheader("Recommended Maintenance Action")

    if fault_prediction == "jam":
        st.write("Inspect conveyor flow, tote alignment, transfer points, and jam-prone areas.")
    elif fault_prediction == "sensor_fault":
        st.write("Check photo-eyes, sensor alignment, cabling, sensor cleanliness, and sensor mounting.")
    elif fault_prediction == "motor_failure":
        st.write("Inspect motor current, drive faults, bearings, wiring, and mechanical load.")
    elif fault_prediction == "belt_issue":
        st.write("Inspect belt tracking, belt tension, rollers, worn belt sections, and belt damage.")
    elif fault_prediction == "overheating":
        st.write("Check motor temperature, airflow, equipment load, lubrication, and cooling conditions.")
    else:
        st.write("Continue normal monitoring. No immediate fault-specific action is recommended.")

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
st.write(f"Total Records: {len(data)}")
st.dataframe(data, use_container_width=True)