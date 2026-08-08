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
st.write("this dashboard predicts the equipment failure risk using machine learning.")

