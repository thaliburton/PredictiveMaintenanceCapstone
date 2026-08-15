# Predictive Maintenance System for Warehouse Equipment

## Project Overview

This project is a machine learning-based predictive maintenance system designed to help identify equipment failure risk in a warehouse environment.

The system uses equipment runtime, temperature, recent fault history, downtime patterns, preventive maintenance history, sensor readings, and throughput data to predict whether a piece of equipment is likely to fail within the next seven days.

The project also includes a fault classification model that predicts the most likely fault type, such as jam, sensor fault, motor failure, belt issue, overheating, or no fault.

A Streamlit dashboard provides an interactive interface where users can enter equipment conditions and receive a failure risk prediction, probability score, likely fault type, and recommended maintenance action.

---

## Problem Statement

Warehouse conveyor systems, sorters, robotic equipment, induct stations, and divert systems can experience unexpected failures that cause downtime, reduce throughput, and increase maintenance workload.

Unexpected equipment failures can lead to:

- Increased downtime
- Missed maintenance windows
- Reduced operational efficiency
- Higher maintenance costs
- Safety risks from jams or equipment faults

This project addresses the problem by using machine learning to predict potential equipment failures before they occur.

---

## Project Goals

The main goals of this project are to:

- Predict whether equipment is likely to fail within the next seven days
- Classify the most likely fault type
- Provide maintenance recommendations based on model output
- Display equipment data and prediction results in an interactive dashboard
- Demonstrate a practical machine learning solution for predictive maintenance

---

## Features

- Predicts equipment failure risk
- Displays failure probability percentage
- Classifies likely fault type
- Provides recommended maintenance actions
- Shows dataset overview charts
- Displays the full maintenance dataset
- Uses saved machine learning models for dashboard predictions
- Built with Python, Scikit-learn, Pandas, Joblib, and Streamlit

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib
- Jupyter Notebook
- Git and GitHub

---

## Machine Learning Approach

This project uses supervised machine learning.

Two Random Forest classification models are used:

### 1. Failure Prediction Model

The failure prediction model predicts whether equipment is likely to fail within the next seven days.
----------------------------------------------------------------------------------------------------
HOW TO RUN THE PROJECT!

1. Clone the repository
git clone https://github.com/thaliburton/PredictiveMaintenanceCapstone.git
2. Move into the project folder
cd PredictiveMaintenanceCapstone
3. Create a virtual environment
"python -m venv venv"
4. Activate the virtual environment

For Git Bash on Windows:
source venv/Scripts/activate

For PowerShell on Windows:
.\venv\Scripts\Activate.ps1

5. Install required packages
"pip install -r requirements.txt"
6. Generate the dataset if needed
"python generate_data.py"
7. Run the Streamlit dashboard
"python -m streamlit run app/dashboard.py"
----------------------------------------------------------------------------------------------------
Dashboard

The Streamlit dashboard allows users to enter equipment conditions such as runtime, temperature, fault count, downtime, preventive maintenance age, sensor reading, and throughput rate.

The dashboard then displays:

Failure risk level
Failure probability
Likely fault type
Recommended maintenance action
Dataset summary charts
Full maintenance dataset
----------------------------------------------------------------------------------------------------
Author

Tommy Haliburton III

Bachelor of Science in Computer Science
Western Governors University

This project was created as part of a 2026 WGU Computer Science capstone project.
