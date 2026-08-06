import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

num_records = 10

equipment_types = [
    "conveyor",
    "sorter",
    "robot",
    "induct station",
    "divert"
]

fault_types = [
    "none",
    "jam",
    "sensor_fault",
    "motor_fault",
    "belt_issues",
    "overheating",
]

data = []

for i in range(1, num_records + 1):
    equipment_type = random.choice(equipment_types)

    runtime_hours = np.random.randint(50, 3000)
    temperature = np.random.normal(78, 12)
    fault_count_30_days = np.random.poisson(3)
    downtime_minutes_30_days = np.random.randint(0, 400)
    days_since_last_pm = np.random.randint(1, 120)
    sensor_reading = np.random.normal(50, 15)
    throughput_rate = np.random.randint(500, 2500)

    risk_score = (0)

    if runtime_hours > 1800:
        risk_score += 1

    if temperature > 90:
        risk_score += 1

    if fault_count_30_days > 5:
        risk_score += 1

    if downtime_minutes_30_days > 180:
        risk_score += 1

    if days_since_last_pm > 60:
        risk_score += 1

    if sensor_reading > 70 or sensor_reading < 25:
        risk_score += 1

    if risk_score >= 3:
        failure_next_7_days = 1
    else:
        failure_next_7_days = 0

    if failure_next_7_days == 0:
        fault_type = "none"
    else:
        fault_type = random.choice(fault_types[1:])

    equipment_id = f"EQ_{i:04d}"

    data.append({
        "equipment_id": equipment_id,
        "equipment_type": equipment_type,
        "runtime_hours": runtime_hours,
        "temperature": round(temperature, 2),
        "fault_count_30_days": fault_count_30_days,
        "downtime_minutes_30_days": downtime_minutes_30_days,
        "days_since_last_pm": days_since_last_pm,
        "sensor_reading": round(sensor_reading, 2),
        "throughput_rate": throughput_rate,
        "failure_next_7_days": failure_next_7_days,
        "fault_type": fault_type
    })

df = pd.DataFrame(data)

os.makedirs('data', exist_ok=True)

df.to_csv('data/maintenance_data.csv', index=False)

print("Dataset created successfully!")
print("File saved to: data/maintenance_data.csv")
print(df.head())
