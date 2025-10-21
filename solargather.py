import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ------------------------
# CONFIGURATION
# ------------------------
start_date = datetime(2025, 5, 30, 0, 0, 0)
interval = timedelta(minutes=5)
total_entries = 288 * 42  # 12,096 entries (5 min × 42 days)
timestamps = [start_date + i * interval for i in range(total_entries)]

# ------------------------
# SIMULATION FUNCTIONS
# ------------------------
def simulate_temperature(hour):
    # Hotter at noon
    return 26 + 8 * np.sin((hour - 6) * np.pi / 12) + np.random.normal(0, 0.4)

def simulate_humidity(hour):
    # More humid at night
    return 65 + 15 * np.cos((hour - 6) * np.pi / 12) + np.random.normal(0, 1.5)

def simulate_solar_irradiance(hour, day_type):
    if 6 <= hour <= 18:
        multiplier = 1.0 if day_type == 1 else 0.5
        irradiance = np.sin((hour - 6) * np.pi / 12) * 1000 * multiplier
        return max(irradiance + np.random.normal(0, 30), 0)
    return 0.0

def simulate_solar_voltage(irradiance):
    if irradiance <= 0:
        return 0.0
    return 16.5 + (irradiance / 1000) * 5 + np.random.normal(0, 0.3)

def simulate_solar_current(irradiance):
    if irradiance <= 0:
        return 0.0
    return (irradiance / 1000) * 4.5 + np.random.normal(0, 0.05)

def simulate_load_power(hour):
    # Load varies (night: fan+lights; day: sensors)
    if 18 <= hour or hour < 6:
        base = 10.0  # W
    else:
        base = 7.0  # W
    return base + np.random.normal(0, 0.5)

def simulate_battery_voltage(prev_v, irradiance, load_w):
    if irradiance > 200:
        charge = (irradiance / 1000) * 0.004
        discharge = 0.001 * (load_w / 10)
    else:
        charge = 0
        discharge = 0.002 * (load_w / 10)
    new_v = prev_v + charge - discharge
    return np.clip(new_v, 11.5, 13.2)

def simulate_battery_current(irradiance, load_w):
    # Roughly estimate current flow: +charge current, -discharge current
    if irradiance > 200:
        return 0.5 + np.random.normal(0, 0.05)
    else:
        return -0.8 + np.random.normal(0, 0.05)

# ------------------------
# DATA GENERATION
# ------------------------
rows = []
battery_v = 12.4

for t in timestamps:
    hour = t.hour
    day_type = np.random.choice([0, 1], p=[0.3, 0.7])  # mostly sunny
    temp = simulate_temperature(hour)
    hum = simulate_humidity(hour)
    irr = simulate_solar_irradiance(hour, day_type)
    v_solar = simulate_solar_voltage(irr)
    i_solar = simulate_solar_current(irr)
    p_load = simulate_load_power(hour)
    battery_v = simulate_battery_voltage(battery_v, irr, p_load)
    i_batt = simulate_battery_current(irr, p_load)

    rows.append([
        t.strftime("%Y-%m-%d %H:%M"),
        round(temp, 2),
        round(hum, 2),
        round(v_solar, 2),
        round(i_solar, 2),
        round(irr, 2),
        round(battery_v, 2),
        round(i_batt, 2),
        round(p_load, 2),
        hour,
        day_type
    ])

# ------------------------
# SAVE TO SQLITE
# ------------------------
df = pd.DataFrame(rows, columns=[
    "Timestamp", "Temperature (°C)", "Humidity (%)", "Solar Voltage (V)",
    "Solar Current (A)", "Solar Irradiance (Lux)", "Battery Voltage (V)",
    "Battery Current (A)", "Power Output (W)", "Time of Day (hour,0–23)",
    "Day Type (0=Cloudy,1=Sunny)"
])

conn = sqlite3.connect("solar_data_collection.db")
df.to_sql("solar_data", conn, if_exists="replace", index=False)
conn.close()

print("✅ Dataset successfully saved to 'solar_data_collection.db' with", len(df), "records.")
print(df.head(10))
