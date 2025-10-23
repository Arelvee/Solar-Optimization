import os
import sqlite3
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import tensorflow as tf
import joblib
from typing import Dict, List, Tuple

# ------------------------
# CONFIGURATION
# ------------------------
# OpenWeatherMap API Configuration
API_KEY = os.getenv("OPENWEATHER_API_KEY", "")  # Set your API key here or in environment
LOCATIONS = {
    "Tanauan": {"lat": 14.0833, "lon": 121.1500},
    "LSPU_San_Pablo": {"lat": 14.0689, "lon": 121.3256}
}

# Solar Panel and System Configuration
SOLAR_SPECS = {
    "max_voltage": 21.5,  # Maximum voltage output
    "max_current": 5.5,   # Maximum current output
    "temp_coefficient": -0.004,  # Temperature coefficient (%/°C)
    "nominal_temp": 25,   # Nominal operating temperature
}

# Threshold Configuration
THRESHOLDS = {
    "irradiance": {
        "low": 200,    # W/m²
        "medium": 600  # W/m²
        # Above medium is considered high
    },
    "temperature": {
        "optimal_range": (20, 30),  # °C
        "critical_high": 45         # °C
    },
    "humidity": {
        "high": 80    # %
    }
}

# ------------------------
# WEATHER API FUNCTIONS
# ------------------------
def fetch_weather_data(lat: float, lon: float, api_key: str) -> Dict:
    """Fetch current weather data from OpenWeatherMap API."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "metric"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

def get_solar_irradiance(weather_data: Dict) -> float:
    """Calculate solar irradiance from weather data."""
    if not weather_data:
        return 0.0
    
    clouds = weather_data.get("clouds", {}).get("all", 0)
    # Estimate irradiance based on cloud coverage
    clear_sky_irradiance = 1000  # Maximum clear sky irradiance
    cloud_factor = 1 - (clouds / 100) * 0.7  # Cloud impact factor
    return clear_sky_irradiance * cloud_factor

# ------------------------
# SOLAR OUTPUT SIMULATION
# ------------------------
def calculate_solar_output(temp: float, humidity: float, irradiance: float) -> Tuple[float, float, float]:
    """Calculate solar panel output based on environmental conditions."""
    
    # Temperature effect
    temp_diff = temp - SOLAR_SPECS["nominal_temp"]
    temp_factor = 1 + (SOLAR_SPECS["temp_coefficient"] * temp_diff)
    
    # Humidity effect (high humidity reduces efficiency)
    humidity_factor = 1 - max(0, (humidity - THRESHOLDS["humidity"]["high"]) / 100 * 0.1)
    
    # Calculate voltage
    base_voltage = SOLAR_SPECS["max_voltage"] * (irradiance / 1000)
    voltage = base_voltage * temp_factor * humidity_factor
    voltage = np.clip(voltage, 0, SOLAR_SPECS["max_voltage"])
    
    # Calculate current
    base_current = SOLAR_SPECS["max_current"] * (irradiance / 1000)
    current = base_current * temp_factor * humidity_factor
    current = np.clip(current, 0, SOLAR_SPECS["max_current"])
    
    # Calculate power
    power = voltage * current
    
    return voltage, current, power

def determine_efficiency_class(power: float, max_power: float) -> int:
    """Determine efficiency class (0=Low, 1=Medium, 2=High)."""
    efficiency = power / max_power
    if efficiency < 0.3:
        return 0
    elif efficiency < 0.7:
        return 1
    return 2

# ------------------------
# LOAD MANAGEMENT
# ------------------------
class LoadManagement:
    def __init__(self):
        self.load_thresholds = {
            "critical": 100,  # W
            "warning": 200    # W
        }
    
    def get_load_recommendation(self, battery_voltage: float, power_output: float) -> Dict:
        """Get load management recommendation."""
        if battery_voltage < 11.8:  # Critical battery level
            return {
                "status": "CRITICAL",
                "action": "Turn off all non-essential loads",
                "max_load": self.load_thresholds["critical"]
            }
        elif power_output < self.load_thresholds["warning"]:
            return {
                "status": "WARNING",
                "action": "Reduce load consumption",
                "max_load": self.load_thresholds["warning"]
            }
        return {
            "status": "NORMAL",
            "action": "Normal operation",
            "max_load": None
        }

# ------------------------
# DATA COLLECTION AND STORAGE
# ------------------------
def collect_and_store_data(location_name: str, lat: float, lon: float, load_manager: LoadManagement):
    """Collect real weather data and store with simulated solar output."""
    
    # Get real weather data
    weather_data = fetch_weather_data(lat, lon, API_KEY)
    if not weather_data:
        print(f"Failed to get weather data for {location_name}")
        return
    
    # Extract weather parameters
    temp = weather_data["main"]["temp"]
    humidity = weather_data["main"]["humidity"]
    irradiance = get_solar_irradiance(weather_data)
    
    # Calculate solar output
    voltage, current, power = calculate_solar_output(temp, humidity, irradiance)
    
    # Determine efficiency class
    max_power = SOLAR_SPECS["max_voltage"] * SOLAR_SPECS["max_current"]
    efficiency_class = determine_efficiency_class(power, max_power)
    
    # Get load management recommendation
    battery_voltage = 12.4  # This should be from real measurement or better simulation
    load_rec = load_manager.get_load_recommendation(battery_voltage, power)
    
    # Prepare data for storage
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "Timestamp": timestamp,
        "Temperature": round(temp, 2),
        "Humidity": round(humidity, 2),
        "Solar_Irradiance": round(irradiance, 2),
        "Solar_Voltage": round(voltage, 2),
        "Solar_Current": round(current, 2),
        "Power_Output": round(power, 2),
        "Efficiency_Class": efficiency_class,
        "Load_Status": load_rec["status"],
        "Load_Action": load_rec["action"],
        "Max_Load": load_rec["max_load"]
    }
    
    # Store in SQLite
    db_name = f"solar_data_{location_name.lower()}.db"
    conn = sqlite3.connect(db_name)
    df = pd.DataFrame([data])
    df.to_sql("solar_monitoring", conn, if_exists="append", index=False)
    conn.close()
    
    return data

# ------------------------
# MAIN EXECUTION
# ------------------------
def main():
    if not API_KEY:
        print("Error: OpenWeatherMap API key not set!")
        return
    
    load_manager = LoadManagement()
    
    # Collect data for each location
    for location_name, coords in LOCATIONS.items():
        print(f"\nCollecting data for {location_name}...")
        data = collect_and_store_data(
            location_name,
            coords["lat"],
            coords["lon"],
            load_manager
        )
        
        if data:
            print(f"✅ {location_name} Data Collected:")
            print(f"Temperature: {data['Temperature']}°C")
            print(f"Humidity: {data['Humidity']}%")
            print(f"Solar Irradiance: {data['Solar_Irradiance']} W/m²")
            print(f"Power Output: {data['Power_Output']} W")
            print(f"Efficiency Class: {['Low', 'Medium', 'High'][data['Efficiency_Class']]}")
            print(f"Load Status: {data['Load_Status']}")
            print(f"Recommended Action: {data['Load_Action']}")

if __name__ == "__main__":
    main()
