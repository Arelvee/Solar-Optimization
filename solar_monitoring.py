import os
import sqlite3
import pandas as pd
import numpy as np
import requests
import tensorflow as tf
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Union, Any
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod

# ------------------------
# CONFIGURATION
# ------------------------
# API Keys (set these in environment variables)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
NREL_API_KEY = os.getenv("NREL_API_KEY", "")  # Get from https://developer.nrel.gov/signup/

# PV System Configuration for PVWatts API
PV_SYSTEM_CONFIG = {
    "system_capacity": 0.1,  # 100W system size (0.1 kW)
    "module_type": 1,        # 0=Standard, 1=Premium, 2=Thin film
    "array_type": 1,         # 0=Fixed open rack, 1=Fixed roof mount, 2=1-axis tracking
    "tilt": 14,             # Tilt angle for Philippines (near latitude)
    "azimuth": 180,         # South-facing
    "losses": 14.08         # Default losses (soiling, shading, etc)
}

LOCATIONS = {
    "Sto_Tomas": {
        "lat": 14.1119, 
        "lon": 121.1483,
        "altitude": 300  # meters above sea level
    },
    "LSPU_San_Pablo": {
        "lat": 14.0689, 
        "lon": 121.3256,
        "altitude": 450
    },
    "San_Pablo_City": {
        "lat": 14.0777, 
        "lon": 121.3257,
        "altitude": 430
    }
}

# Solar System Specifications
SOLAR_SPECS = {
    "max_voltage": 21.5,
    "max_current": 5.5,
    "temp_coefficient": -0.0040,  # Power reduction per °C above STC (25°C)
    "nominal_temp": 25.0,
    "max_power": 118.25  # Watts (Vmax * Imax)
}

# Alert Thresholds
ALERT_THRESHOLDS = {
    "temperature": {
        "critical": 45.0,  # °C
        "warning": 40.0
    },
    "humidity": {
        "high": 85.0,  # %
        "warning": 75.0
    },
    "power": {
        "low": SOLAR_SPECS["max_power"] * 0.3,
        "medium": SOLAR_SPECS["max_power"] * 0.7
    },
    "battery": {
        "critical": 11.2,  # V
        "warning": 11.8
    }
}

@dataclass
class Alert:
    timestamp: str
    location: str
    type: str
    severity: str
    message: str

class AlertSystem:
    def __init__(self):
        self.alerts: List[Alert] = []
        
    def clear_alerts(self) -> None:
        self.alerts = []
    
    def add_alert(self, location: str, alert_type: str, severity: str, message: str) -> None:
        alert = Alert(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            location=location,
            type=alert_type,
            severity=severity,
            message=message
        )
        self.alerts.append(alert)
    
    def save_alerts(self, location: str) -> None:
        if self.alerts:
            try:
                db_name = f"solar_alerts_{location.lower()}.db"
                conn = sqlite3.connect(db_name)
                df = pd.DataFrame([vars(alert) for alert in self.alerts])
                df.to_sql("alerts", conn, if_exists="append", index=False)
                conn.close()
            except Exception as e:
                print(f"Error saving alerts for {location}: {e}")

class WeatherMonitor:
    def __init__(self, openweather_key: str, nrel_key: str):
        self.openweather_key = openweather_key
        self.nrel_key = nrel_key
        self.weather_cache = {}
        self.solar_cache = {}
        
    def get_weather(self, lat: float, lon: float) -> Dict:
        """Get comprehensive weather data from OpenWeatherMap."""
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.openweather_key,
                "units": "metric"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching OpenWeather data: {e}")
            return None

    def get_nrel_solar_data(self, lat: float, lon: float, altitude: float) -> Dict:
        """Get detailed solar radiation data from NREL."""
        cache_key = f"{lat},{lon}"
        if cache_key in self.solar_cache:
            if (datetime.now() - self.solar_cache[cache_key]["timestamp"]).seconds < 3600:
                return self.solar_cache[cache_key]["data"]
        
        try:
            url = "https://developer.nrel.gov/api/solar/solar_resource/v1.json"
            params = {
                "api_key": self.nrel_key,
                "lat": lat,
                "lon": lon,
                "system_capacity": 1,  # 1 kW system
                "azimuth": 180,        # South-facing
                "tilt": abs(lat),      # Tilt = latitude for optimal year-round performance
                "dataset": "intl"      # International dataset
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            self.solar_cache[cache_key] = {
                "timestamp": datetime.now(),
                "data": data
            }
            return data
        except Exception as e:
            print(f"Error fetching NREL solar data: {e}")
            return None

    def get_pvwatts_data(self, lat: float, lon: float) -> Dict:
        """Get detailed PV system performance data from PVWatts."""
        try:
            url = "https://developer.nrel.gov/api/pvwatts/v6.json"
            params = {
                "api_key": self.nrel_key,
                "lat": lat,
                "lon": lon,
                **PV_SYSTEM_CONFIG,
                "timeframe": "hourly"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Get current hour's data
            hour = datetime.now().hour
            day_of_year = datetime.now().timetuple().tm_yday
            index = (day_of_year - 1) * 24 + hour
            
            return {
                "ac": data["outputs"]["ac"][index],  # AC power output
                "dc": data["outputs"]["dc"][index],  # DC power output
                "poa": data["outputs"]["poa"][index]  # Plane of array irradiance
            }
        except Exception as e:
            print(f"Error fetching PVWatts data: {e}")
            return None

    def calculate_solar_irradiance(self, weather_data: Dict, nrel_data: Dict, pvwatts_data: Dict) -> Dict:
        """Calculate comprehensive solar irradiance using multiple data sources."""
        if not all([weather_data, nrel_data]):
            return {"ghi": 0.0, "dni": 0.0, "dhi": 0.0}
        
        current_hour = datetime.now().hour
        if not (6 <= current_hour <= 18):  # Night time
            return {"ghi": 0.0, "dni": 0.0, "dhi": 0.0}
        
        # Get cloud coverage from OpenWeather
        clouds = weather_data.get("clouds", {}).get("all", 0)
        cloud_factor = 1 - (clouds / 100) * 0.7
        
        # Get baseline values from NREL
        baseline = nrel_data.get("outputs", {}).get("avg_dni", {}).get("annual", 1000)
        
        # Adjust for time of day
        time_factor = np.sin((current_hour - 6) * np.pi / 12)
        time_factor = max(0, time_factor)  # Ensure non-negative
        
        # Calculate components
        dni = baseline * cloud_factor * time_factor  # Direct Normal Irradiance
        dhi = dni * 0.2  # Diffuse Horizontal Irradiance (simplified)
        
        # Calculate GHI with proper solar angle consideration
        solar_angle = abs(np.cos((current_hour - 12) * np.pi / 12))
        ghi = dni * solar_angle + dhi  # Global Horizontal Irradiance
        
        # If we have PVWatts data, use it to refine our estimates
        if pvwatts_data and isinstance(pvwatts_data, dict):
            poa_irradiance = pvwatts_data.get("poa", 0)
            if poa_irradiance > 0:
                # Adjust our estimates based on PVWatts plane of array irradiance
                adjustment_factor = poa_irradiance / (ghi + 0.1)  # Avoid division by zero
                ghi *= adjustment_factor
                dni *= adjustment_factor
                dhi *= adjustment_factor
        
        return {
            "ghi": max(0, min(1500, ghi)),  # Global Horizontal Irradiance, capped at reasonable maximum
            "dni": max(0, min(1200, dni)),  # Direct Normal Irradiance
            "dhi": max(0, min(500, dhi))    # Diffuse Horizontal Irradiance
        }
        
        current_hour = datetime.now().hour
        if not (6 <= current_hour <= 18):  # Night time
            return {"ghi": 0.0, "dni": 0.0, "dhi": 0.0}
        
        # Get cloud coverage from OpenWeather
        clouds = weather_data.get("clouds", {}).get("all", 0)
        cloud_factor = 1 - (clouds / 100) * 0.7
        
        # Get baseline values from NREL
        baseline = nrel_data.get("outputs", {}).get("avg_dni", {}).get("annual", 1000)
        
        # Adjust for time of day
        time_factor = np.sin((current_hour - 6) * np.pi / 12)
        
        # Calculate components
        dni = baseline * cloud_factor * time_factor  # Direct Normal Irradiance
        dhi = dni * 0.2  # Diffuse Horizontal Irradiance (simplified)
        ghi = dni * np.cos((current_hour - 12) * np.pi / 12) + dhi  # Global Horizontal Irradiance
        
        # If we have Solcast data, use it to refine our estimates
        if solcast_data and "forecasts" in solcast_data:
            current_forecast = solcast_data["forecasts"][0]
            ghi = current_forecast.get("ghi", ghi)
            dni = current_forecast.get("dni", dni)
            dhi = current_forecast.get("dhi", dhi)
        
        return {
            "ghi": max(0, ghi),  # Global Horizontal Irradiance
            "dni": max(0, dni),  # Direct Normal Irradiance
            "dhi": max(0, dhi)   # Diffuse Horizontal Irradiance
        }

class SolarSystem:
    def __init__(self, specs: Dict, alert_system: AlertSystem):
        self.specs = specs
        self.alert_system = alert_system
        
    def calculate_output(self, temp: float, humidity: float, irradiance_data: Dict, location: str) -> Dict:
        # Input validation
        if not isinstance(irradiance_data, dict) or not all(k in irradiance_data for k in ["ghi", "dni", "dhi"]):
            return {
                "voltage": 0.0,
                "current": 0.0,
                "power": 0.0,
                "efficiency_class": 0
            }

        # Temperature effect
        temp_diff = temp - self.specs["nominal_temp"]
        temp_factor = 1 + (self.specs["temp_coefficient"] * temp_diff)
        temp_factor = max(0.6, min(1.2, temp_factor))  # Limit temperature factor range
        
        # Humidity effect (high humidity reduces efficiency)
        humidity_factor = 1 - max(0, min(0.3, (humidity - 75) / 100 * 0.1))  # Cap humidity reduction
        
        # Calculate panel tilt factor based on latitude
        tilt_factor = np.cos(np.radians(PV_SYSTEM_CONFIG["tilt"]))
        
        # Use total irradiance with proper tilt consideration
        total_irradiance = (irradiance_data["ghi"] + 
                          (irradiance_data["dni"] * tilt_factor) + 
                          (irradiance_data["dhi"] * 0.5))  # Improved tilt factor
        
        # Apply time of day factor
        hour = datetime.now().hour
        time_factor = np.sin((hour - 6) * np.pi / 12) if 6 <= hour <= 18 else 0
        total_irradiance *= max(0, time_factor)
        
        # Calculate voltage considering irradiance components
        voltage_base = self.specs["max_voltage"] * (total_irradiance / 1000)
        voltage_direct = voltage_base * 0.85  # Direct component
        voltage_diffuse = (irradiance_data["dhi"] / 1000) * self.specs["max_voltage"] * 0.15
        voltage = (voltage_direct + voltage_diffuse) * temp_factor * humidity_factor
        
        # Calculate current with similar consideration
        current_base = self.specs["max_current"] * (total_irradiance / 1000)
        current_direct = current_base * 0.85
        current_diffuse = (irradiance_data["dhi"] / 1000) * self.specs["max_current"] * 0.15
        current = (current_direct + current_diffuse) * temp_factor * humidity_factor
        
        # Apply environmental factors
        env_factor = self._calculate_environmental_factor(temp, humidity)
        voltage *= env_factor
        current *= env_factor
        
        # Clip to valid ranges
        voltage = np.clip(voltage, 0, self.specs["max_voltage"])
        current = np.clip(current, 0, self.specs["max_current"])
        
        # Calculate power with efficiency consideration
        power = voltage * current
        
        # Apply system losses from PVWatts configuration
        power *= (1 - PV_SYSTEM_CONFIG["losses"] / 100)
        
        # Generate alerts based on conditions
        self._check_conditions(temp, humidity, power, location)
        
        return {
            "voltage": voltage,
            "current": current,
            "power": power,
            "efficiency_class": self._classify_efficiency(power)
        }
        
    def _calculate_environmental_factor(self, temp: float, humidity: float) -> float:
        """Calculate environmental impact factor (0-1) based on temperature and humidity."""
        # Temperature effect (decreases efficiency as temperature rises above nominal)
        temp_effect = 1.0
        if temp > self.specs["nominal_temp"]:
            temp_diff = temp - self.specs["nominal_temp"]
            temp_effect = max(0.7, 1.0 - (temp_diff * 0.005))  # 0.5% reduction per degree above nominal
        
        # Humidity effect (high humidity reduces efficiency)
        humidity_effect = 1.0
        if humidity > 70:
            humidity_effect = max(0.8, 1.0 - ((humidity - 70) * 0.005))  # 0.5% reduction per % above 70%
        
        # Combined effect
        return temp_effect * humidity_effect
    
    def _classify_efficiency(self, power: float) -> int:
        efficiency = power / self.specs["max_power"]
        if efficiency < 0.3:
            return 0  # Low
        elif efficiency < 0.7:
            return 1  # Medium
        return 2  # High
    
    def _check_conditions(self, temp: float, humidity: float, power: float, location: str):
        # Temperature alerts
        if temp >= ALERT_THRESHOLDS["temperature"]["critical"]:
            self.alert_system.add_alert(
                location, "temperature", "critical",
                f"Critical temperature ({temp:.1f}°C) affecting solar panel efficiency"
            )
        elif temp >= ALERT_THRESHOLDS["temperature"]["warning"]:
            self.alert_system.add_alert(
                location, "temperature", "warning",
                f"High temperature ({temp:.1f}°C) may reduce efficiency"
            )
        
        # Humidity alerts
        if humidity >= ALERT_THRESHOLDS["humidity"]["high"]:
            self.alert_system.add_alert(
                location, "humidity", "warning",
                f"High humidity ({humidity:.1f}%) may affect panel performance"
            )
        
        # Power output alerts
        if power < ALERT_THRESHOLDS["power"]["low"]:
            self.alert_system.add_alert(
                location, "power", "warning",
                f"Low power output ({power:.1f}W) - check for issues"
            )

class LoadManager:
    def __init__(self, alert_system: AlertSystem):
        self.alert_system = alert_system
        
    def get_recommendation(self, battery_voltage: float, power_output: float, location: str) -> Dict:
        if battery_voltage <= ALERT_THRESHOLDS["battery"]["critical"]:
            self.alert_system.add_alert(
                location, "battery", "critical",
                "Critical battery level - shut down non-essential loads"
            )
            return {
                "status": "CRITICAL",
                "action": "Shut down non-essential loads",
                "max_load": 50  # W
            }
        elif battery_voltage <= ALERT_THRESHOLDS["battery"]["warning"]:
            self.alert_system.add_alert(
                location, "battery", "warning",
                "Low battery level - reduce power consumption"
            )
            return {
                "status": "WARNING",
                "action": "Reduce power consumption",
                "max_load": 100  # W
            }
        elif power_output < ALERT_THRESHOLDS["power"]["low"]:
            self.alert_system.add_alert(
                location, "power", "warning",
                "Low power generation - consider load reduction"
            )
            return {
                "status": "CAUTION",
                "action": "Monitor power consumption",
                "max_load": 150  # W
            }
        return {
            "status": "NORMAL",
            "action": "Normal operation",
            "max_load": None
        }

    class WeatherData:
    def __init__(self, temp: float, humidity: float):
        self.temp = temp
        self.humidity = humidity

class LocationData:
    def __init__(self, lat: float, lon: float, altitude: float):
        self.lat = lat
        self.lon = lon
        self.altitude = altitude

class MonitoringData:
    def __init__(
        self,
        timestamp: str,
        temperature: float,
        humidity: float,
        solar_ghi: float,
        solar_dni: float,
        solar_dhi: float,
        solar_voltage: float,
        solar_current: float,
        power_output: float,
        efficiency_class: int,
        battery_voltage: float,
        load_status: str,
        load_action: str,
        max_load: Union[float, None]
    ):
        self.timestamp = timestamp
        self.temperature = temperature
        self.humidity = humidity
        self.solar_ghi = solar_ghi
        self.solar_dni = solar_dni
        self.solar_dhi = solar_dhi
        self.solar_voltage = solar_voltage
        self.solar_current = solar_current
        self.power_output = power_output
        self.efficiency_class = efficiency_class
        self.battery_voltage = battery_voltage
        self.load_status = load_status
        self.load_action = load_action
        self.max_load = max_load

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Timestamp": self.timestamp,
            "Temperature": round(self.temperature, 2),
            "Humidity": round(self.humidity, 2),
            "Solar_GHI": round(self.solar_ghi, 2),
            "Solar_DNI": round(self.solar_dni, 2),
            "Solar_DHI": round(self.solar_dhi, 2),
            "Solar_Voltage": round(self.solar_voltage, 2),
            "Solar_Current": round(self.solar_current, 2),
            "Power_Output": round(self.power_output, 2),
            "Efficiency_Class": self.efficiency_class,
            "Battery_Voltage": self.battery_voltage,
            "Load_Status": self.load_status,
            "Load_Action": self.load_action,
            "Max_Load": self.max_load
        }

class MonitoringSystem:
    def __init__(self):
        self.weather_monitor = WeatherMonitor(OPENWEATHER_API_KEY, NREL_API_KEY)
        self.solar_system = SolarSystem(SOLAR_SPECS, AlertSystem())
        self.load_manager = LoadManager(AlertSystem())

    def monitor_location(self, location_name: str, coords: LocationData) -> Union[MonitoringData, None]:
        """Monitor solar system at a specific location.
        
        Args:
            location_name (str): Name of the location to monitor
            coords (LocationData): Location coordinates and altitude
            
        Returns:
            MonitoringData: Monitoring data or None if error occurs
        """
        try:
            # Get data from all sources
            weather_data = self.weather_monitor.get_weather(coords.lat, coords.lon)
            nrel_data = self.weather_monitor.get_nrel_solar_data(coords.lat, coords.lon, coords.altitude)
            pvwatts_data = self.weather_monitor.get_pvwatts_data(coords.lat, coords.lon)
            
            if not weather_data or not nrel_data:
                print(f"Failed to get data for {location_name}")
                return None
            
            # Extract weather parameters
            temp = weather_data.get("main", {}).get("temp", 25.0)
            humidity = weather_data.get("main", {}).get("humidity", 50.0)
            
            # Calculate solar data
            irradiance_data = self.weather_monitor.calculate_solar_irradiance(
                weather_data, nrel_data, pvwatts_data
            )
            
            # Calculate solar output
            solar_output = self.solar_system.calculate_output(
                temp, humidity, irradiance_data, location_name
            )
            
            if not solar_output:
                print(f"Failed to calculate solar output for {location_name}")
                return None
            
            # Get load recommendations
            battery_voltage = 12.4  # This should come from real measurements
            load_rec = self.load_manager.get_recommendation(
                battery_voltage, solar_output["power"], location_name
            )
            
            # Create monitoring data
            monitoring_data = MonitoringData(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                temperature=temp,
                humidity=humidity,
                solar_ghi=irradiance_data["ghi"],
                solar_dni=irradiance_data["dni"],
                solar_dhi=irradiance_data["dhi"],
                solar_voltage=solar_output["voltage"],
                solar_current=solar_output["current"],
                power_output=solar_output["power"],
                efficiency_class=solar_output["efficiency_class"],
                battery_voltage=battery_voltage,
                load_status=load_rec["status"],
                load_action=load_rec["action"],
                max_load=load_rec["max_load"]
            )
            
            # Store in database
            try:
                db_name = f"solar_monitoring_{location_name.lower()}.db"
                conn = sqlite3.connect(db_name)
                df = pd.DataFrame([monitoring_data.to_dict()])
                df.to_sql("monitoring", conn, if_exists="append", index=False)
                conn.close()
            except Exception as e:
                print(f"Database error for {location_name}: {e}")
            
            return monitoring_data
            
        except Exception as e:
            print(f"Error monitoring {location_name}: {e}")
            return None
        
        if not weather_data or not nrel_data:
            print(f"Failed to get data for {location_name}")
            return None
        
        # Extract weather parameters with error handling
        temp = weather_data.get("main", {}).get("temp", 25.0)  # Default to 25°C if missing
        humidity = weather_data.get("main", {}).get("humidity", 50.0)  # Default to 50% if missing
        
        # Calculate irradiance and solar output
        irradiance_data = weather_monitor.calculate_solar_irradiance(weather_data, nrel_data, pvwatts_data)
        solar_output = solar_system.calculate_output(temp, humidity, irradiance_data, location_name)
        
        if not solar_output:
            print(f"Failed to calculate solar output for {location_name}")
            return None
        
        # Get load recommendations (simulate battery voltage for now)
        battery_voltage = 12.4  # This should come from real measurements
        load_rec = load_manager.get_recommendation(
            battery_voltage, solar_output["power"], location_name
        )
        
        # Prepare monitoring data
        data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Temperature": round(temp, 2),
            "Humidity": round(humidity, 2),
            "Solar_GHI": round(irradiance_data["ghi"], 2),
            "Solar_DNI": round(irradiance_data["dni"], 2),
            "Solar_DHI": round(irradiance_data["dhi"], 2),
            "Solar_Voltage": round(solar_output["voltage"], 2),
            "Solar_Current": round(solar_output["current"], 2),
            "Power_Output": round(solar_output["power"], 2),
            "Efficiency_Class": solar_output["efficiency_class"],
            "Battery_Voltage": battery_voltage,
            "Load_Status": load_rec["status"],
            "Load_Action": load_rec["action"],
            "Max_Load": load_rec["max_load"]
        }
        
        # Store in database
        try:
            db_name = f"solar_monitoring_{location_name.lower()}.db"
            conn = sqlite3.connect(db_name)
            df = pd.DataFrame([data])
            df.to_sql("monitoring", conn, if_exists="append", index=False)
            conn.close()
        except Exception as e:
            print(f"Database error for {location_name}: {e}")
            # Continue execution even if database storage fails
        
        return data
        
    except Exception as e:
        print(f"Error monitoring {location_name}: {e}")
        return None    # Calculate comprehensive irradiance using all data sources
    irradiance_data = weather_monitor.calculate_solar_irradiance(weather_data, nrel_data, pvwatts_data)
    
    # Calculate solar output with detailed irradiance data
    solar_output = solar_system.calculate_output(temp, humidity, irradiance_data, location_name)
    
    # Get load recommendations (simulate battery voltage for now)
    battery_voltage = 12.4  # This should come from real measurements
    load_rec = load_manager.get_recommendation(
        battery_voltage, solar_output["power"], location_name
    )
    
    # Prepare data for storage
    data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Temperature": round(temp, 2),
        "Humidity": round(humidity, 2),
        "Solar_GHI": round(irradiance_data["ghi"], 2),
        "Solar_DNI": round(irradiance_data["dni"], 2),
        "Solar_DHI": round(irradiance_data["dhi"], 2),
        "Solar_Voltage": round(solar_output["voltage"], 2),
        "Solar_Current": round(solar_output["current"], 2),
        "Power_Output": round(solar_output["power"], 2),
        "Efficiency_Class": solar_output["efficiency_class"],
        "Battery_Voltage": battery_voltage,
        "Load_Status": load_rec["status"],
        "Load_Action": load_rec["action"],
        "Max_Load": load_rec["max_load"]
    }
    
    # Store in database
    db_name = f"solar_monitoring_{location_name.lower()}.db"
    conn = sqlite3.connect(db_name)
    df = pd.DataFrame([data])
    df.to_sql("monitoring", conn, if_exists="append", index=False)
    conn.close()
    
    return data

def main():
    """Main function to run the solar monitoring system."""
    if not all([OPENWEATHER_API_KEY, NREL_API_KEY]):
        print("Error: One or more API keys not set! Please set:")
        print("- OPENWEATHER_API_KEY")
        print("- NREL_API_KEY")
        return
    
    # Initialize monitoring system
    monitoring_system = MonitoringSystem()
    
    while True:
        print("\n" + "="*50)
        print(f"Solar Monitoring Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        
        # Process each location
        for location_name, coords_dict in LOCATIONS.items():
            print(f"\nMonitoring {location_name}...")
            
            # Convert dictionary to LocationData object
            coords = LocationData(
                lat=coords_dict["lat"],
                lon=coords_dict["lon"],
                altitude=coords_dict["altitude"]
            )
            
            # Monitor location
            data = monitoring_system.monitor_location(location_name, coords)
            
            if data:
                # Print monitoring data
                print(f"Temperature: {data.temperature:.1f}°C")
                print(f"Humidity: {data.humidity:.1f}%")
                print(f"Solar Power: {data.power_output:.1f}W")
                print(f"Efficiency: {['Low', 'Medium', 'High'][data.efficiency_class]}")
                print(f"Load Status: {data.load_status}")
                
                # Print current alerts
                alerts = monitoring_system.solar_system.alert_system.alerts
                if alerts:
                    print("\nCurrent Alerts:")
                    for alert in alerts:
                        print(f"⚠️ {alert.severity.upper()}: {alert.message}")
                        
                # Clear alerts for next iteration
                monitoring_system.solar_system.alert_system.clear_alerts()
            
        # Wait for next update
        print("\nWaiting 5 minutes for next update...")
        time.sleep(300)
            
            if data:
                print(f"Temperature: {data['Temperature']}°C")
                print(f"Humidity: {data['Humidity']}%")
                print(f"Solar Power: {data['Power_Output']}W")
                print(f"Efficiency: {['Low', 'Medium', 'High'][data['Efficiency_Class']]}")
                print(f"Load Status: {data['Load_Status']}")
                
                # Save any generated alerts
                alert_system.save_alerts(location_name)
                
                # Print current alerts
                if alert_system.alerts:
                    print("\nCurrent Alerts:")
                    for alert in alert_system.alerts:
                        print(f"⚠️ {alert['severity'].upper()}: {alert['message']}")
            
        # Wait for 5 minutes before next update
        print("\nWaiting 5 minutes for next update...")
        time.sleep(300)

def run_monitoring():
    """Run the solar monitoring system with error handling."""
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nError in monitoring system: {e}")
        raise  # Re-raise the exception for debugging

if __name__ == "__main__":
    run_monitoring()