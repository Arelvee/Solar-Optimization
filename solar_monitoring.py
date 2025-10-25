import os
import sqlite3
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Union, Any
import time
from dataclasses import dataclass

# ------------------------
# CONFIGURATION
# ------------------------
def get_api_keys():
    """Get API keys from config file"""
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
            openweather_key = config.get("OPENWEATHER_API_KEY")
            nrel_key = config.get("NREL_API_KEY")
            
        if not openweather_key or not nrel_key:
            raise ValueError("API keys not found in config file")
            
        print("✅ API keys loaded successfully from config.json")
        return openweather_key, nrel_key
        
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        print("Please ensure config.json exists with valid API keys")
        return None, None

OPENWEATHER_API_KEY, NREL_API_KEY = get_api_keys()

if not OPENWEATHER_API_KEY or not NREL_API_KEY:
    print("❌ Cannot start without valid API keys")
    exit(1)

# Real 100W Solar Panel Specifications (Monocrystalline)
REAL_SOLAR_SPECS = {
    "max_power": 100.0,  # Watts
    "voltage_max_power": 18.5,  # Vmp
    "current_max_power": 5.41,  # Imp
    "open_circuit_voltage": 22.3,  # Voc
    "short_circuit_current": 5.86,  # Isc
    "temp_coefficient_voc": -0.0032,  # %/°C
    "temp_coefficient_isc": 0.0005,   # %/°C
    "nominal_temp": 25.0,  # STC temperature
    "efficiency": 0.195,   # 19.5% efficiency
    "panel_area": 0.54     # m²
}

# Real Battery Specifications (12V 20Ah LiPo)
REAL_BATTERY_SPECS = {
    "nominal_voltage": 12.0,  # V
    "capacity": 20.0,         # Ah
    "max_charge_voltage": 12.6,  # V - LiPo is typically 4.2V per cell (3S = 12.6V)
    "float_voltage": 12.4,    # V - LiPo doesn't really have float charging
    "cutoff_voltage": 10.5,   # V - Minimum discharge voltage (3.5V per cell)
    "charge_efficiency": 0.95,  # 95% - LiPo is more efficient
    "depth_of_discharge": 0.8,  # 80% - LiPo can handle deeper discharge
    "cell_count": 3,          # 3S configuration
    "max_charge_current": 10.0,  # A - 0.5C for 20Ah battery
    "max_discharge_current": 20.0,  # A - 1C for 20Ah battery
}

# System Configuration
PV_SYSTEM_CONFIG = {
    "system_capacity": 0.1,  # 100W system size (0.1 kW)
    "module_type": 1,        # Premium monocrystalline
    "array_type": 1,         # Fixed roof mount
    "tilt": 14,             # Tilt angle for Philippines
    "azimuth": 180,         # South-facing
    "losses": 14.08         # System losses
}

LOCATIONS = {
    "Sto_Tomas": {
        "lat": 14.1119, 
        "lon": 121.1483,
        "altitude": 300,
        "battery_soc": 80.0,  # Initial SOC for each location
        "climate_factor": 1.0  # Climate adjustment factor
    },
    "LSPU_San_Pablo": {
        "lat": 14.0689, 
        "lon": 121.3256,
        "altitude": 450,
        "battery_soc": 75.0,
        "climate_factor": 0.95  # Slightly less sunny
    },
    "San_Pablo_City": {
        "lat": 14.0777, 
        "lon": 121.3257,
        "altitude": 430,
        "battery_soc": 85.0,
        "climate_factor": 1.05  # Slightly more sunny
    }
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
        "low": REAL_SOLAR_SPECS["max_power"] * 0.3,
        "medium": REAL_SOLAR_SPECS["max_power"] * 0.7
    },
    "battery": {
        "critical": REAL_BATTERY_SPECS["cutoff_voltage"],
        "warning": 11.0,  # Lower for LiPo
        "full": REAL_BATTERY_SPECS["float_voltage"]
    },
    "solar_voltage": {
        "low": REAL_SOLAR_SPECS["voltage_max_power"] * 0.6,
        "high": REAL_SOLAR_SPECS["open_circuit_voltage"] * 1.1
    }
}

@dataclass
class Alert:
    timestamp: str
    location: str
    type: str
    severity: str
    message: str

class HistoricalDataGenerator:
    """Generates realistic historical solar data from October 1, 2025 with 5-minute intervals"""
    
    def __init__(self):
        self.start_date = datetime(2025, 10, 1)
        self.end_date = datetime.now()
        self.last_rain_time = None
        self.current_rain_duration = 0
        
    def generate_historical_weather_pattern(self, date: datetime, location_factor: float) -> Dict:
        """Generate realistic weather patterns for Philippines"""
        day_of_year = date.timetuple().tm_yday
        hour = date.hour
        minute = date.minute
        
        # Philippines climate: Wet season May-Oct, Dry season Nov-Apr
        is_wet_season = 121 <= day_of_year <= 304  # May 1 to Oct 31
        
        # Base temperature for Philippines
        base_temp = 28.0 + 4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Diurnal temperature variation
        time_of_day = hour + minute/60.0
        diurnal_variation = 6 * np.sin(np.pi * (time_of_day - 6) / 12)
        
        temperature = base_temp + diurnal_variation + np.random.normal(0, 0.3)
        
        # Determine weather condition
        weather_condition = self._determine_daily_weather(date, is_wet_season, location_factor)
        
        # Set values based on weather condition
        if weather_condition == "sunny":
            cloud_coverage = np.random.uniform(10, 30)
            humidity = np.random.uniform(50, 65)
            rainfall = 0.0
            is_raining = False
        elif weather_condition == "partly_cloudy":
            cloud_coverage = np.random.uniform(30, 60)
            humidity = np.random.uniform(60, 75)
            rainfall = 0.0
            is_raining = False
        elif weather_condition == "cloudy":
            cloud_coverage = np.random.uniform(60, 85)
            humidity = np.random.uniform(70, 85)
            rainfall = 0.0
            is_raining = False
        elif weather_condition == "light_rain":
            cloud_coverage = np.random.uniform(70, 95)
            humidity = np.random.uniform(80, 90)
            rainfall = np.random.uniform(0.1, 2.0)
            is_raining = True
        elif weather_condition == "heavy_rain":
            cloud_coverage = np.random.uniform(85, 100)
            humidity = np.random.uniform(85, 95)
            rainfall = np.random.uniform(2.0, 8.0)
            is_raining = True
        else:  # thunderstorm
            cloud_coverage = np.random.uniform(90, 100)
            humidity = np.random.uniform(90, 98)
            rainfall = np.random.uniform(5.0, 15.0)
            is_raining = True
        
        # Adjust for night time
        if hour >= 18 or hour <= 6:
            temperature -= 2
            if is_raining:
                humidity = min(98, humidity + 5)
        
        return {
            "temperature": round(temperature, 1),
            "humidity": round(humidity, 1),
            "cloud_coverage": round(cloud_coverage, 1),
            "is_raining": is_raining,
            "rainfall_mm": round(rainfall, 1),
            "weather_condition": weather_condition,
            "is_wet_season": is_wet_season
        }
    
    def _determine_daily_weather(self, date: datetime, is_wet_season: bool, location_factor: float) -> str:
        """Determine daily weather pattern with realistic transitions"""
        date_str = date.strftime("%Y-%m-%d")
        
        # Use date as seed for consistent daily weather
        daily_seed = hash(date_str) % 10000
        np.random.seed(daily_seed)
        
        if is_wet_season:
            # Wet season probabilities
            probabilities = {
                "sunny": 0.2,
                "partly_cloudy": 0.3,
                "cloudy": 0.2,
                "light_rain": 0.15,
                "heavy_rain": 0.1,
                "thunderstorm": 0.05
            }
        else:
            # Dry season probabilities
            probabilities = {
                "sunny": 0.5,
                "partly_cloudy": 0.3,
                "cloudy": 0.15,
                "light_rain": 0.04,
                "heavy_rain": 0.01,
                "thunderstorm": 0.0
            }
        
        # Adjust for afternoon showers in wet season
        hour = date.hour
        if is_wet_season and 14 <= hour <= 17:
            probabilities = {
                "sunny": 0.1,
                "partly_cloudy": 0.2,
                "cloudy": 0.2,
                "light_rain": 0.3,
                "heavy_rain": 0.15,
                "thunderstorm": 0.05
            }
        
        # Select weather condition
        rand_val = np.random.random()
        cumulative = 0
        for condition, prob in probabilities.items():
            cumulative += prob
            if rand_val <= cumulative:
                return condition
        
        return "partly_cloudy"
    
    def calculate_historical_solar_irradiance(self, date: datetime, weather_data: Dict, location_factor: float) -> Dict:
        """Calculate solar irradiance with realistic values - NEVER ZERO DURING DAYTIME"""
        hour = date.hour
        minute = date.minute
        time_of_day = hour + minute/60.0
        
        # Night time - return minimal but not zero values for monitoring
        if not (6 <= time_of_day <= 18):
            return {
                "irradiance": 0.0, 
                "dni": 0.0, 
                "dhi": 0.0, 
                "ghi": 0.0
            }
        
        # Time of day factor - peak at solar noon
        time_factor = np.sin((time_of_day - 6) * np.pi / 12)
        time_factor = max(0, time_factor)  # Ensure non-negative
        
        # Weather effects on irradiance
        weather_factors = {
            "sunny": 1.0,
            "partly_cloudy": 0.7,
            "cloudy": 0.4,
            "light_rain": 0.15,
            "heavy_rain": 0.05,
            "thunderstorm": 0.02
        }
        
        weather_factor = weather_factors.get(weather_data["weather_condition"], 0.5)
        
        # Cloud effect
        cloud_factor = 1 - (weather_data["cloud_coverage"] / 100) * 0.6
        
        # Base irradiance (W/m²)
        base_irradiance = 1000 * time_factor * weather_factor * cloud_factor * location_factor
        
        # Add realistic variation
        noise = np.random.normal(1.0, 0.1)
        total_irradiance = max(50, base_irradiance * noise)  # Minimum 50 W/m² during daytime
        
        # Calculate components
        if weather_data["is_raining"]:
            # More diffuse light during rain
            dni = total_irradiance * 0.2
            dhi = total_irradiance * 0.8
        else:
            dni = total_irradiance * 0.6
            dhi = total_irradiance * 0.4
        
        ghi = total_irradiance
        
        # Convert to Lux (approximate conversion)
        irradiance_lux = ghi * 120
        
        return {
            "irradiance": max(0, min(120000, irradiance_lux)),
            "dni": max(0, min(1000, dni)),
            "dhi": max(0, min(800, dhi)),
            "ghi": max(0, min(1000, ghi))
        }
    
    def calculate_historical_power_output(self, irradiance_data: Dict, temperature: float, humidity: float, timestamp: datetime) -> Dict:
        """Calculate power output with REALISTIC NON-ZERO VALUES"""
        hour = timestamp.hour
        
        # Night time - minimal but non-zero values for system monitoring
        if irradiance_data["irradiance"] == 0:
            return {
                "panel_voltage": 0.8,    # Small voltage for monitoring
                "panel_current": 0.02,   # Minimal current
                "raw_power": 0.0,
                "actual_power": 0.0,
                "efficiency": 0.0
            }
        
        # Convert Lux to W/m²
        irradiance_w_m2 = irradiance_data["irradiance"] / 120.0
        
        # Calculate panel performance - REALISTIC VALUES
        panel_voltage = REAL_SOLAR_SPECS["voltage_max_power"] * (irradiance_w_m2 / 1000.0)**0.08
        panel_current = REAL_SOLAR_SPECS["current_max_power"] * (irradiance_w_m2 / 1000.0)
        
        # Temperature compensation
        temp_diff = temperature - REAL_SOLAR_SPECS["nominal_temp"]
        voltage_temp_factor = 1 + (REAL_SOLAR_SPECS["temp_coefficient_voc"] * temp_diff)
        panel_voltage *= voltage_temp_factor
        
        # Ensure realistic limits
        panel_voltage = max(12.0, min(REAL_SOLAR_SPECS["open_circuit_voltage"], panel_voltage))
        panel_current = max(0.1, min(REAL_SOLAR_SPECS["short_circuit_current"], panel_current))
        
        # Calculate power
        raw_power = panel_voltage * panel_current
        
        # System efficiency (85-95%)
        system_efficiency = 0.92 + np.random.normal(0, 0.02)
        system_efficiency = max(0.85, min(0.95, system_efficiency))
        
        actual_power = raw_power * system_efficiency
        
        # Ensure power doesn't exceed panel rating
        actual_power = min(actual_power, REAL_SOLAR_SPECS["max_power"])
        
        # Add realistic variation
        power_variation = 1.0 + np.random.normal(0, 0.05)
        actual_power = max(0.5, actual_power * power_variation)  # Minimum 0.5W during daytime
        
        # Efficiency calculation
        max_possible_power = REAL_SOLAR_SPECS["max_power"] * (irradiance_w_m2 / 1000.0)
        efficiency_ratio = actual_power / max_possible_power if max_possible_power > 0 else 0
        
        return {
            "panel_voltage": round(panel_voltage, 1),
            "panel_current": round(panel_current, 2),
            "raw_power": round(raw_power, 1),
            "actual_power": round(actual_power, 1),
            "efficiency": round(efficiency_ratio, 3)
        }
    
    def simulate_battery_behavior(self, solar_power: float, previous_soc: float, time_delta_hours: float, timestamp: datetime) -> Dict:
        """Simulate battery behavior with REALISTIC CHARGING/DISCHARGING"""
        hour = timestamp.hour
        
        # Variable load based on time of day (more realistic)
        if 6 <= hour <= 8:   # Morning peak
            load_power = np.random.uniform(25, 35)
        elif 18 <= hour <= 22:  # Evening peak
            load_power = np.random.uniform(30, 40)
        elif 0 <= hour <= 5:    # Night - minimal
            load_power = np.random.uniform(10, 20)
        else:  # Daytime
            load_power = np.random.uniform(20, 30)
        
        net_power = solar_power - load_power
        
        # Calculate battery changes
        battery_capacity_wh = REAL_BATTERY_SPECS["nominal_voltage"] * REAL_BATTERY_SPECS["capacity"]
        
        if net_power > 0:  # Charging
            charge_efficiency = REAL_BATTERY_SPECS["charge_efficiency"]
            energy_added = net_power * time_delta_hours * charge_efficiency
            soc_change = (energy_added / battery_capacity_wh) * 100
        else:  # Discharging
            energy_used = abs(net_power) * time_delta_hours
            soc_change = -(energy_used / battery_capacity_wh) * 100
        
        new_soc = previous_soc + soc_change
        # Keep SOC between 20-100% for battery health
        new_soc = max(20.0, min(100.0, new_soc))
        
        # Calculate realistic LiPo battery voltage based on SOC and charging state
        if net_power > 0:  # Charging
            if new_soc >= 95:
                battery_voltage = REAL_BATTERY_SPECS["max_charge_voltage"]
            elif new_soc >= 80:
                battery_voltage = 12.5
            else:
                battery_voltage = 11.8 + (new_soc / 100) * 0.7
        else:  # Discharging
            # LiPo discharge curve is relatively flat
            if new_soc >= 80:
                battery_voltage = 12.4
            elif new_soc >= 50:
                battery_voltage = 12.2
            elif new_soc >= 30:
                battery_voltage = 12.0
            else:
                battery_voltage = 11.6
        
        # Add small random variation
        voltage_variation = np.random.normal(1.0, 0.01)
        battery_voltage = round(battery_voltage * voltage_variation, 1)
        
        charge_current = net_power / REAL_BATTERY_SPECS["nominal_voltage"]
        
        return {
            "battery_voltage": battery_voltage,
            "state_of_charge": round(new_soc, 1),
            "charge_current": round(charge_current, 2),
            "net_power": round(net_power, 1),
            "load_power": round(load_power, 1),
            "new_soc": new_soc
        }
    
    def determine_system_status(self, battery_voltage: float, state_of_charge: float, power_output: float, timestamp: datetime) -> Dict:
        """Determine system status and recommendations"""
        hour = timestamp.hour
        
        if battery_voltage <= 11.0:
            status = "CRITICAL"
            action = "Shut down non-essential loads"
            max_load = 10
            recommendation = "Disconnect all non-critical loads immediately"
        elif battery_voltage <= 11.8:
            status = "WARNING"
            action = "Reduce power consumption"
            max_load = 15
            recommendation = "Turn off unnecessary devices"
        elif state_of_charge < 30:
            status = "CAUTION"
            action = "Monitor battery level"
            max_load = 20
            recommendation = "Conserve energy, low battery"
        elif power_output > 70 and state_of_charge > 85:
            status = "OPTIMAL"
            action = "Normal operation - excellent generation"
            max_load = 40
            recommendation = "System operating at peak performance"
        elif power_output > 50:
            status = "GOOD"
            action = "Normal operation"
            max_load = 35
            recommendation = "Good solar generation"
        elif power_output > 20:
            status = "NORMAL"
            action = "Normal operation"
            max_load = 30
            recommendation = "Adequate solar generation"
        else:
            status = "LOW_GENERATION"
            action = "Monitor power consumption"
            max_load = 25
            recommendation = "Low solar generation, consider reducing load"
        
        # Night time adjustments
        if not (6 <= hour <= 18) and power_output == 0:
            if state_of_charge < 40:
                status = "CAUTION"
                action = "Conserve energy until morning"
                max_load = 15
                recommendation = "Minimize power usage until solar generation resumes"
        
        return {
            "status": status,
            "action": action,
            "max_load": max_load,
            "recommendation": recommendation
        }

    def generate_historical_data_for_location(self, location_name: str, initial_soc: float, climate_factor: float) -> List[Dict]:
        """Generate complete historical dataset for a location"""
        print(f"📊 Generating 5-minute interval historical data for {location_name} from October 1, 2025...")
        
        historical_records = []
        current_datetime = self.start_date
        current_soc = initial_soc
        
        while current_datetime <= self.end_date:
            # Generate weather data
            weather_data = self.generate_historical_weather_pattern(current_datetime, climate_factor)
            
            # Calculate solar irradiance
            irradiance_data = self.calculate_historical_solar_irradiance(
                current_datetime, weather_data, climate_factor
            )
            
            # Calculate solar power output
            solar_output = self.calculate_historical_power_output(
                irradiance_data, weather_data["temperature"], weather_data["humidity"], current_datetime
            )
            
            # Calculate battery behavior
            battery_data = self.simulate_battery_behavior(
                solar_output["actual_power"], current_soc, 1/12, current_datetime
            )
            current_soc = battery_data["new_soc"]
            
            # Determine system status
            system_status = self.determine_system_status(
                battery_data["battery_voltage"],
                battery_data["state_of_charge"],
                solar_output["actual_power"],
                current_datetime
            )
            
            # Efficiency classification
            if solar_output["actual_power"] >= 70:
                efficiency_class = 2  # High
            elif solar_output["actual_power"] >= 30:
                efficiency_class = 1  # Medium
            else:
                efficiency_class = 0  # Low
            
            # Create historical record
            record = {
                "Timestamp": current_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                "Temperature": weather_data["temperature"],
                "Humidity": weather_data["humidity"],
                "Rainfall": weather_data["rainfall_mm"],
                "Solar_Irradiance": round(irradiance_data["irradiance"], 0),
                "Solar_DNI": round(irradiance_data["dni"], 1),
                "Solar_DHI": round(irradiance_data["dhi"], 1),
                "Panel_Voltage": solar_output["panel_voltage"],
                "Panel_Current": solar_output["panel_current"],
                "Raw_Power": solar_output["raw_power"],
                "Actual_Power": solar_output["actual_power"],
                "Panel_Efficiency": solar_output["efficiency"],
                "Battery_Voltage": battery_data["battery_voltage"],
                "State_of_Charge": battery_data["state_of_charge"],
                "Charge_Current": battery_data["charge_current"],
                "Net_Power": battery_data["net_power"],
                "Load_Power": battery_data["load_power"],
                "Efficiency_Class": efficiency_class,
                "Load_Status": system_status["status"],
                "Load_Action": system_status["action"],
                "Max_Load": system_status["max_load"],
                "Recommended_Action": system_status["recommendation"]
            }
            
            historical_records.append(record)
            
            # Move to next 5-minute interval
            current_datetime += timedelta(minutes=5)
            
            # Progress indicator every 12 hours of data
            if len(historical_records) % 144 == 0:
                days_generated = len(historical_records) / 288
                print(f"  📅 Generated {days_generated:.1f} days of data...")
        
        print(f"✅ Generated {len(historical_records)} historical records for {location_name}")
        print(f"   📊 Sample - Irradiance range: {min(r['Solar_Irradiance'] for r in historical_records if r['Solar_Irradiance'] > 0):,.0f} to {max(r['Solar_Irradiance'] for r in historical_records):,.0f} lux")
        print(f"   ⚡ Sample - Power range: {min(r['Actual_Power'] for r in historical_records if r['Actual_Power'] > 0):.1f} to {max(r['Actual_Power'] for r in historical_records):.1f} W")
        print(f"   🔋 Sample - Battery SOC range: {min(r['State_of_Charge'] for r in historical_records):.1f} to {max(r['State_of_Charge'] for r in historical_records):.1f}%")
        
        return historical_records


class RealDataSimulator:
    """Simulates real solar panel and battery data based on API environmental data"""
    
    def __init__(self, location_name: str):
        self.location_name = location_name
        self.battery_state_of_charge = LOCATIONS[location_name]["battery_soc"]
        self.battery_voltage = 12.8
        self.last_update = datetime.now()
        self.load_power = 25.0  # Average load power in Watts

    def calculate_solar_panel_output(self, irradiance_lux: float, temperature: float, humidity: float) -> Dict:
        """Calculate solar panel output based on environmental conditions from API."""
        try:
            # Convert Lux to W/m² (standard conversion)
            irradiance_w_m2 = irradiance_lux / 120.0
            
            # Night time - minimal monitoring values
            if irradiance_lux < 50:
                return {
                    "panel_voltage": 0.8,
                    "panel_current": 0.02,
                    "raw_power": 0.0,
                    "actual_power": 0.0,
                    "efficiency": 0.0,
                    "is_night": True
                }
            
            # Calculate maximum possible power based on irradiance
            max_possible_power = REAL_SOLAR_SPECS["max_power"] * (irradiance_w_m2 / 1000.0)
            
            # Temperature effect on voltage (solar panels perform worse when hot)
            temp_diff = temperature - REAL_SOLAR_SPECS["nominal_temp"]
            voltage_temp_factor = 1 + (REAL_SOLAR_SPECS["temp_coefficient_voc"] * temp_diff)
            
            # Calculate panel voltage based on irradiance and temperature
            base_voltage = REAL_SOLAR_SPECS["voltage_max_power"] * (irradiance_w_m2 / 1000.0)**0.08
            panel_voltage = base_voltage * voltage_temp_factor
            
            # Calculate panel current based on irradiance
            panel_current = REAL_SOLAR_SPECS["current_max_power"] * (irradiance_w_m2 / 1000.0)
            
            # Ensure realistic limits
            panel_voltage = max(12.0, min(REAL_SOLAR_SPECS["open_circuit_voltage"], panel_voltage))
            panel_current = max(0.1, min(REAL_SOLAR_SPECS["short_circuit_current"], panel_current))
            
            # Calculate power
            raw_power = panel_voltage * panel_current
            
            # System efficiency factors
            temperature_efficiency = 1.0 - max(0, (temperature - 25) * 0.004)  # -0.4% per °C above 25°C
            humidity_efficiency = 1.0 - max(0, (humidity - 50) * 0.0002)      # Small humidity effect
            system_efficiency = 0.92 * temperature_efficiency * humidity_efficiency
            
            actual_power = raw_power * system_efficiency
            
            # Ensure power doesn't exceed panel rating
            actual_power = min(actual_power, REAL_SOLAR_SPECS["max_power"])
            
            # Add realistic small variation
            power_variation = 1.0 + np.random.normal(0, 0.03)
            actual_power = max(0.5, actual_power * power_variation)
            
            # Calculate efficiency ratio
            efficiency_ratio = actual_power / max_possible_power if max_possible_power > 0 else 0
            
            return {
                "panel_voltage": round(panel_voltage, 2),
                "panel_current": round(panel_current, 2),
                "raw_power": round(raw_power, 2),
                "actual_power": round(actual_power, 2),
                "efficiency": round(efficiency_ratio, 3),
                "is_night": False
            }
            
        except Exception as e:
            print(f"❌ Error calculating solar panel output: {e}")
            return {
                "panel_voltage": 0.8,
                "panel_current": 0.02,
                "raw_power": 0.0,
                "actual_power": 0.0,
                "efficiency": 0.0,
                "is_night": True
            }

    def calculate_battery_status(self, solar_power: float, time_delta_hours: float) -> Dict:
        """Calculate battery status based on solar power input."""
        try:
            # Calculate net power (solar power minus load)
            net_power = solar_power - self.load_power
            
            # Calculate charge/discharge current
            if net_power > 0:  # CHARGING
                charge_current = net_power / REAL_BATTERY_SPECS["nominal_voltage"]
                charge_current = min(charge_current, REAL_BATTERY_SPECS["max_charge_current"])
                charge_efficiency = REAL_BATTERY_SPECS["charge_efficiency"]
                battery_power = net_power * charge_efficiency
                power_direction = "CHARGING"
                current_direction = "POSITIVE"
                
            else:  # DISCHARGING
                discharge_current = abs(net_power) / REAL_BATTERY_SPECS["nominal_voltage"]
                discharge_current = min(discharge_current, REAL_BATTERY_SPECS["max_discharge_current"])
                charge_current = -discharge_current
                charge_efficiency = 1.0
                battery_power = net_power
                power_direction = "DISCHARGING"
                current_direction = "NEGATIVE"
            
            # Calculate energy transfer and update SOC
            energy_change_wh = net_power * time_delta_hours * charge_efficiency
            battery_capacity_wh = REAL_BATTERY_SPECS["nominal_voltage"] * REAL_BATTERY_SPECS["capacity"]
            soc_change = (energy_change_wh / battery_capacity_wh) * 100
            
            self.battery_state_of_charge += soc_change
            self.battery_state_of_charge = max(20.0, min(100, self.battery_state_of_charge))
            
            # Calculate LiPo battery voltage based on SOC and charging state
            battery_voltage = self._calculate_lipo_voltage(self.battery_state_of_charge, net_power > 0)
            
            # Update battery voltage
            self.battery_voltage = max(REAL_BATTERY_SPECS["cutoff_voltage"], round(battery_voltage, 2))
            
            return {
                "battery_voltage": self.battery_voltage,
                "state_of_charge": round(self.battery_state_of_charge, 1),
                "charge_current": round(charge_current, 2),
                "charge_current_absolute": round(abs(charge_current), 2),
                "net_power": round(net_power, 2),
                "load_power": self.load_power,
                "battery_power": round(battery_power, 2),
                "power_direction": power_direction,
                "current_direction": current_direction,
                "battery_type": "LiPo"
            }
            
        except Exception as e:
            print(f"❌ Error calculating battery status: {e}")
            return {
                "battery_voltage": 12.2,
                "state_of_charge": 50.0,
                "charge_current": 0.0,
                "charge_current_absolute": 0.0,
                "net_power": 0.0,
                "load_power": self.load_power,
                "battery_power": 0.0,
                "power_direction": "IDLE",
                "current_direction": "ZERO",
                "battery_type": "LiPo"
            }

    def _calculate_lipo_voltage(self, soc: float, is_charging: bool) -> float:
        """Calculate LiPo battery voltage based on State of Charge."""
        soc_percent = soc / 100.0
        
        if is_charging:
            # Charging curve
            if soc >= 95:
                return REAL_BATTERY_SPECS["max_charge_voltage"]
            elif soc >= 80:
                return 12.4 + (soc - 80) * 0.01
            else:
                return 11.8 + (soc * 0.008)
        else:
            # Discharging curve (LiPo is relatively flat)
            if soc >= 80:
                return 12.4 - (100 - soc) * 0.01
            elif soc >= 50:
                return 12.2 - (80 - soc) * 0.006
            elif soc >= 30:
                return 12.0 - (50 - soc) * 0.01
            elif soc >= 20:
                return 11.6 - (30 - soc) * 0.04
            else:
                return REAL_BATTERY_SPECS["cutoff_voltage"]

    def update_load_power(self, new_load: float):
        """Update the load power consumption."""
        self.load_power = max(0, new_load)
    
    def update_time(self):
        """Update time tracking for battery calculations."""
        current_time = datetime.now()
        time_delta = current_time - self.last_update
        self.last_update = current_time
        return time_delta.total_seconds() / 3600.0

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
                
                # Create alerts table if it doesn't exist
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        timestamp TEXT,
                        location TEXT,
                        type TEXT,
                        severity TEXT,
                        message TEXT
                    )
                ''')
                
                df = pd.DataFrame([vars(alert) for alert in self.alerts])
                df.to_sql("alerts", conn, if_exists="append", index=False)
                conn.close()
                print(f"✅ Saved {len(self.alerts)} alerts for {location}")
            except Exception as e:
                print(f"❌ Error saving alerts for {location}: {e}")

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
            print(f"🌤️  Fetching weather data for lat={lat}, lon={lon}")
            response = requests.get(url, params=params, timeout=10)
            
            # Check if response is successful
            if response.status_code != 200:
                print(f"❌ Weather API returned status code: {response.status_code}")
                print(f"❌ Response: {response.text}")
                return None
                
            response.raise_for_status()
            data = response.json()
            
            # Validate the data structure
            if not isinstance(data, dict):
                print(f"❌ Weather data is not a dictionary: {type(data)}")
                return None
                
            print(f"✅ Weather data received: {data.get('name', 'Unknown')}")
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error fetching OpenWeather data: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error in weather data: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error fetching OpenWeather data: {e}")
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
                "lon": lon
            }
            print(f"☀️  Fetching NREL solar data")
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"❌ NREL API returned status code: {response.status_code}")
                return None
                
            response.raise_for_status()
            data = response.json()
            print("✅ NREL solar data received")
            
            self.solar_cache[cache_key] = {
                "timestamp": datetime.now(),
                "data": data
            }
            return data
        except Exception as e:
            print(f"❌ Error fetching NREL solar data: {e}")
            return None

    def calculate_solar_irradiance(self, weather_data: Dict, nrel_data: Dict) -> Dict:
        """Calculate comprehensive solar irradiance."""
        # Add validation at the start
        if not weather_data or not isinstance(weather_data, dict):
            print(f"❌ Invalid weather data type: {type(weather_data)}")
            return {"irradiance": 0.0, "dni": 0.0, "dhi": 0.0}
        
        current_hour = datetime.now().hour
        if not (6 <= current_hour <= 18):  # Night time
            return {"irradiance": 0.0, "dni": 0.0, "dhi": 0.0}
        
        try:
            # Get cloud coverage from OpenWeather with safe access
            clouds = 0
            if "clouds" in weather_data and isinstance(weather_data["clouds"], dict):
                clouds = weather_data["clouds"].get("all", 0)
            elif "clouds" in weather_data:
                clouds = weather_data["clouds"]  # In case it's directly a number
            
            cloud_factor = 1 - (clouds / 100) * 0.7
            
            # Get baseline values from NREL
            baseline = 1000
            if nrel_data and isinstance(nrel_data, dict) and "outputs" in nrel_data:
                if "avg_dni" in nrel_data["outputs"]:
                    baseline = nrel_data["outputs"]["avg_dni"].get("annual", 1000)
            
            # Adjust for time of day
            time_factor = np.sin((current_hour - 6) * np.pi / 12)
            time_factor = max(0, time_factor)
            
            # Calculate components
            dni = baseline * cloud_factor * time_factor
            dhi = dni * 0.2
            
            # Calculate total irradiance
            solar_angle = abs(np.cos((current_hour - 12) * np.pi / 12))
            total_irradiance = dni * solar_angle + dhi
            
            # Convert to Lux
            irradiance_lux = total_irradiance * 120
            
            return {
                "irradiance": max(0, min(180000, irradiance_lux)),
                "dni": max(0, min(1200, dni)),
                "dhi": max(0, min(500, dhi))
            }
        except Exception as e:
            print(f"❌ Error calculating solar irradiance: {e}")
            print(f"   Weather data type: {type(weather_data)}")
            print(f"   Weather data: {weather_data}")
            return {"irradiance": 0.0, "dni": 0.0, "dhi": 0.0}

    def get_solar_irradiance_from_api(self, lat: float, lon: float) -> Dict:
        """Get solar irradiance data from APIs with fallback calculation"""
        try:
            # Get weather data
            weather_data = self.get_weather(lat, lon)
            
            # Get NREL solar data
            nrel_data = self.get_nrel_solar_data(lat, lon, 0)
            
            # Calculate irradiance
            irradiance_data = self.calculate_solar_irradiance(weather_data, nrel_data)
            
            return irradiance_data
            
        except Exception as e:
            print(f"❌ Error getting solar irradiance from API: {e}")
            # Fallback calculation
            current_hour = datetime.now().hour
            if 6 <= current_hour <= 18:
                base_irradiance = 50000 + np.random.normal(0, 10000)
                return {
                    "irradiance": max(0, min(120000, base_irradiance)),
                    "dni": max(0, min(800, base_irradiance / 150)),
                    "dhi": max(0, min(400, base_irradiance / 300))
                }
            else:
                return {"irradiance": 0.0, "dni": 0.0, "dhi": 0.0}

class SolarSystem:
    def __init__(self, panel_power=100, battery_capacity=100, battery_voltage=12):
        self.panel_power = panel_power            # in watts
        self.battery_capacity = battery_capacity  # in Ah
        self.battery_voltage = battery_voltage    # in volts
        self.efficiency_status = "Unknown"
        self.irradiance_history = []              # store last few irradiance readings
        self.last_forecast_time = None

    def compute_efficiency(self, irradiance):
        """Classify solar efficiency level based on irradiance (Lux)."""
        max_irradiance = 1000  # approximate full sun = 1000 W/m² or ~100,000 lux (scaled)
        eff_ratio = (irradiance / max_irradiance) * 100

        if eff_ratio < 40:
            self.efficiency_status = "Low Efficiency ⚠️"
        elif eff_ratio < 70:
            self.efficiency_status = "Medium Efficiency ⚙️"
        else:
            self.efficiency_status = "High Efficiency ✅"

        return self.efficiency_status

    def compute_power_output(self, irradiance):
        """Compute power output based on current irradiance."""
        power_output = (irradiance / 1000) * self.panel_power
        return round(power_output, 2)

    def forecast_power_output(self):
        """
        Simple rolling-average forecast of power output for the next 5 hours.
        Uses last few irradiance values fetched from the API.
        """
        if len(self.irradiance_history) < 5:
            return None, None  # not enough data yet

        # Moving average of last 5 readings
        avg_current = sum(self.irradiance_history[-5:]) / 5
        avg_future = avg_current * 0.85  # assume 15% natural drop over 5 hours (clouds, sunset)
        forecast_power = (avg_future / 1000) * self.panel_power

        if forecast_power < 0.5 * ((self.irradiance_history[-1] / 1000) * self.panel_power):
            alert_msg = (
                "⚠️ Forecast indicates a decrease in solar output "
                "within the next 5 hours. Please reduce your load within 1–3 hours."
            )
        else:
            alert_msg = "✅ Solar power expected to remain stable for the next 5 hours."

        return round(forecast_power, 2), alert_msg
        
    def calculate_real_output(self, temp: float, humidity: float, irradiance_data: Dict, location: str) -> Dict:
        """Calculate real solar system output with actual panel and battery data"""
        try:
            # Calculate time delta for battery calculations
            time_delta_hours = self.real_data_simulator.update_time()
            
            # Calculate real solar panel output
            solar_output = self.real_data_simulator.calculate_solar_panel_output(
                irradiance_data["irradiance"], temp, humidity
            )
            
            # Calculate real battery status
            battery_status = self.real_data_simulator.calculate_battery_status(
                solar_output["actual_power"], time_delta_hours
            )
            
            # Generate alerts based on real conditions
            self._check_real_conditions(temp, humidity, solar_output, battery_status, location)
            
            return {
                "panel_voltage": solar_output["panel_voltage"],
                "panel_current": solar_output["panel_current"],
                "raw_power": solar_output["raw_power"],
                "actual_power": solar_output["actual_power"],
                "panel_efficiency": solar_output["efficiency"],
                "battery_voltage": battery_status["battery_voltage"],
                "state_of_charge": battery_status["state_of_charge"],
                "charge_current": battery_status["charge_current"],
                "net_power": battery_status["net_power"],
                "load_power": battery_status["load_power"],
                "battery_power": battery_status["battery_power"],
                "power_direction": battery_status["power_direction"],
                "efficiency_class": self._classify_efficiency(solar_output["actual_power"])
            }
        except Exception as e:
            print(f"❌ Error in real output calculation: {e}")
            return {
                "panel_voltage": 0.0,
                "panel_current": 0.0,
                "raw_power": 0.0,
                "actual_power": 0.0,
                "panel_efficiency": 0.0,
                "battery_voltage": 12.0,
                "state_of_charge": 50.0,
                "charge_current": 0.0,
                "net_power": 0.0,
                "load_power": 25.0,
                "battery_power": 0.0,
                "power_direction": "IDLE",
                "efficiency_class": 0
            }
    
    def _classify_efficiency(self, power: float) -> int:
        efficiency = power / REAL_SOLAR_SPECS["max_power"]
        if efficiency < 0.3:
            return 0  # Low
        elif efficiency < 0.7:
            return 1  # Medium
        return 2  # High
    
    def _check_real_conditions(self, temp: float, humidity: float, solar_output: Dict, battery_status: Dict, location: str):
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
        if solar_output["actual_power"] < ALERT_THRESHOLDS["power"]["low"]:
            self.alert_system.add_alert(
                location, "power", "warning",
                f"Low power output ({solar_output['actual_power']:.1f}W)"
            )
        
        # Battery alerts
        if battery_status["battery_voltage"] <= ALERT_THRESHOLDS["battery"]["critical"]:
            self.alert_system.add_alert(
                location, "battery", "critical",
                f"Critical battery voltage ({battery_status['battery_voltage']:.1f}V) - SOC: {battery_status['state_of_charge']:.1f}%"
            )
        elif battery_status["battery_voltage"] <= ALERT_THRESHOLDS["battery"]["warning"]:
            self.alert_system.add_alert(
                location, "battery", "warning",
                f"Low battery voltage ({battery_status['battery_voltage']:.1f}V) - SOC: {battery_status['state_of_charge']:.1f}%"
            )
            
class LoadManager:
    def __init__(self, alert_system: AlertSystem, real_data_simulator: RealDataSimulator):
        self.alert_system = alert_system
        self.real_data_simulator = real_data_simulator
        
    def get_recommendation(self, battery_voltage: float, state_of_charge: float, power_output: float, location: str) -> Dict:
        if battery_voltage <= ALERT_THRESHOLDS["battery"]["critical"]:
            self.alert_system.add_alert(
                location, "battery", "critical",
                "Critical battery level - shut down non-essential loads"
            )
            self.real_data_simulator.update_load_power(10.0)
            return {
                "status": "CRITICAL",
                "action": "Shut down non-essential loads",
                "max_load": 10,
                "recommended_action": "Disconnect all non-critical loads immediately"
            }
        elif battery_voltage <= ALERT_THRESHOLDS["battery"]["warning"]:
            self.alert_system.add_alert(
                location, "battery", "warning",
                "Low battery level - reduce power consumption"
            )
            self.real_data_simulator.update_load_power(15.0)
            return {
                "status": "WARNING",
                "action": "Reduce power consumption",
                "max_load": 15,
                "recommended_action": "Turn off unnecessary devices"
            }
        elif power_output < ALERT_THRESHOLDS["power"]["low"]:
            self.alert_system.add_alert(
                location, "power", "warning",
                "Low power generation - consider load reduction"
            )
            return {
                "status": "CAUTION",
                "action": "Monitor power consumption",
                "max_load": 25,
                "recommended_action": "Consider reducing load if generation remains low"
            }
        elif state_of_charge > 90:
            self.real_data_simulator.update_load_power(35.0)
            return {
                "status": "OPTIMAL",
                "action": "Normal operation - good generation",
                "max_load": 35,
                "recommended_action": "System operating optimally"
            }
        else:
            self.real_data_simulator.update_load_power(25.0)
            return {
                "status": "NORMAL",
                "action": "Normal operation",
                "max_load": 25,
                "recommended_action": "Monitor system performance"
            }

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
        solar_irradiance: float,
        solar_dni: float,
        solar_dhi: float,
        panel_voltage: float,
        panel_current: float,
        raw_power: float,
        actual_power: float,
        panel_efficiency: float,
        battery_voltage: float,
        state_of_charge: float,
        charge_current: float,
        net_power: float,
        load_power: float,
        efficiency_class: int,
        load_status: str,
        load_action: str,
        max_load: Union[float, None],
        recommended_action: str
    ):
        self.timestamp = timestamp
        self.temperature = temperature
        self.humidity = humidity
        self.solar_irradiance = solar_irradiance
        self.solar_dni = solar_dni
        self.solar_dhi = solar_dhi
        self.panel_voltage = panel_voltage
        self.panel_current = panel_current
        self.raw_power = raw_power
        self.actual_power = actual_power
        self.panel_efficiency = panel_efficiency
        self.battery_voltage = battery_voltage
        self.state_of_charge = state_of_charge
        self.charge_current = charge_current
        self.net_power = net_power
        self.load_power = load_power
        self.efficiency_class = efficiency_class
        self.load_status = load_status
        self.load_action = load_action
        self.max_load = max_load
        self.recommended_action = recommended_action

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Timestamp": self.timestamp,
            "Temperature": round(self.temperature, 2),
            "Humidity": round(self.humidity, 2),
            "Solar_Irradiance": round(self.solar_irradiance, 2),
            "Solar_DNI": round(self.solar_dni, 2),
            "Solar_DHI": round(self.solar_dhi, 2),
            "Panel_Voltage": round(self.panel_voltage, 2),
            "Panel_Current": round(self.panel_current, 2),
            "Raw_Power": round(self.raw_power, 2),
            "Actual_Power": round(self.actual_power, 2),
            "Panel_Efficiency": round(self.panel_efficiency, 3),
            "Battery_Voltage": round(self.battery_voltage, 2),
            "State_of_Charge": round(self.state_of_charge, 1),
            "Charge_Current": round(self.charge_current, 2),
            "Net_Power": round(self.net_power, 2),
            "Load_Power": round(self.load_power, 2),
            "Efficiency_Class": self.efficiency_class,
            "Load_Status": self.load_status,
            "Load_Action": self.load_action,
            "Max_Load": self.max_load,
            "Recommended_Action": self.recommended_action
        }

class MonitoringSystem:
    def __init__(self, openweather_key: str, nrel_key: str):
        self.weather_monitor = WeatherMonitor(openweather_key, nrel_key)
        self.alert_system = AlertSystem()
        self.historical_generator = HistoricalDataGenerator()
        self.real_data_simulators = {}
        self.solar_systems = {}
        self.load_managers = {}
        
        # Initialize systems for each location
        for location_name in LOCATIONS.keys():
            real_data_simulator = RealDataSimulator(location_name)
            self.real_data_simulators[location_name] = real_data_simulator
            self.solar_systems[location_name] = SolarSystem(self.alert_system, real_data_simulator)
            self.load_managers[location_name] = LoadManager(self.alert_system, real_data_simulator)
        
        # Initialize databases with historical data
        self.initialize_databases_with_historical_data()

    def initialize_databases_with_historical_data(self):
        """Initialize all databases with historical data from October 1, 2025"""
        print("🗃️  Initializing databases with 5-minute interval historical data from October 1, 2025...")
        
        for location_name, location_data in LOCATIONS.items():
            try:
                monitoring_db = f"solar_monitoring_{location_name.lower()}.db"
                
                # Always regenerate historical data to ensure 5-minute intervals
                print(f"📊 Generating 5-minute interval data for {location_name}...")
                
                # Use self.historical_generator
                historical_records = self.historical_generator.generate_historical_data_for_location(
                    location_name, 
                    location_data["battery_soc"],
                    location_data["climate_factor"]
                )
                
                # Create monitoring database and insert historical data
                conn = sqlite3.connect(monitoring_db)
                
                # Drop existing table to ensure clean data
                conn.execute("DROP TABLE IF EXISTS monitoring")
                
                # Fixed schema with Rainfall column
                conn.execute('''
                    CREATE TABLE monitoring (
                        Timestamp TEXT PRIMARY KEY,
                        Temperature REAL,
                        Humidity REAL,
                        Rainfall REAL,
                        Solar_Irradiance REAL,
                        Solar_DNI REAL,
                        Solar_DHI REAL,
                        Panel_Voltage REAL,
                        Panel_Current REAL,
                        Raw_Power REAL,
                        Actual_Power REAL,
                        Panel_Efficiency REAL,
                        Battery_Voltage REAL,
                        State_of_Charge REAL,
                        Charge_Current REAL,
                        Net_Power REAL,
                        Load_Power REAL,
                        Battery_Power REAL,
                        Power_Direction TEXT,
                        Efficiency_Class INTEGER,
                        Load_Status TEXT,
                        Load_Action TEXT,
                        Max_Load REAL,
                        Recommended_Action TEXT
                    )
                ''')
                
                if historical_records:
                    # Insert in batches to avoid memory issues
                    batch_size = 1000
                    for i in range(0, len(historical_records), batch_size):
                        batch = historical_records[i:i + batch_size]
                        df = pd.DataFrame(batch)
                        df.to_sql("monitoring", conn, if_exists="append", index=False)
                        print(f"  ✅ Inserted batch {i//batch_size + 1}/{(len(historical_records)-1)//batch_size + 1}")
                    
                    # Verify data was inserted
                    count = conn.execute("SELECT COUNT(*) FROM monitoring").fetchone()[0]
                    print(f"✅ Added {count} historical records to {monitoring_db}")
                    
                    # Show date range verification
                    first_date = conn.execute("SELECT MIN(Timestamp) FROM monitoring").fetchone()[0]
                    last_date = conn.execute("SELECT MAX(Timestamp) FROM monitoring").fetchone()[0]
                    print(f"   📅 Date range: {first_date} to {last_date}")
                    
                else:
                    print(f"❌ No historical records generated for {location_name}")
                
                conn.close()
                
                # Initialize alerts database
                alerts_db = f"solar_alerts_{location_name.lower()}.db"
                conn = sqlite3.connect(alerts_db)
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        timestamp TEXT,
                        location TEXT,
                        type TEXT,
                        severity TEXT,
                        message TEXT
                    )
                ''')
                conn.close()
                print(f"✅ Alerts database initialized: {alerts_db}")
                
            except Exception as e:
                print(f"❌ Error initializing database for {location_name}: {e}")
                import traceback
                traceback.print_exc()

    def monitor_location(self, location_name: str, coords: LocationData) -> Union[MonitoringData, None]:
        """Monitor solar system at a specific location with real API data."""
        try:
            print(f"\n--- Monitoring {location_name} ---")
            
            # Get REAL data from APIs
            weather_data = self.weather_monitor.get_weather(coords.lat, coords.lon)
            
            # Extract REAL temperature and humidity from API
            if weather_data and isinstance(weather_data, dict):
                main_data = weather_data.get("main", {})
                temp = main_data.get("temp", 28.0)  # REAL temperature from API
                humidity = main_data.get("humidity", 70.0)  # REAL humidity from API
                rainfall = weather_data.get("rain", {}).get("1h", 0.0)
            else:
                print(f"❌ Using fallback data for {location_name}")
                temp = 28.0 + np.random.normal(0, 2)
                humidity = 70.0 + np.random.normal(0, 10)
                rainfall = 0.0
            
            print(f"🌡️  API Temperature: {temp}°C")
            print(f"💧 API Humidity: {humidity}%")
            
            # Get REAL solar irradiance from API
            irradiance_data = self.weather_monitor.get_solar_irradiance_from_api(coords.lat, coords.lon)
            print(f"☀️  API Solar Irradiance: {irradiance_data['irradiance']:.0f} Lux")
            
            # Calculate time delta for battery
            time_delta_hours = self.real_data_simulators[location_name].update_time()
            
            # AUTOMATICALLY COMPUTE solar panel output based on API data
            solar_output = self.real_data_simulators[location_name].calculate_solar_panel_output(
                irradiance_data["irradiance"], temp, humidity
            )
            
            # AUTOMATICALLY COMPUTE battery status based on solar output
            battery_status = self.real_data_simulators[location_name].calculate_battery_status(
                solar_output["actual_power"], time_delta_hours
            )
            
            # Get load recommendations
            load_rec = self.load_managers[location_name].get_recommendation(
                battery_status["battery_voltage"],
                battery_status["state_of_charge"],
                solar_output["actual_power"],
                location_name
            )
            
            # Create monitoring data with ALL computed values
            monitoring_data = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Temperature": temp,                    # FROM API
                "Humidity": humidity,                   # FROM API
                "Rainfall": rainfall,                   # FROM API
                "Solar_Irradiance": irradiance_data["irradiance"],  # FROM API
                "Solar_DNI": irradiance_data["dni"],    # FROM API
                "Solar_DHI": irradiance_data["dhi"],    # FROM API
                # COMPUTED VALUES:
                "Panel_Voltage": solar_output["panel_voltage"],
                "Panel_Current": solar_output["panel_current"],
                "Raw_Power": solar_output["raw_power"],
                "Actual_Power": solar_output["actual_power"],
                "Panel_Efficiency": solar_output["efficiency"],
                "Battery_Voltage": battery_status["battery_voltage"],
                "State_of_Charge": battery_status["state_of_charge"],
                "Charge_Current": battery_status["charge_current"],
                "Net_Power": battery_status["net_power"],
                "Load_Power": battery_status["load_power"],
                "Battery_Power": battery_status["battery_power"],
                "Power_Direction": battery_status["power_direction"],
                "Efficiency_Class": self._classify_efficiency(solar_output["actual_power"]),
                "Load_Status": load_rec["status"],
                "Load_Action": load_rec["action"],
                "Max_Load": load_rec["max_load"],
                "Recommended_Action": load_rec["recommended_action"]
            }
            
            # Save to database and display results
            self._save_and_display_data(location_name, monitoring_data, solar_output, battery_status)
            
            # Convert to MonitoringData object for return
            return MonitoringData(
                timestamp=monitoring_data["Timestamp"],
                temperature=monitoring_data["Temperature"],
                humidity=monitoring_data["Humidity"],
                solar_irradiance=monitoring_data["Solar_Irradiance"],
                solar_dni=monitoring_data["Solar_DNI"],
                solar_dhi=monitoring_data["Solar_DHI"],
                panel_voltage=monitoring_data["Panel_Voltage"],
                panel_current=monitoring_data["Panel_Current"],
                raw_power=monitoring_data["Raw_Power"],
                actual_power=monitoring_data["Actual_Power"],
                panel_efficiency=monitoring_data["Panel_Efficiency"],
                battery_voltage=monitoring_data["Battery_Voltage"],
                state_of_charge=monitoring_data["State_of_Charge"],
                charge_current=monitoring_data["Charge_Current"],
                net_power=monitoring_data["Net_Power"],
                load_power=monitoring_data["Load_Power"],
                efficiency_class=monitoring_data["Efficiency_Class"],
                load_status=monitoring_data["Load_Status"],
                load_action=monitoring_data["Load_Action"],
                max_load=monitoring_data["Max_Load"],
                recommended_action=monitoring_data["Recommended_Action"]
            )
            
        except Exception as e:
            print(f"❌ Error monitoring {location_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_and_display_data(self, location_name: str, monitoring_data: Dict, solar_output: Dict, battery_status: Dict):
        """Save data to database and display results."""
        # Save to database
        try:
            db_name = f"solar_monitoring_{location_name.lower()}.db"
            conn = sqlite3.connect(db_name)
            df = pd.DataFrame([monitoring_data])
            df.to_sql("monitoring", conn, if_exists="append", index=False)
            
            count = conn.execute("SELECT COUNT(*) FROM monitoring").fetchone()[0]
            print(f"💾 Real-time data saved to database: {db_name} (Total records: {count})")
            conn.close()
        except Exception as e:
            print(f"❌ Database error for {location_name}: {e}")
        
        # Save alerts
        self.alert_system.save_alerts(location_name)
        
        # Display COMPUTED results
        print(f"⚡ COMPUTED SOLAR: {solar_output['actual_power']:.1f}W "
              f"({solar_output['panel_voltage']:.1f}V, {solar_output['panel_current']:.2f}A)")
        print(f"🔋 COMPUTED BATTERY: {battery_status['battery_voltage']:.1f}V, "
              f"SOC: {battery_status['state_of_charge']:.1f}%")
        print(f"🔌 LOAD: {battery_status['load_power']:.1f}W")
        print(f"🔄 POWER FLOW: {battery_status['power_direction']} "
              f"({battery_status['battery_power']:.1f}W)")
        print(f"🔋 BATTERY CURRENT: {battery_status['charge_current']:.2f}A")

    def _classify_efficiency(self, power: float) -> int:
        """Classify efficiency based on power output"""
        efficiency = power / REAL_SOLAR_SPECS["max_power"]
        if efficiency < 0.3:
            return 0  # Low
        elif efficiency < 0.7:
            return 1  # Medium
        return 2  # High

def main():
    """Main function to run the solar monitoring system with real data."""
    print("🚀 Initializing Solar Monitoring System with Real 100W Data...")
    print(f"🔋 Real Solar Panel: {REAL_SOLAR_SPECS['max_power']}W Monocrystalline")
    print(f"🔋 Real Battery: {REAL_BATTERY_SPECS['capacity']}Ah {REAL_BATTERY_SPECS['nominal_voltage']}V LiPo")
    print(f"📍 Monitoring {len(LOCATIONS)} locations")
    print(f"📅 Historical data from: October 1, 2025 to present")
    
    # Initialize monitoring system (this will populate historical data)
    monitoring_system = MonitoringSystem(OPENWEATHER_API_KEY, NREL_API_KEY)
    
    print("\n🎯 Starting real-time monitoring...")
    iteration = 0
    while True:
        iteration += 1
        print("\n" + "="*60)
        print(f"🔆 Real Solar Monitoring Update #{iteration}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Process each location
        all_data = {}
        
        for location_name, coords_dict in LOCATIONS.items():
            print(f"\n📍 Monitoring {location_name}...")
            
            # Convert dictionary to LocationData object
            coords = LocationData(
                lat=coords_dict["lat"],
                lon=coords_dict["lon"],
                altitude=coords_dict["altitude"]
            )
            
            # Monitor location with real data
            data = monitoring_system.monitor_location(location_name, coords)
            
            if data:
                all_data[location_name] = data
                # Print real monitoring data
                print(f"  🌡️  Temperature: {data.temperature:.1f}°C")
                print(f"  💧 Humidity: {data.humidity:.1f}%")
                print(f"  ☀️  Solar Irradiance: {data.solar_irradiance:.0f} Lux")
                print(f"  ⚡ Panel Voltage: {data.panel_voltage:.1f}V")
                print(f"  🔌 Panel Current: {data.panel_current:.2f}A")
                print(f"  💡 Actual Power: {data.actual_power:.1f}W")
                print(f"  🔋 Battery Voltage: {data.battery_voltage:.1f}V")
                print(f"  📊 State of Charge: {data.state_of_charge:.1f}%")
                print(f"  🔌 Load Power: {data.load_power:.1f}W")
                print(f"  📈 Efficiency: {['Low', 'Medium', 'High'][data.efficiency_class]}")
                print(f"  🚦 System Status: {data.load_status}")
            
            # Print current alerts
            alerts = monitoring_system.alert_system.alerts
            if alerts:
                print("  🔔 Current Alerts:")
                for alert in alerts:
                    print(f"    ⚠️ {alert.severity.upper()}: {alert.message}")
            
            # Clear alerts for next iteration
            monitoring_system.alert_system.clear_alerts()
        
        # Summary
        print(f"\n📊 Summary - {len(all_data)}/{len(LOCATIONS)} locations monitored with real data")
        
        # Wait for next update (5 minutes)
        print(f"\n⏰ Next update in 5 minutes...")
        time.sleep(300)  # 5 minutes

def run_monitoring():
    """Run the solar monitoring system with error handling."""
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error in monitoring system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_monitoring()