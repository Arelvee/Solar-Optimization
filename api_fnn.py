from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import sqlite3
import requests
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import logging
import json
import os
from dataclasses import dataclass

# ------------------------
# CONFIGURATION & SETUP
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys from config
def get_api_keys():
    """Get API keys from config file"""
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
            openweather_key = config.get("OPENWEATHER_API_KEY")
            nrel_key = config.get("NREL_API_KEY")
            
        if not openweather_key or not nrel_key:
            raise ValueError("API keys not found in config file")
            
        logger.info("✅ API keys loaded successfully from config.json")
        return openweather_key, nrel_key
        
    except Exception as e:
        logger.error(f"❌ Error loading config file: {e}")
        return None, None

OPENWEATHER_API_KEY, NREL_API_KEY = get_api_keys()

# Real Solar Panel Specifications (Monocrystalline)
REAL_SOLAR_SPECS = {
    "max_power": 100.0,
    "voltage_max_power": 18.5,
    "current_max_power": 5.41,
    "open_circuit_voltage": 22.3,
    "short_circuit_current": 5.86,
    "temp_coefficient_voc": -0.0032,
    "temp_coefficient_isc": 0.0005,
    "nominal_temp": 25.0,
    "efficiency": 0.195,
    "panel_area": 0.54
}

# Real Battery Specifications (12V 20Ah Lead Acid)
REAL_BATTERY_SPECS = {
    "nominal_voltage": 12.0,
    "capacity": 20.0,
    "max_charge_voltage": 14.4,
    "float_voltage": 13.6,
    "cutoff_voltage": 10.5,
    "charge_efficiency": 0.85,
    "depth_of_discharge": 0.5,
}

# System Configuration
PV_SYSTEM_CONFIG = {
    "system_capacity": 0.1,
    "module_type": 1,
    "array_type": 1,
    "tilt": 14,
    "azimuth": 180,
    "losses": 14.08
}

LOCATIONS = {
    "Sto_Tomas": {
        "lat": 14.1119, 
        "lon": 121.1483,
        "altitude": 300,
        "battery_soc": 80.0,
        "climate_factor": 1.0
    },
    "LSPU_San_Pablo": {
        "lat": 14.0689, 
        "lon": 121.3256,
        "altitude": 450,
        "battery_soc": 75.0,
        "climate_factor": 0.95
    },
    "San_Pablo_City": {
        "lat": 14.0777, 
        "lon": 121.3257,
        "altitude": 430,
        "battery_soc": 85.0,
        "climate_factor": 1.05
    }
}

# Alert Thresholds
ALERT_THRESHOLDS = {
    "temperature": {
        "critical": 45.0,
        "warning": 40.0
    },
    "humidity": {
        "high": 85.0,
        "warning": 75.0
    },
    "power": {
        "low": REAL_SOLAR_SPECS["max_power"] * 0.3,
        "medium": REAL_SOLAR_SPECS["max_power"] * 0.7
    },
    "battery": {
        "critical": REAL_BATTERY_SPECS["cutoff_voltage"],
        "warning": 11.8,
        "full": REAL_BATTERY_SPECS["float_voltage"]
    },
    "solar_voltage": {
        "low": REAL_SOLAR_SPECS["voltage_max_power"] * 0.6,
        "high": REAL_SOLAR_SPECS["open_circuit_voltage"] * 1.1
    }
}

# ------------------------
# PYDANTIC MODELS
# ------------------------
@dataclass
class Alert:
    timestamp: str
    location: str
    type: str
    severity: str
    message: str

class SensorInput(BaseModel):
    temperature: float = Field(..., ge=-50, le=60, description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    solar_voltage: float = Field(..., ge=0, description="Solar voltage in volts")
    solar_current: float = Field(..., ge=0, description="Solar current in amps")
    solar_irradiance: float = Field(..., ge=0, description="Solar irradiance in W/m²")
    battery_voltage: float = Field(..., ge=0, description="Battery voltage in volts")
    battery_current: float = Field(..., description="Battery current in amps")
    power_output: float = Field(..., ge=0, description="Power output in watts")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_type: int = Field(..., ge=0, le=1, description="Day type (0=weekday, 1=weekend)")

class PredictionResponse(BaseModel):
    efficiency: float
    power: float

class MonitoringData(BaseModel):
    timestamp: str
    temperature: float
    humidity: float
    solar_irradiance: float
    solar_dni: float
    solar_dhi: float
    panel_voltage: float
    panel_current: float
    raw_power: float
    actual_power: float
    panel_efficiency: float
    battery_voltage: float
    state_of_charge: float
    charge_current: float
    net_power: float
    load_power: float
    efficiency_class: int
    load_status: str
    load_action: str
    max_load: float
    recommended_action: str

class LocationData(BaseModel):
    lat: float
    lon: float
    altitude: float

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

# ------------------------
# FASTAPI APP
# ------------------------
app = FastAPI(
    title="Solar Monitoring & Prediction API",
    description="Real-time solar monitoring with FNN predictions based on sensor data",
    version="1.0.0"
)

# ------------------------
# MODEL LOADING
# ------------------------
try:
    interpreter = tf.lite.Interpreter(model_path="fnn_model_float32.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    scaler = joblib.load("scaler_fnn.pkl")
    logger.info("✅ Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading model/scaler: {e}")
    # Don't raise here, allow API to start without model for monitoring features

# ------------------------
# CORE CLASSES (from solar_monitoring.py)
# ------------------------
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
                logger.info(f"✅ Saved {len(self.alerts)} alerts for {location}")
            except Exception as e:
                logger.error(f"❌ Error saving alerts for {location}: {e}")

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
            logger.info(f"🌤️  Fetching weather data for lat={lat}, lon={lon}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.info(f"✅ Weather data received: {data.get('name', 'Unknown')}")
            return data
        except Exception as e:
            logger.error(f"❌ Error fetching OpenWeather data: {e}")
            return None

# Initialize core systems
alert_system = AlertSystem()
weather_monitor = WeatherMonitor(OPENWEATHER_API_KEY, NREL_API_KEY)

# ------------------------
# API ENDPOINTS
# ------------------------
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Solar Monitoring & Prediction API",
        "version": "1.0.0",
        "description": "Real-time solar monitoring with machine learning predictions",
        "endpoints": {
            "prediction": "POST /predict - Get solar efficiency and power predictions",
            "monitoring": "GET /monitoring/{location} - Get current monitoring data",
            "historical": "GET /historical/{location} - Get historical data",
            "alerts": "GET /alerts/{location} - Get current alerts",
            "health": "GET /health - API health check",
            "locations": "GET /locations - Get all monitored locations"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connections
        db_status = {}
        for location in LOCATIONS.keys():
            monitoring_db = f"solar_monitoring_{location.lower()}.db"
            alerts_db = f"solar_alerts_{location.lower()}.db"
            
            monitoring_exists = os.path.exists(monitoring_db)
            alerts_exists = os.path.exists(alerts_db)
            
            db_status[location] = {
                "monitoring": "exists" if monitoring_exists else "missing",
                "alerts": "exists" if alerts_exists else "missing"
            }
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": interpreter is not None,
            "scaler_loaded": scaler is not None,
            "api_keys_loaded": OPENWEATHER_API_KEY is not None and NREL_API_KEY is not None,
            "databases": db_status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/locations")
async def get_locations():
    """Get all monitored locations"""
    return {
        "locations": LOCATIONS,
        "count": len(LOCATIONS)
    }

@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def predict(data: SensorInput):
    """Get solar efficiency and power predictions from FNN model"""
    try:
        if interpreter is None or scaler is None:
            raise HTTPException(
                status_code=500,
                detail="Model or scaler not loaded"
            )

        # Prepare input features
        features = [
            data.temperature, data.humidity, data.solar_voltage,
            data.solar_current, data.solar_irradiance, data.battery_voltage,
            data.battery_current, data.power_output, data.hour, data.day_type
        ]
        
        X = np.array([features])
        X_scaled = scaler.transform(X)

        # Make prediction
        interpreter.set_tensor(input_details[0]['index'], X_scaled.astype(np.float32))
        interpreter.invoke()
        y_pred = interpreter.get_tensor(output_details[0]['index'])

        efficiency_pred = float(y_pred[0][0])
        power_pred = float(y_pred[0][1])

        logger.info(f"✅ Prediction made - Efficiency: {efficiency_pred:.4f}, Power: {power_pred:.4f}")

        return PredictionResponse(efficiency=efficiency_pred, power=power_pred)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/monitoring/{location}")
async def get_current_monitoring(location: str):
    """Get current monitoring data for a specific location"""
    if location not in LOCATIONS:
        raise HTTPException(status_code=404, detail="Location not found")
    
    try:
        db_name = f"solar_monitoring_{location.lower()}.db"
        if not os.path.exists(db_name):
            raise HTTPException(status_code=404, detail="Monitoring database not found")
        
        conn = sqlite3.connect(db_name)
        df = pd.read_sql_query(
            "SELECT * FROM monitoring ORDER BY Timestamp DESC LIMIT 1", 
            conn
        )
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No monitoring data available")
        
        record = df.iloc[0].to_dict()
        
        # Convert numeric fields
        numeric_fields = ['Temperature', 'Humidity', 'Solar_Irradiance', 'Solar_DNI', 'Solar_DHI',
                         'Panel_Voltage', 'Panel_Current', 'Raw_Power', 'Actual_Power', 'Panel_Efficiency',
                         'Battery_Voltage', 'State_of_Charge', 'Charge_Current', 'Net_Power', 'Load_Power',
                         'Efficiency_Class']
        
        for field in numeric_fields:
            if field in record and record[field] is not None:
                try:
                    record[field] = float(record[field])
                except (ValueError, TypeError):
                    record[field] = 0.0
        
        return record
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting monitoring data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring data: {str(e)}")

@app.get("/historical/{location}")
async def get_historical_data(
    location: str,
    start: Optional[str] = Query(None, description="Start datetime (YYYY-MM-DD HH:MM:SS)"),
    end: Optional[str] = Query(None, description="End datetime (YYYY-MM-DD HH:MM:SS)"),
    limit: int = Query(1000, ge=1, le=10000, description="Limit number of records")
):
    """Get historical monitoring data for a specific location"""
    if location not in LOCATIONS:
        raise HTTPException(status_code=404, detail="Location not found")
    
    try:
        # Validate date formats
        if start:
            try:
                datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start date format. Use YYYY-MM-DD HH:MM:SS")
        
        if end:
            try:
                datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end date format. Use YYYY-MM-DD HH:MM:SS")

        db_name = f"solar_monitoring_{location.lower()}.db"
        if not os.path.exists(db_name):
            raise HTTPException(status_code=404, detail="Monitoring database not found")
        
        conn = sqlite3.connect(db_name)
        
        query = "SELECT * FROM monitoring WHERE 1=1"
        params = []

        if start:
            query += " AND Timestamp >= ?"
            params.append(start)
        if end:
            query += " AND Timestamp <= ?"
            params.append(end)
            
        query += " ORDER BY Timestamp DESC LIMIT ?"
        params.append(limit)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        records = []
        for _, row in df.iterrows():
            record = row.to_dict()
            
            numeric_fields = ['Temperature', 'Humidity', 'Solar_Irradiance', 'Solar_DNI', 'Solar_DHI',
                             'Panel_Voltage', 'Panel_Current', 'Raw_Power', 'Actual_Power', 'Panel_Efficiency',
                             'Battery_Voltage', 'State_of_Charge', 'Charge_Current', 'Net_Power', 'Load_Power',
                             'Efficiency_Class']
            
            for field in numeric_fields:
                if field in record and record[field] is not None:
                    try:
                        record[field] = float(record[field])
                    except (ValueError, TypeError):
                        record[field] = 0.0
            
            records.append(record)
        
        return {
            "location": location,
            "count": len(records),
            "data": records
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting historical data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get historical data: {str(e)}")

@app.get("/alerts/{location}")
async def get_alerts(
    location: str,
    hours: int = Query(24, ge=1, le=168, description="Hours of alerts to retrieve (1-168)")
):
    """Get alerts for a specific location"""
    if location not in LOCATIONS:
        raise HTTPException(status_code=404, detail="Location not found")
    
    try:
        db_name = f"solar_alerts_{location.lower()}.db"
        if not os.path.exists(db_name):
            return {"location": location, "alerts": [], "count": 0}
        
        conn = sqlite3.connect(db_name)
        df = pd.read_sql_query(
            f"SELECT * FROM alerts WHERE timestamp >= datetime('now', '-{hours} hours') ORDER BY timestamp DESC",
            conn
        )
        conn.close()
        
        alerts = df.to_dict('records')
        
        return {
            "location": location,
            "alerts": alerts,
            "count": len(alerts)
        }
    
    except Exception as e:
        logger.error(f"❌ Error getting alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@app.get("/all-locations/current")
async def get_all_locations_current():
    """Get current data from all locations"""
    data = {}
    for location in LOCATIONS.keys():
        try:
            db_name = f"solar_monitoring_{location.lower()}.db"
            if os.path.exists(db_name):
                conn = sqlite3.connect(db_name)
                df = pd.read_sql_query(
                    "SELECT * FROM monitoring ORDER BY Timestamp DESC LIMIT 1", 
                    conn
                )
                conn.close()
                
                if not df.empty:
                    record = df.iloc[0].to_dict()
                    data[location] = record
        except Exception as e:
            logger.warning(f"⚠️ Could not get data for {location}: {e}")
            continue
    
    return {
        "timestamp": datetime.now().isoformat(),
        "locations": data,
        "count": len(data)
    }

# ------------------------
# STARTUP EVENT
# ------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize databases on startup"""
    logger.info("🚀 Starting Solar Monitoring API...")
    logger.info(f"📍 Monitoring {len(LOCATIONS)} locations")
    logger.info(f"🔋 Real Solar Panel: {REAL_SOLAR_SPECS['max_power']}W Monocrystalline")
    logger.info(f"🔋 Real Battery: {REAL_BATTERY_SPECS['capacity']}Ah {REAL_BATTERY_SPECS['nominal_voltage']}V Lead Acid")
    
    # Initialize databases
    for location in LOCATIONS.keys():
        monitoring_db = f"solar_monitoring_{location.lower()}.db"
        alerts_db = f"solar_alerts_{location.lower()}.db"
        
        # Initialize monitoring database
        try:
            conn = sqlite3.connect(monitoring_db)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS monitoring (
                    Timestamp TEXT,
                    Temperature REAL,
                    Humidity REAL,
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
                    Efficiency_Class INTEGER,
                    Load_Status TEXT,
                    Load_Action TEXT,
                    Max_Load REAL,
                    Recommended_Action TEXT
                )
            ''')
            conn.close()
            logger.info(f"✅ Monitoring database: {monitoring_db}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {monitoring_db}: {e}")
            
        # Initialize alerts database
        try:
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
            logger.info(f"✅ Alerts database: {alerts_db}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {alerts_db}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)