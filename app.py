from flask import Flask, render_template, request, jsonify, flash, send_from_directory
import requests
from datetime import datetime, timedelta
import logging
import os
import sqlite3
import pandas as pd
import json
import joblib
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Union, Any

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "solar-monitoring-secret-key")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------
# CONFIGURATION & MODEL LOADING
# ------------------------
# Load API keys from config
def get_api_keys():
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
            return config.get("OPENWEATHER_API_KEY"), config.get("NREL_API_KEY")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return None, None

OPENWEATHER_API_KEY, NREL_API_KEY = get_api_keys()

# Load FNN Model and Scaler
try:
    interpreter = tf.lite.Interpreter(model_path="fnn_model_float32.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    scaler = joblib.load("scaler_fnn.pkl")
    logger.info("✅ FNN Model and scaler loaded successfully")
    MODEL_LOADED = True
except Exception as e:
    logger.error(f"❌ Error loading model/scaler: {e}")
    MODEL_LOADED = False

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

# Database Configuration
LOCATIONS = {
    'sto_tomas': {
        'name': 'Sto Tomas',
        'lat': 14.1119,
        'lon': 121.1483,
        'altitude': 300,
        'battery_soc': 80.0,
        'climate_factor': 1.0
    },
    'lspu_san_pablo': {
        'name': 'LSPU San Pablo',
        'lat': 14.0689,
        'lon': 121.3256,
        'altitude': 450,
        'battery_soc': 75.0,
        'climate_factor': 0.95
    },
    'san_pablo_city': {
        'name': 'San Pablo City',
        'lat': 14.0777,
        'lon': 121.3257,
        'altitude': 430,
        'battery_soc': 85.0,
        'climate_factor': 1.05
    }
}

# ------------------------
# DATABASE FUNCTIONS
# ------------------------
def initialize_database(db_file):
    """Initialize database with proper table structure"""
    try:
        conn = sqlite3.connect(db_file)
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
        logger.info(f"Database initialized: {db_file}")
        return True
    except Exception as e:
        logger.error(f"Error initializing database {db_file}: {e}")
        return False

def initialize_alerts_database(db_file):
    """Initialize alerts database"""
    try:
        conn = sqlite3.connect(db_file)
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
        logger.info(f"Alerts database initialized: {db_file}")
        return True
    except Exception as e:
        logger.error(f"Error initializing alerts database {db_file}: {e}")
        return False

def initialize_predictions_database(db_file):
    """Initialize predictions database for FNN results"""
    try:
        conn = sqlite3.connect(db_file)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                timestamp TEXT,
                location TEXT,
                temperature REAL,
                humidity REAL,
                solar_voltage REAL,
                solar_current REAL,
                solar_irradiance REAL,
                battery_voltage REAL,
                battery_current REAL,
                power_output REAL,
                hour INTEGER,
                day_type INTEGER,
                predicted_efficiency REAL,
                predicted_power REAL
            )
        ''')
        conn.close()
        logger.info(f"Predictions database initialized: {db_file}")
        return True
    except Exception as e:
        logger.error(f"Error initializing predictions database {db_file}: {e}")
        return False

def get_db_connection(db_file):
    """Create a database connection to the SQLite database"""
    if not os.path.exists(db_file):
        # Initialize database if it doesn't exist
        if "alerts" in db_file:
            initialize_alerts_database(db_file)
        elif "predictions" in db_file:
            initialize_predictions_database(db_file)
        else:
            initialize_database(db_file)
    
    try:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database {db_file}: {e}")
        return None

# Initialize all databases on startup
for location in LOCATIONS.keys():
    monitoring_db = f"solar_monitoring_{location}.db"
    alerts_db = f"solar_alerts_{location}.db"
    predictions_db = f"solar_predictions_{location}.db"
    
    initialize_database(monitoring_db)
    initialize_alerts_database(alerts_db)
    initialize_predictions_database(predictions_db)

# ------------------------
# FNN PREDICTION FUNCTIONS
# ------------------------
def validate_prediction_input(data):
    """Validate prediction input data"""
    errors = []
    
    if not (-50 <= data.get('temperature', 0) <= 60):
        errors.append("Temperature must be between -50°C and 60°C")
    
    if not (0 <= data.get('humidity', 0) <= 100):
        errors.append("Humidity must be between 0% and 100%")
    
    if data.get('solar_voltage', 0) < 0:
        errors.append("Solar voltage cannot be negative")
    
    if data.get('solar_current', 0) < 0:
        errors.append("Solar current cannot be negative")
    
    if data.get('solar_irradiance', 0) < 0:
        errors.append("Solar irradiance cannot be negative")
    
    if data.get('battery_voltage', 0) < 0:
        errors.append("Battery voltage cannot be negative")
    
    if data.get('power_output', 0) < 0:
        errors.append("Power output cannot be negative")
    
    if not (0 <= data.get('hour', 0) <= 23):
        errors.append("Hour must be between 0 and 23")
    
    if data.get('day_type', 0) not in [0, 1]:
        errors.append("Day type must be 0 (weekday) or 1 (weekend)")
    
    return errors

def make_prediction(data):
    """Make prediction using FNN model"""
    if not MODEL_LOADED:
        return None, "Model not loaded"
    
    try:
        # Prepare input features
        features = [
            data['temperature'], data['humidity'], data['solar_voltage'],
            data['solar_current'], data['solar_irradiance'], data['battery_voltage'],
            data['battery_current'], data['power_output'], data['hour'], data['day_type']
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

        return {
            "efficiency": efficiency_pred,
            "power": power_pred
        }, None
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        return None, f"Prediction failed: {str(e)}"

def save_prediction(location, input_data, prediction):
    """Save prediction to database"""
    try:
        db_name = f"solar_predictions_{location}.db"
        conn = get_db_connection(db_name)
        
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, location, temperature, humidity, solar_voltage, solar_current, 
             solar_irradiance, battery_voltage, battery_current, power_output, hour, day_type,
             predicted_efficiency, predicted_power)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            location,
            input_data['temperature'],
            input_data['humidity'],
            input_data['solar_voltage'],
            input_data['solar_current'],
            input_data['solar_irradiance'],
            input_data['battery_voltage'],
            input_data['battery_current'],
            input_data['power_output'],
            input_data['hour'],
            input_data['day_type'],
            prediction['efficiency'],
            prediction['power']
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Prediction saved for {location}")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving prediction: {e}")
        return False

# ------------------------
# FLASK ROUTES
# ------------------------
@app.route("/")
def dashboard():
    """Main dashboard page"""
    return render_template("dashboard.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for FNN predictions"""
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate input
        errors = validate_prediction_input(data)
        if errors:
            return jsonify({"error": "Validation failed", "details": errors}), 400
        
        # Make prediction
        prediction, error = make_prediction(data)
        if error:
            return jsonify({"error": error}), 500
        
        # Save prediction if location provided
        location = data.get('location')
        if location and location in LOCATIONS:
            save_prediction(location, data, prediction)
        
        return jsonify({
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Prediction API error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/api/predictions/<location>")
def api_predictions(location):
    """API endpoint for historical predictions"""
    if location not in LOCATIONS:
        return jsonify({"error": "Location not found"}), 404
    
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    limit = int(request.args.get('limit', 1000))
    
    db_name = f"solar_predictions_{location}.db"
    conn = get_db_connection(db_name)
    if not conn:
        return jsonify([])
    
    try:
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.replace('T', ' '))
        
        if end_date:
            query += " AND timestamp <= ?" 
            params.append(end_date.replace('T', ' '))
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        records = df.to_dict('records')
        return jsonify(records)
    
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500

@app.route("/api/current/<location>")
def api_current_data(location):
    """API endpoint for current data"""
    if location not in LOCATIONS:
        return jsonify({"error": "Location not found"}), 404
    
    db_name = f"solar_monitoring_{location}.db"
    conn = get_db_connection(db_name)
    if not conn:
        return jsonify({"error": "Database not found"}), 404
    
    try:
        # Get the latest record
        df = pd.read_sql_query(
            "SELECT * FROM monitoring ORDER BY Timestamp DESC LIMIT 1", 
            conn
        )
        conn.close()
        
        if df.empty:
            return jsonify({"error": "No data available"}), 404
        
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
        
        return jsonify(record)
    
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500

@app.route("/api/historical/<location>")
def api_historical_data(location):
    """API endpoint for historical data"""
    if location not in LOCATIONS:
        return jsonify({"error": "Location not found"}), 404
    
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    limit = int(request.args.get('limit', 1000))
    
    db_name = f"solar_monitoring_{location}.db"
    conn = get_db_connection(db_name)
    if not conn:
        return jsonify([])
    
    try:
        query = "SELECT * FROM monitoring WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND Timestamp >= ?"
            params.append(start_date.replace('T', ' '))
        
        if end_date:
            query += " AND Timestamp <= ?" 
            params.append(end_date.replace('T', ' '))
        
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
        
        return jsonify(records)
    
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500

@app.route("/api/alerts/<location>")
def api_alerts(location):
    """API endpoint for alerts"""
    if location not in LOCATIONS:
        return jsonify({"error": "Location not found"}), 404
    
    hours = int(request.args.get('hours', 24))
    
    db_name = f"solar_alerts_{location}.db"
    conn = get_db_connection(db_name)
    if not conn:
        return jsonify([])
    
    try:
        # Get alerts from the specified hours
        df = pd.read_sql_query(
            f"SELECT * FROM alerts WHERE timestamp >= datetime('now', '-{hours} hours') ORDER BY timestamp DESC",
            conn
        )
        conn.close()
        
        alerts = df.to_dict('records')
        return jsonify(alerts)
    
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500

@app.route("/api/locations/current")
def api_all_locations_current():
    """API endpoint for current data from all locations"""
    data = {}
    for location in LOCATIONS.keys():
        try:
            db_name = f"solar_monitoring_{location}.db"
            conn = get_db_connection(db_name)
            if conn:
                df = pd.read_sql_query(
                    "SELECT * FROM monitoring ORDER BY Timestamp DESC LIMIT 1", 
                    conn
                )
                conn.close()
                
                if not df.empty:
                    record = df.iloc[0].to_dict()
                    data[location] = record
        except:
            continue
    
    return jsonify(data)

@app.route("/health")
def health_check():
    """Health check endpoint"""
    try:
        db_status = {}
        for location in LOCATIONS.keys():
            monitoring_db = f"solar_monitoring_{location}.db"
            alerts_db = f"solar_alerts_{location}.db"
            predictions_db = f"solar_predictions_{location}.db"
            
            monitoring_exists = os.path.exists(monitoring_db)
            alerts_exists = os.path.exists(alerts_db)
            predictions_exists = os.path.exists(predictions_db)
            
            db_status[location] = {
                "monitoring": "exists" if monitoring_exists else "missing",
                "alerts": "exists" if alerts_exists else "missing",
                "predictions": "exists" if predictions_exists else "missing"
            }
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": MODEL_LOADED,
            "api_keys_loaded": OPENWEATHER_API_KEY is not None and NREL_API_KEY is not None,
            "databases": db_status
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/api/locations")
def api_locations():
    """API endpoint for location information"""
    return jsonify({
        "locations": LOCATIONS,
        "count": len(LOCATIONS)
    })

if __name__ == '__main__':
    # Initialize all databases
    print("🗃️  Initializing databases...")
    for location in LOCATIONS.keys():
        monitoring_db = f"solar_monitoring_{location}.db"
        alerts_db = f"solar_alerts_{location}.db"
        predictions_db = f"solar_predictions_{location}.db"
        
        if initialize_database(monitoring_db):
            print(f"✅ Monitoring database: {monitoring_db}")
        else:
            print(f"❌ Failed to initialize: {monitoring_db}")
            
        if initialize_alerts_database(alerts_db):
            print(f"✅ Alerts database: {alerts_db}")
        else:
            print(f"❌ Failed to initialize: {alerts_db}")
            
        if initialize_predictions_database(predictions_db):
            print(f"✅ Predictions database: {predictions_db}")
        else:
            print(f"❌ Failed to initialize: {predictions_db}")
    
    print(f"🤖 Model loaded: {MODEL_LOADED}")
    print("🚀 Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=5000)