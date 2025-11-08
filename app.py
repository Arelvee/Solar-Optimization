from flask import Flask, render_template, request, jsonify, flash, send_from_directory
import requests
from datetime import datetime, timedelta
import logging
import os
import sqlite3
import pandas as pd
import json
import numpy as np
from typing import Dict, List, Optional, Union, Any

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "solar-monitoring-secret-key")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------
# CONFIGURATION
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
# ENHANCED DATABASE FUNCTIONS
# ------------------------
def initialize_database(db_file):
    """Initialize database with proper table structure"""
    try:
        conn = sqlite3.connect(db_file)
        
        # Check if table exists and get current columns
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='monitoring'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            # Get existing columns
            cursor.execute("PRAGMA table_info(monitoring)")
            existing_columns = [column[1] for column in cursor.fetchall()]
            
            # Add missing columns if needed
            required_columns = {
                "Battery_Power": "REAL",
                "Power_Direction": "TEXT",
                "Rainfall": "REAL"
            }
            
            for column, col_type in required_columns.items():
                if column not in existing_columns:
                    try:
                        conn.execute(f"ALTER TABLE monitoring ADD COLUMN {column} {col_type}")
                        logger.info(f"Added column {column} to {db_file}")
                    except Exception as e:
                        logger.warning(f"Could not add column {column}: {e}")
        else:
            # Create new table with complete schema
            conn.execute('''
                CREATE TABLE monitoring (
                    Timestamp TEXT,
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

def get_db_connection(db_file):
    """Create a database connection to the SQLite database"""
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
    
    initialize_database(monitoring_db)
    initialize_alerts_database(alerts_db)

# ------------------------
# EFFICIENCY CLASS HELPER FUNCTIONS
# ------------------------
def calculate_efficiency_class(efficiency):
    """Calculate efficiency class: 0=Low, 1=Medium, 2=High"""
    if efficiency is None:
        return 0
    
    if efficiency >= 0.15:  # 15% or higher
        return 2  # High
    elif efficiency >= 0.10:  # 10% to 14.9%
        return 1  # Medium
    else:  # Below 10%
        return 0  # Low

def get_efficiency_class_label(efficiency_class):
    """Get human-readable label for efficiency class"""
    labels = {
        0: "Low",
        1: "Medium", 
        2: "High"
    }
    return labels.get(efficiency_class, "Unknown")

# ------------------------
# REAL-TIME DATA FUNCTIONS
# ------------------------
def get_latest_monitoring_data(location):
    """Get the latest monitoring data for a location"""
    try:
        db_name = f"solar_monitoring_{location}.db"
        conn = get_db_connection(db_name)
        if not conn:
            return None
        
        # Get the latest complete record
        df = pd.read_sql_query('''
            SELECT * FROM monitoring 
            WHERE Temperature IS NOT NULL 
            AND Solar_Irradiance IS NOT NULL
            ORDER BY Timestamp DESC 
            LIMIT 1
        ''', conn)
        conn.close()
        
        if df.empty:
            return None
        
        record = df.iloc[0].to_dict()
        
        # Convert numeric fields to 2 decimal places
        numeric_fields = ['Temperature', 'Humidity', 'Rainfall', 'Solar_Irradiance', 
                         'Solar_DNI', 'Solar_DHI', 'Panel_Voltage', 'Panel_Current',
                         'Raw_Power', 'Actual_Power', 'Panel_Efficiency', 'Battery_Voltage',
                         'State_of_Charge', 'Charge_Current', 'Net_Power', 'Load_Power',
                         'Battery_Power', 'Max_Load']
        
        for field in numeric_fields:
            if field in record and record[field] is not None:
                try:
                    record[field] = round(float(record[field]), 2)
                except (ValueError, TypeError):
                    record[field] = 0.0
        
        # Ensure Efficiency_Class is integer
        if 'Efficiency_Class' in record and record['Efficiency_Class'] is not None:
            try:
                record['Efficiency_Class'] = int(record['Efficiency_Class'])
            except (ValueError, TypeError):
                # Calculate efficiency class if not present or invalid
                efficiency = record.get('Panel_Efficiency', 0)
                record['Efficiency_Class'] = calculate_efficiency_class(efficiency)
        else:
            # Calculate efficiency class if not present
            efficiency = record.get('Panel_Efficiency', 0)
            record['Efficiency_Class'] = calculate_efficiency_class(efficiency)
        
        # Add efficiency class label
        record['Efficiency_Class_Label'] = get_efficiency_class_label(record['Efficiency_Class'])
        
        # Calculate additional real-time metrics
        record = calculate_realtime_metrics(record)
        
        return record
        
    except Exception as e:
        logger.error(f"Error getting latest monitoring data for {location}: {e}")
        return None

def get_recent_alerts(location, limit=50, severity=None, start_date=None, end_date=None):
    """Get recent alerts for a location"""
    try:
        db_name = f"solar_alerts_{location}.db"
        
        # Check if database file exists
        if not os.path.exists(db_name):
            logger.warning(f"Alerts database {db_name} does not exist")
            return []
        
        conn = get_db_connection(db_name)
        if not conn:
            logger.warning(f"Could not connect to alerts database {db_name}")
            return []
        
        # Check if alerts table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'")
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            logger.warning(f"Alerts table does not exist in {db_name}")
            conn.close()
            return []
        
        query = "SELECT * FROM alerts WHERE 1=1"
        params = []
        
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        
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
        
        # Convert DataFrame to list of dicts and ensure proper data types
        alerts = df.to_dict('records')
        
        # Ensure all alerts have required fields
        for alert in alerts:
            alert['timestamp'] = alert.get('timestamp', '')
            alert['type'] = alert.get('type', 'unknown')
            alert['severity'] = alert.get('severity', 'info')
            alert['message'] = alert.get('message', 'No message')
            alert['location'] = alert.get('location', location)
        
        logger.info(f"Retrieved {len(alerts)} alerts for {location}")
        return alerts
        
    except Exception as e:
        logger.error(f"Error getting alerts for {location}: {e}")
        return []
    
def calculate_realtime_metrics(data):
    """Calculate additional real-time metrics for display"""
    if not data:
        return data
    
    # Calculate power status
    actual_power = data.get('Actual_Power', 0)
    max_power = REAL_SOLAR_SPECS['max_power']
    
    if max_power > 0:
        data['power_utilization'] = round((actual_power / max_power) * 100, 2)
    else:
        data['power_utilization'] = 0.0
    
    # Calculate efficiency percentage
    efficiency = data.get('Panel_Efficiency', 0)
    max_efficiency = REAL_SOLAR_SPECS['efficiency']
    
    if max_efficiency > 0:
        data['efficiency_percentage'] = round((efficiency / max_efficiency) * 100, 2)
    else:
        data['efficiency_percentage'] = 0.0
    
    # Determine system status
    data['system_status'] = determine_system_status(data)
    
    # Calculate battery health
    soc = data.get('State_of_Charge', 0)
    if soc >= 80:
        data['battery_status'] = 'Excellent'
    elif soc >= 50:
        data['battery_status'] = 'Good'
    elif soc >= 20:
        data['battery_status'] = 'Low'
    else:
        data['battery_status'] = 'Critical'
    
    # Calculate power direction status
    net_power = data.get('Net_Power', 0)
    if net_power > 0:
        data['power_direction_status'] = 'Charging'
    elif net_power < 0:
        data['power_direction_status'] = 'Discharging'
    else:
        data['power_direction_status'] = 'Balanced'
    
    return data

def determine_system_status(data):
    """Determine overall system status based on multiple factors"""
    alerts = []
    
    # Check battery SOC
    soc = data.get('State_of_Charge', 0)
    if soc < 20:
        alerts.append('Battery Critical')
    elif soc < 50:
        alerts.append('Battery Low')
    
    # Check efficiency class
    efficiency_class = data.get('Efficiency_Class', 0)
    if efficiency_class == 0:  # Low efficiency
        alerts.append('Low Efficiency')
    
    # Check power output
    actual_power = data.get('Actual_Power', 0)
    if actual_power < 10:
        alerts.append('Low Power')
    
    # Check for high temperature
    temperature = data.get('Temperature', 0)
    if temperature > 45:
        alerts.append('High Temperature')
    
    if alerts:
        return f"⚠️ {' | '.join(alerts)}"
    else:
        return "✅ Normal"

def get_system_health_summary(location):
    """Get overall system health summary for a location"""
    monitoring_data = get_latest_monitoring_data(location)
    recent_alerts = get_recent_alerts(location, limit=10, severity='high')
    
    if not monitoring_data:
        return {
            "status": "unknown",
            "message": "No data available",
            "score": 0
        }
    
    # Calculate health score (0-100)
    score = 100
    
    # Deduct points based on issues
    soc = monitoring_data.get('State_of_Charge', 0)
    if soc < 20:
        score -= 40
    elif soc < 50:
        score -= 20
    
    efficiency_class = monitoring_data.get('Efficiency_Class', 0)
    if efficiency_class == 0:  # Low efficiency
        score -= 30
    elif efficiency_class == 1:  # Medium efficiency
        score -= 10
    
    actual_power = monitoring_data.get('Actual_Power', 0)
    if actual_power < 10:
        score -= 20
    
    # Deduct for recent high severity alerts
    score -= len(recent_alerts) * 5
    
    # Ensure score is within bounds
    score = max(0, min(100, score))
    
    # Determine status
    if score >= 80:
        status = "healthy"
    elif score >= 60:
        status = "warning"
    else:
        status = "critical"
    
    return {
        "status": status,
        "score": round(score, 2),
        "message": monitoring_data.get('system_status', 'Unknown'),
        "critical_alerts": len(recent_alerts)
    }

# ------------------------
# FLASK ROUTES
# ------------------------
@app.route("/")
def dashboard():
    """Main dashboard page"""
    return render_template("dashboard.html", 
                         locations=LOCATIONS,
                         solar_specs=REAL_SOLAR_SPECS)

# Real-Time Data Routes
@app.route("/api/realtime/<location>")
def api_realtime_data(location):
    """API endpoint for real-time data with latest updates"""
    if location not in LOCATIONS:
        return jsonify({"error": "Location not found"}), 404
    
    # Get the latest monitoring data
    monitoring_data = get_latest_monitoring_data(location)
    alerts_data = get_recent_alerts(location, limit=10)
    
    return jsonify({
        "location": location,
        "location_info": LOCATIONS[location],
        "monitoring": monitoring_data,
        "alerts": alerts_data,
        "timestamp": datetime.now().isoformat(),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/api/realtime/all")
def api_realtime_all():
    """API endpoint for all locations real-time data"""
    all_data = {}
    
    for location in LOCATIONS.keys():
        monitoring_data = get_latest_monitoring_data(location)
        alerts_data = get_recent_alerts(location, limit=5)
        
        all_data[location] = {
            "location_info": LOCATIONS[location],
            "monitoring": monitoring_data,
            "alerts": alerts_data,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    return jsonify(all_data)

@app.route("/api/alerts/<location>")
def api_alerts_data(location):
    """API endpoint for alerts data"""
    if location not in LOCATIONS:
        return jsonify({"error": "Location not found"}), 404
    
    limit = int(request.args.get('limit', 50))
    severity = request.args.get('severity')
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    alerts = get_recent_alerts(location, limit, severity, start_date, end_date)
    
    return jsonify({
        "location": location,
        "alerts": alerts,
        "total_count": len(alerts),
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/current/<location>")
def api_current_data(location):
    """API endpoint for current data"""
    if location not in LOCATIONS:
        return jsonify({"error": "Location not found"}), 404
    
    db_name = f"solar_monitoring_{location}.db"
    conn = get_db_connection(db_name)
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        # Get the latest record
        df = pd.read_sql_query(
            "SELECT * FROM monitoring ORDER BY Timestamp DESC LIMIT 1", 
            conn
        )
        conn.close()
        
        if df.empty:
            return jsonify({"error": "No data available for this location"}), 404
        
        record = df.iloc[0].to_dict()
        
        # Convert numeric fields to 2 decimal places
        numeric_fields = ['Temperature', 'Humidity', 'Solar_Irradiance', 'Solar_DNI', 'Solar_DHI',
                         'Panel_Voltage', 'Panel_Current', 'Raw_Power', 'Actual_Power', 'Panel_Efficiency',
                         'Battery_Voltage', 'State_of_Charge', 'Charge_Current', 'Net_Power', 'Load_Power',
                         'Efficiency_Class', 'Battery_Power', 'Rainfall', 'Max_Load']
        
        for field in numeric_fields:
            if field in record:
                try:
                    if record[field] is not None:
                        record[field] = round(float(record[field]), 2)
                    else:
                        record[field] = 0.0
                except (ValueError, TypeError):
                    record[field] = 0.0
        
        # Ensure Efficiency_Class is integer
        if 'Efficiency_Class' in record and record['Efficiency_Class'] is not None:
            try:
                record['Efficiency_Class'] = int(record['Efficiency_Class'])
            except (ValueError, TypeError):
                efficiency = record.get('Panel_Efficiency', 0)
                record['Efficiency_Class'] = calculate_efficiency_class(efficiency)
        else:
            efficiency = record.get('Panel_Efficiency', 0)
            record['Efficiency_Class'] = calculate_efficiency_class(efficiency)
        
        # Add efficiency class label
        record['Efficiency_Class_Label'] = get_efficiency_class_label(record['Efficiency_Class'])
        
        # Add location info
        record['location_info'] = LOCATIONS[location]
        
        return jsonify(record)
    
    except Exception as e:
        conn.close()
        logger.error(f"Error fetching current data for {location}: {e}")
        return jsonify({"error": f"Failed to fetch data: {str(e)}"}), 500

@app.route("/api/current/all")
def api_current_all_locations():
    """API endpoint for current data across all locations"""
    all_current_data = {}
    
    for location in LOCATIONS.keys():
        monitoring_data = get_latest_monitoring_data(location)
        if monitoring_data:
            all_current_data[location] = {
                **monitoring_data,
                "location_info": LOCATIONS[location]
            }
    
    return jsonify(all_current_data)

@app.route("/api/historical/<location>")
def api_historical_data(location):
    """API endpoint for historical data"""
    if location not in LOCATIONS:
        return jsonify({"error": "Location not found"}), 404
    
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    limit = min(int(request.args.get('limit', 1000)), 10000)  # Cap at 10k records
    
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
        
        # Convert to proper data types with 2 decimal places
        numeric_fields = ['Temperature', 'Humidity', 'Solar_Irradiance', 'Solar_DNI', 'Solar_DHI',
                         'Panel_Voltage', 'Panel_Current', 'Raw_Power', 'Actual_Power', 'Panel_Efficiency',
                         'Battery_Voltage', 'State_of_Charge', 'Charge_Current', 'Net_Power', 'Load_Power',
                         'Efficiency_Class', 'Battery_Power', 'Rainfall', 'Max_Load']
        
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0).round(2)
        
        # Ensure Efficiency_Class is integer
        if 'Efficiency_Class' in df.columns:
            df['Efficiency_Class'] = df['Efficiency_Class'].fillna(0).astype(int)
            df['Efficiency_Class_Label'] = df['Efficiency_Class'].apply(get_efficiency_class_label)
        
        records = df.to_dict('records')
        return jsonify(records)
    
    except Exception as e:
        if conn:
            conn.close()
        return jsonify({"error": str(e)}), 500

# Health and Statistics Routes
@app.route("/api/health/summary")
def api_health_summary():
    """API endpoint for system health summary across all locations"""
    health_summary = {}
    
    for location in LOCATIONS.keys():
        health_summary[location] = get_system_health_summary(location)
    
    return jsonify(health_summary)

@app.route("/api/health/<location>")
def api_location_health(location):
    """API endpoint for specific location health"""
    if location not in LOCATIONS:
        return jsonify({"error": "Location not found"}), 404
    
    health_summary = get_system_health_summary(location)
    return jsonify(health_summary)

@app.route("/api/stats/<location>")
def api_location_stats(location):
    """API endpoint for location statistics"""
    if location not in LOCATIONS:
        return jsonify({"error": "Location not found"}), 404
    
    try:
        db_name = f"solar_monitoring_{location}.db"
        conn = get_db_connection(db_name)
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Get basic statistics
        stats_query = '''
            SELECT 
                COUNT(*) as total_records,
                AVG(Actual_Power) as avg_power,
                MAX(Actual_Power) as max_power,
                AVG(Panel_Efficiency) as avg_efficiency,
                AVG(State_of_Charge) as avg_soc,
                AVG(Temperature) as avg_temperature,
                AVG(Solar_Irradiance) as avg_irradiance
            FROM monitoring
            WHERE Timestamp >= datetime('now', '-1 day')
        '''
        
        df_stats = pd.read_sql_query(stats_query, conn)
        
        # Get today's energy production
        energy_query = '''
            SELECT SUM(Actual_Power * 0.25) as today_energy
            FROM monitoring
            WHERE date(Timestamp) = date('now')
            AND Actual_Power > 0
        '''
        
        df_energy = pd.read_sql_query(energy_query, conn)
        conn.close()
        
        stats = {}
        if not df_stats.empty:
            stats = {
                "total_records": int(df_stats.iloc[0]['total_records']),
                "average_power": round(float(df_stats.iloc[0]['avg_power'] or 0), 2),
                "max_power": round(float(df_stats.iloc[0]['max_power'] or 0), 2),
                "average_efficiency": round(float(df_stats.iloc[0]['avg_efficiency'] or 0), 3),
                "average_soc": round(float(df_stats.iloc[0]['avg_soc'] or 0), 2),
                "average_temperature": round(float(df_stats.iloc[0]['avg_temperature'] or 0), 2),
                "average_irradiance": round(float(df_stats.iloc[0]['avg_irradiance'] or 0), 2),
                "today_energy_wh": round(float(df_energy.iloc[0]['today_energy'] or 0), 2)
            }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting stats for {location}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/dashboard/summary")
def api_dashboard_summary():
    """API endpoint for dashboard summary data"""
    dashboard_data = {
        "locations": {},
        "overall_health": "unknown",
        "total_power": 0.0,
        "total_alerts": 0,
        "timestamp": datetime.now().isoformat()
    }
    
    total_health_score = 0.0
    location_count = 0
    
    for location in LOCATIONS.keys():
        # Get current data
        current_data = get_latest_monitoring_data(location)
        health_data = get_system_health_summary(location)
        recent_alerts = get_recent_alerts(location, limit=5)
        
        if current_data:
            dashboard_data["locations"][location] = {
                "current_data": current_data,
                "health": health_data,
                "recent_alerts": recent_alerts,
                "location_info": LOCATIONS[location]
            }
            
            total_health_score += health_data["score"]
            location_count += 1
            dashboard_data["total_power"] += current_data.get('Actual_Power', 0)
            dashboard_data["total_alerts"] += len(recent_alerts)
    
    # Calculate overall health
    if location_count > 0:
        avg_health_score = total_health_score / location_count
        if avg_health_score >= 80:
            dashboard_data["overall_health"] = "healthy"
        elif avg_health_score >= 60:
            dashboard_data["overall_health"] = "warning"
        else:
            dashboard_data["overall_health"] = "critical"
    
    # Round total power to 2 decimal places
    dashboard_data["total_power"] = round(dashboard_data["total_power"], 2)
    
    return jsonify(dashboard_data)

@app.route("/health")
def health_check():
    """Health check endpoint with detailed information"""
    try:
        db_status = {}
        record_counts = {}
        
        for location in LOCATIONS.keys():
            monitoring_db = f"solar_monitoring_{location}.db"
            alerts_db = f"solar_alerts_{location}.db"
            
            # Check database existence and record counts
            for db_type, db_file in [('monitoring', monitoring_db), ('alerts', alerts_db)]:
                if os.path.exists(db_file):
                    try:
                        conn = sqlite3.connect(db_file)
                        cursor = conn.cursor()
                        table_name = 'monitoring' if db_type == 'monitoring' else 'alerts'
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]
                        conn.close()
                        
                        if location not in record_counts:
                            record_counts[location] = {}
                        record_counts[location][db_type] = count
                        db_status[f"{location}_{db_type}"] = "healthy"
                    except Exception as e:
                        db_status[f"{location}_{db_type}"] = f"error: {str(e)}"
                else:
                    db_status[f"{location}_{db_type}"] = "missing"
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "api_keys_loaded": OPENWEATHER_API_KEY is not None and NREL_API_KEY is not None,
            "locations_configured": len(LOCATIONS),
            "databases": db_status,
            "record_counts": record_counts,
            "system_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500
@app.route("/api/load-status/<location>")
def api_load_status(location):
    """API endpoint for load status, load action, and recommended action"""
    if location not in LOCATIONS:
        return jsonify({"error": "Location not found"}), 404
    
    try:
        db_name = f"solar_monitoring_{location}.db"
        conn = get_db_connection(db_name)
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Get the latest record with load status information
        df = pd.read_sql_query('''
            SELECT Timestamp, Load_Status, Load_Action, Recommended_Action 
            FROM monitoring 
            WHERE Load_Status IS NOT NULL 
            OR Load_Action IS NOT NULL 
            OR Recommended_Action IS NOT NULL
            ORDER BY Timestamp DESC 
            LIMIT 1
        ''', conn)
        conn.close()
        
        if df.empty:
            return jsonify({
                "location": location,
                "timestamp": datetime.now().isoformat(),
                "load_status": "No data available",
                "load_action": "No data available", 
                "recommended_action": "No data available",
                "message": "No load status data found"
            })
        
        record = df.iloc[0].to_dict()
        
        # Ensure values are not None
        load_status = record.get('Load_Status', 'Unknown')
        load_action = record.get('Load_Action', 'Unknown')
        recommended_action = record.get('Recommended_Action', 'Unknown')
        
        return jsonify({
            "location": location,
            "location_info": LOCATIONS[location],
            "timestamp": record.get('Timestamp', datetime.now().isoformat()),
            "load_status": load_status if load_status is not None else "Unknown",
            "load_action": load_action if load_action is not None else "Unknown",
            "recommended_action": recommended_action if recommended_action is not None else "Unknown",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        logger.error(f"Error getting load status for {location}: {e}")
        return jsonify({"error": f"Failed to fetch load status: {str(e)}"}), 500

@app.route("/api/load-status/all")
def api_all_load_status():
    """API endpoint for load status across all locations"""
    all_load_status = {}
    
    for location in LOCATIONS.keys():
        try:
            db_name = f"solar_monitoring_{location}.db"
            conn = get_db_connection(db_name)
            if not conn:
                all_load_status[location] = {"error": "Database connection failed"}
                continue
            
            # Get the latest record with load status information
            df = pd.read_sql_query('''
                SELECT Timestamp, Load_Status, Load_Action, Recommended_Action 
                FROM monitoring 
                WHERE Load_Status IS NOT NULL 
                OR Load_Action IS NOT NULL 
                OR Recommended_Action IS NOT NULL
                ORDER BY Timestamp DESC 
                LIMIT 1
            ''', conn)
            conn.close()
            
            if df.empty:
                all_load_status[location] = {
                    "load_status": "No data available",
                    "load_action": "No data available",
                    "recommended_action": "No data available",
                    "timestamp": datetime.now().isoformat(),
                    "message": "No load status data found"
                }
            else:
                record = df.iloc[0].to_dict()
                all_load_status[location] = {
                    "load_status": record.get('Load_Status', 'Unknown'),
                    "load_action": record.get('Load_Action', 'Unknown'),
                    "recommended_action": record.get('Recommended_Action', 'Unknown'),
                    "timestamp": record.get('Timestamp', datetime.now().isoformat()),
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
        except Exception as e:
            logger.error(f"Error getting load status for {location}: {e}")
            all_load_status[location] = {"error": f"Failed to fetch load status: {str(e)}"}
    
    return jsonify(all_load_status)

@app.route("/api/load-history/<location>")
def api_load_history(location):
    """API endpoint for historical load status data"""
    if location not in LOCATIONS:
        return jsonify({"error": "Location not found"}), 404
    
    limit = min(int(request.args.get('limit', 100)), 1000)  # Cap at 1000 records
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    try:
        db_name = f"solar_monitoring_{location}.db"
        conn = get_db_connection(db_name)
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        query = '''
            SELECT Timestamp, Load_Status, Load_Action, Recommended_Action 
            FROM monitoring 
            WHERE (Load_Status IS NOT NULL OR Load_Action IS NOT NULL OR Recommended_Action IS NOT NULL)
        '''
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
        
        # Replace None values with "Unknown"
        df['Load_Status'] = df['Load_Status'].fillna('Unknown')
        df['Load_Action'] = df['Load_Action'].fillna('Unknown')
        df['Recommended_Action'] = df['Recommended_Action'].fillna('Unknown')
        
        records = df.to_dict('records')
        
        return jsonify({
            "location": location,
            "location_info": LOCATIONS[location],
            "data": records,
            "total_records": len(records),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting load history for {location}: {e}")
        return jsonify({"error": f"Failed to fetch load history: {str(e)}"}), 500
# ------------------------
# APPLICATION STARTUP
# ------------------------
if __name__ == '__main__':
    # Initialize all databases
    print("🗃️  Initializing databases...")
    for location in LOCATIONS.keys():
        monitoring_db = f"solar_monitoring_{location}.db"
        alerts_db = f"solar_alerts_{location}.db"
        
        if initialize_database(monitoring_db):
            print(f"✅ Monitoring database: {monitoring_db}")
        else:
            print(f"❌ Failed to initialize: {monitoring_db}")
            
        if initialize_alerts_database(alerts_db):
            print(f"✅ Alerts database: {alerts_db}")
        else:
            print(f"❌ Failed to initialize: {alerts_db}")
    
    print("🚀 Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=5000)