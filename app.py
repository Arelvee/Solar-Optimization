from flask import Flask, render_template, request, jsonify, flash
import requests
from datetime import datetime, timedelta
import logging
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_URL = "http://127.0.0.1:8000"

# ---------------------------
# Helper Functions
# ---------------------------
def get_fnn_prediction(sensor_data):
    """Get prediction from FNN API with error handling"""
    try:
        response = requests.post(f"{API_URL}/predict", json=sensor_data, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return {"error": f"Prediction service unavailable: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}

def get_sensor_data(start_date, end_date):
    """Fetch sensor data from FastAPI with error handling"""
    try:
        params = {}
        if start_date:
            params["start"] = start_date
        if end_date:
            params["end"] = end_date
            
        response = requests.get(f"{API_URL}/sensor-data", params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("data", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Sensor data request failed: {str(e)}")
        flash("Error fetching sensor data. Please check if the API server is running.", "error")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching sensor data: {str(e)}")
        flash("Unexpected error fetching sensor data.", "error")
        return []

def validate_sensor_input(form_data):
    """Validate sensor input form data"""
    errors = []
    required_fields = [
        'temperature', 'humidity', 'solar_voltage', 'solar_current',
        'solar_irradiance', 'battery_voltage', 'battery_current',
        'power_output', 'hour', 'day_type'
    ]
    
    # Check required fields
    for field in required_fields:
        if not form_data.get(field):
            errors.append(f"{field.replace('_', ' ').title()} is required")
    
    # Validate numeric fields
    numeric_fields = ['temperature', 'humidity', 'solar_voltage', 'solar_current',
                     'solar_irradiance', 'battery_voltage', 'battery_current', 'power_output']
    
    for field in numeric_fields:
        value = form_data.get(field)
        if value:
            try:
                float(value)
            except ValueError:
                errors.append(f"{field.replace('_', ' ').title()} must be a number")
    
    # Validate hour
    hour = form_data.get('hour')
    if hour:
        try:
            hour_int = int(hour)
            if hour_int < 0 or hour_int > 23:
                errors.append("Hour must be between 0 and 23")
        except ValueError:
            errors.append("Hour must be an integer")
    
    # Validate day_type
    day_type = form_data.get('day_type')
    if day_type:
        try:
            day_type_int = int(day_type)
            if day_type_int not in [0, 1]:
                errors.append("Day type must be 0 (weekday) or 1 (weekend)")
        except ValueError:
            errors.append("Day type must be 0 or 1")
    
    return errors

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def dashboard():
    sensor_records = []
    fnn_predictions = {}
    start_date = request.form.get("start_date") or request.args.get("start_date")
    end_date = request.form.get("end_date") or request.args.get("end_date")
    
    # Set default date range (last 7 days)
    if not start_date or not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M")

    # Fetch sensor data
    try:
        sensor_records = get_sensor_data(start_date, end_date)
        if sensor_records:
            flash(f"Successfully loaded {len(sensor_records)} sensor records", "success")
    except Exception as e:
        logger.error(f"Error in dashboard: {str(e)}")
        flash("Error loading sensor data", "error")

    # Handle prediction request
    if request.method == "POST" and request.form.get("predict") == "1":
        # Validate form data
        validation_errors = validate_sensor_input(request.form)
        
        if validation_errors:
            for error in validation_errors:
                flash(error, "error")
        else:
            try:
                sensor_input = {
                    "temperature": float(request.form.get("temperature")),
                    "humidity": float(request.form.get("humidity")),
                    "solar_voltage": float(request.form.get("solar_voltage")),
                    "solar_current": float(request.form.get("solar_current")),
                    "solar_irradiance": float(request.form.get("solar_irradiance")),
                    "battery_voltage": float(request.form.get("battery_voltage")),
                    "battery_current": float(request.form.get("battery_current")),
                    "power_output": float(request.form.get("power_output")),
                    "hour": int(request.form.get("hour")),
                    "day_type": int(request.form.get("day_type"))
                }
                
                fnn_predictions = get_fnn_prediction(sensor_input)
                
                if "error" in fnn_predictions:
                    flash(f"Prediction error: {fnn_predictions['error']}", "error")
                else:
                    flash("Prediction completed successfully!", "success")
                    
            except Exception as e:
                logger.error(f"Prediction processing error: {str(e)}")
                flash(f"Error processing prediction: {str(e)}", "error")

    return render_template(
        "dashboard.html",
        sensor_records=sensor_records,
        fnn_predictions=fnn_predictions,
        start_date=start_date,
        end_date=end_date
    )

@app.route("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Check if API is responsive
        api_response = requests.get(f"{API_URL}/health", timeout=5)
        api_status = "healthy" if api_response.status_code == 200 else "unhealthy"
        
        return jsonify({
            "status": "healthy",
            "api_connection": api_status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "api_connection": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for predictions (for external clients)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = [
            'temperature', 'humidity', 'solar_voltage', 'solar_current',
            'solar_irradiance', 'battery_voltage', 'battery_current',
            'power_output', 'hour', 'day_type'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Get prediction
        prediction = get_fnn_prediction(data)
        
        if "error" in prediction:
            return jsonify({"error": prediction["error"]}), 500
        
        return jsonify(prediction)
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal server error"), 500

if __name__ == "__main__":
    # Check if API is available
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            logger.info("FastAPI server is running and accessible")
        else:
            logger.warning("FastAPI server returned non-200 status")
    except Exception as e:
        logger.warning(f"FastAPI server may not be running: {str(e)}")
    
    app.run(debug=True, host="0.0.0.0", port=5000)