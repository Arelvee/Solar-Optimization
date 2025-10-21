from fastapi import FastAPI, Query, HTTPException, status
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import tensorflow as tf
import sqlite3
from typing import Optional
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# LOAD TFLITE MODEL AND SCALER
# ---------------------------
try:
    interpreter = tf.lite.Interpreter(model_path="fnn_model_float32.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    scaler = joblib.load("scaler_fnn.pkl")
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model/scaler: {e}")
    raise

# ---------------------------
# FASTAPI APP
# ---------------------------
app = FastAPI(
    title="Solar FNN Prediction & Sensor Data API",
    description="API for solar efficiency predictions and sensor data retrieval",
    version="1.0.0"
)

# ---------------------------
# SENSOR INPUT SCHEMA
# ---------------------------
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

    @validator('battery_current')
    def validate_battery_current(cls, v):
        # Battery current can be positive (charging) or negative (discharging)
        if abs(v) > 100:  # Assuming reasonable limits
            raise ValueError('Battery current out of reasonable range')
        return v

class PredictionResponse(BaseModel):
    efficiency: float
    power: float

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

# ---------------------------
# FNN PREDICTION ENDPOINT
# ---------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
def predict(data: SensorInput):
    try:
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

        logger.info(f"Prediction made - Efficiency: {efficiency_pred:.4f}, Power: {power_pred:.4f}")

        return PredictionResponse(efficiency=efficiency_pred, power=power_pred)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

# ---------------------------
# HISTORICAL SENSOR DATA ENDPOINT
# ---------------------------
@app.get("/sensor-data")
def get_sensor_data(
    start: Optional[str] = Query(None, description="Start datetime (YYYY-MM-DD HH:MM)"),
    end: Optional[str] = Query(None, description="End datetime (YYYY-MM-DD HH:MM)"),
    limit: Optional[int] = Query(1000, ge=1, le=10000, description="Limit number of records")
):
    try:
        # Validate date formats
        if start:
            try:
                datetime.strptime(start, "%Y-%m-%d %H:%M")
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid start date format. Use YYYY-MM-DD HH:MM"
                )
        
        if end:
            try:
                datetime.strptime(end, "%Y-%m-%d %H:%M")
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid end date format. Use YYYY-MM-DD HH:MM"
                )

        conn = sqlite3.connect("solar_data_collection.db")
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()
        
        query = "SELECT * FROM solar_data WHERE 1=1"
        params = []

        if start:
            query += " AND Timestamp >= ?"
            params.append(start)
        if end:
            query += " AND Timestamp <= ?"
            params.append(end)
            
        query += " ORDER BY Timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        # Convert rows to dictionaries
        data = [dict(row) for row in rows]
        
        logger.info(f"Retrieved {len(data)} sensor records")
        
        return {
            "count": len(data),
            "data": data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sensor data: {str(e)}"
        )

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint to verify API status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": True,
        "scaler_loaded": True
    }

# Root endpoint
@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Solar FNN Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict - Get solar efficiency and power predictions",
            "sensor_data": "GET /sensor-data - Retrieve historical sensor data",
            "health": "GET /health - API health check"
        }
    }