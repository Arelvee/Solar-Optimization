# ==================================================
# OPTIMIZED FEEDFORWARD NEURAL NETWORK (FNN)
# FOR SOLAR ENERGY HARVESTING AND LOAD MANAGEMENT (BMS)
# ==================================================
import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==================================================
# STEP 1: LOAD AND PREPARE DATA
# ==================================================
conn = sqlite3.connect("solar_data_collection.db")
df = pd.read_sql_query("SELECT * FROM solar_data", conn)
conn.close()

# Derived features
df["Solar_Power"] = df["Solar Voltage (V)"] * df["Solar Current (A)"]
df["Efficiency (%)"] = (df["Power Output (W)"] / (df["Solar_Power"] + 1e-6)) * 100
df.dropna(inplace=True)

# Define classification label: Efficient (>=80%) vs Inefficient (<80%)
df["Efficiency_Class"] = np.where(df["Efficiency (%)"] >= 80, 1, 0)

# Features
X = df[[
    "Temperature (°C)", "Humidity (%)", "Solar Voltage (V)",
    "Solar Current (A)", "Solar Irradiance (Lux)",
    "Battery Voltage (V)", "Battery Current (A)",
    "Power Output (W)", "Time of Day (hour,0–23)", "Day Type (0=Cloudy,1=Sunny)"
]].astype(float).values

# Targets
y_eff = df["Efficiency_Class"].values
y_pow = df["Power Output (W)"].values

# ==================================================
# STEP 2: SPLIT DATA AND NORMALIZE
# ==================================================
X_train, X_test, y_eff_train, y_eff_test, y_pow_train, y_pow_test = train_test_split(
    X, y_eff, y_pow, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "scaler_fnn.pkl")
print("✅ Scaler saved as scaler_fnn.pkl")

# ==================================================
# STEP 3: BUILD OPTIMIZED MULTI-OUTPUT FNN MODEL
# ==================================================
inputs = tf.keras.Input(shape=(X_train.shape[1],))

# 🔹 Optimized architecture with BatchNorm + Dropout + residual skip
x = layers.Dense(128, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(64, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(32, activation='relu')(x)

# Heads
eff_out = layers.Dense(1, activation='sigmoid', name='efficiency')(x)
pow_out = layers.Dense(1, activation='linear', name='power')(x)

model = models.Model(inputs=inputs, outputs=[eff_out, pow_out])

# Compile with adaptive optimization
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={'efficiency': 'binary_crossentropy', 'power': 'mse'},
    metrics={'efficiency': ['accuracy', tf.keras.metrics.AUC(name='auc')],
             'power': ['mae', 'mse']}
)

# ==================================================
# STEP 4: TRAIN WITH OPTIMIZATION STRATEGIES
# ==================================================
# Callbacks for optimization
cb = [
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    callbacks.ModelCheckpoint("best_fnn_model.keras", save_best_only=True)
]

history = model.fit(
    X_train,
    {'efficiency': y_eff_train, 'power': y_pow_train},
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=cb,
    verbose=1
)

# ==================================================
# STEP 5: EVALUATION
# ==================================================
y_eff_pred, y_pow_pred = model.predict(X_test)
y_eff_pred_class = (y_eff_pred.flatten() > 0.5).astype(int)

# Classification metrics
acc = accuracy_score(y_eff_test, y_eff_pred_class)
prec = precision_score(y_eff_test, y_eff_pred_class)
rec = recall_score(y_eff_test, y_eff_pred_class)
f1 = f1_score(y_eff_test, y_eff_pred_class)
auc = roc_auc_score(y_eff_test, y_eff_pred)

# Regression metrics
rmse = np.sqrt(mean_squared_error(y_pow_test, y_pow_pred))
mae = mean_absolute_error(y_pow_test, y_pow_pred)
r2 = r2_score(y_pow_test, y_pow_pred)
mape = np.mean(np.abs((y_pow_test - y_pow_pred.flatten()) / (y_pow_test + 1e-6))) * 100

# ==================================================
# STEP 6: CONFUSION MATRIX & REPORT
# ==================================================
cm = confusion_matrix(y_eff_test, y_eff_pred_class)
report = classification_report(y_eff_test, y_eff_pred_class, target_names=["Low Efficiency", "High Efficiency"])

# ==================================================
# STEP 7: BATTERY MANAGEMENT SYSTEM (BMS) LOGIC
# ==================================================
# Simulated threshold-based alert (load control)
battery_voltage = df["Battery Voltage (V)"].iloc[-1]
low_batt_threshold = 11.8  # example threshold
if battery_voltage < low_batt_threshold:
    bms_status = "⚠️ LOW BATTERY - Load (lights) should be turned OFF to save energy."
else:
    bms_status = "✅ Battery level normal - Load (lights) can remain ON."

# ==================================================
# STEP 8: PERFORMANCE SUMMARY
# ==================================================
print("\n📊 EFFICIENCY CLASSIFICATION METRICS")
print("----------------------------------")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC-ROC  : {auc:.4f}\n")

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

print("\n🔋 POWER PREDICTION METRICS")
print("----------------------------------")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape:.2f}%\n")

print("⚡ BATTERY MANAGEMENT STATUS:")
print("----------------------------------")
print(bms_status)

# ==================================================
# STEP 9: VISUALIZATION
# ==================================================
plt.figure(figsize=(10, 5))
plt.plot(history.history['efficiency_accuracy'], label='Train Acc')
plt.plot(history.history['val_efficiency_accuracy'], label='Val Acc', linestyle='--')
plt.title("Efficiency Classification Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['power_mae'], label='Train MAE')
plt.plot(history.history['val_power_mae'], label='Val MAE', linestyle='--')
plt.title("Power Prediction Mean Absolute Error")
plt.xlabel("Epochs")
plt.ylabel("MAE (W)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Low", "High"], yticklabels=["Low", "High"])
plt.title("Confusion Matrix - Efficiency Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(6, 5))
plt.scatter(y_pow_test, y_pow_pred, alpha=0.7)
plt.title("Power Prediction (Actual vs Predicted)")
plt.xlabel("Actual Power (W)")
plt.ylabel("Predicted Power (W)")
plt.grid(True)
plt.show()

# ==================================================
# STEP 10: EXPORT TO TFLITE (for ESP32-S3 deployment)
# ==================================================
def representative_dataset():
    for i in range(500):
        idx = np.random.randint(0, X_train.shape[0])
        yield [X_train[idx:idx+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.experimental_new_quantizer = True
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open("optimized_fnn_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Exported: optimized_fnn_int8.tflite for ESP32-S3 deployment")
