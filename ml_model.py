# ==================================================
# OPTIMIZED MULTI-OUTPUT FEEDFORWARD NEURAL NETWORK (FNN)
# FOR SOLAR ENERGY HARVESTING OPTIMIZATION & LOAD MANAGEMENT (BMS)
# ==================================================
import sqlite3
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

# Define 3-class efficiency label
def classify_efficiency(eff):
    if eff < 70:
        return 0  # Low
    elif eff < 90:
        return 1  # Medium
    else:
        return 2  # High

df["Efficiency_Class"] = df["Efficiency (%)"].apply(classify_efficiency)

# Features and targets
feature_cols = [
    "Temperature (°C)", "Humidity (%)", "Solar Voltage (V)",
    "Solar Current (A)", "Solar Irradiance (Lux)",
    "Battery Voltage (V)", "Battery Current (A)",
    "Power Output (W)", "Time of Day (hour,0–23)", "Day Type (0=Cloudy,1=Sunny)"
]
X = df[feature_cols].astype(float).values
y_eff = df["Efficiency_Class"].values.astype(int)
y_pow = df["Power Output (W)"].values.astype(float)

# ==================================================
# STEP 2: SPLIT DATA INTO 80% TRAIN, 10% VAL, 10% TEST
# ==================================================
X_train, X_temp, y_eff_train, y_eff_temp, y_pow_train, y_pow_temp = train_test_split(
    X, y_eff, y_pow, test_size=0.2, random_state=42, stratify=y_eff
)

X_val, X_test, y_eff_val, y_eff_test, y_pow_val, y_pow_test = train_test_split(
    X_temp, y_eff_temp, y_pow_temp, test_size=0.5, random_state=42, stratify=y_eff_temp
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "scaler_fnn.pkl")
print("✅ Scaler saved as scaler_fnn.pkl")

# Class distribution check
unique, counts = np.unique(y_eff_train, return_counts=True)
print("\n📊 Class Distribution in Training Set:")
for u, c in zip(unique, counts):
    label = ["Low (0)", "Medium (1)", "High (2)"][u]
    print(f"{label}: {c} samples")

# ==================================================
# STEP 3: BUILD OPTIMIZED MULTI-OUTPUT FNN MODEL
# ==================================================
inputs = tf.keras.Input(shape=(X_train.shape[1],))

x = layers.Dense(128, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(64, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(32, activation='relu')(x)

eff_out = layers.Dense(3, activation='softmax', name='efficiency')(x)
pow_out = layers.Dense(1, activation='linear', name='power')(x)

model = models.Model(inputs=inputs, outputs=[eff_out, pow_out])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={'efficiency': 'sparse_categorical_crossentropy', 'power': 'mse'},
    metrics={'efficiency': ['accuracy'], 'power': ['mae', 'mse']}
)

# ==================================================
# STEP 4: COMPUTE & BALANCE CLASS WEIGHTS
# ==================================================
classes = np.unique(y_eff_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_eff_train)
class_weight_dict = {int(k): float(v) for k, v in zip(classes, weights)}

# Adjust Medium class to be slightly higher for better balance
if 1 in class_weight_dict:
    class_weight_dict[1] *= 1.2  # increase Medium weight by 20%

# Normalize so total weights ≈ 3.0
total = sum(class_weight_dict.values())
for k in class_weight_dict:
    class_weight_dict[k] = (class_weight_dict[k] / total) * 3.0

# Cap extreme values for stability
MAX_WEIGHT_CAP = 10.0
for k in class_weight_dict:
    if class_weight_dict[k] > MAX_WEIGHT_CAP:
        class_weight_dict[k] = MAX_WEIGHT_CAP

print("\n⚖️ Final Adjusted Class Weights:")
for k, v in class_weight_dict.items():
    label = ["Low", "Medium", "High"][k]
    print(f"{label} ({k}): {v:.3f}")

# Save for reproducibility
with open("class_weights.json", "w") as f:
    json.dump(class_weight_dict, f, indent=4)
print("💾 Saved class weights → class_weights.json")

# Visualization of class weights
plt.figure(figsize=(5, 4))
plt.bar(["Low", "Medium", "High"], class_weight_dict.values(), color=['#4e79a7','#f28e2b','#76b7b2'])
plt.title("Adjusted Class Weights for Efficiency Classification")
plt.ylabel("Weight Value")
plt.tight_layout()
plt.show()

# Per-sample weights
sample_weight_train_eff = np.array([class_weight_dict[int(l)] for l in y_eff_train], dtype=float)
sample_weight_val_eff = np.array([class_weight_dict[int(l)] for l in y_eff_val], dtype=float)
sample_weight_train_power = np.ones_like(y_pow_train, dtype=float)
sample_weight_val_power = np.ones_like(y_pow_val, dtype=float)
sample_weight_train = {'efficiency': sample_weight_train_eff, 'power': sample_weight_train_power}
sample_weight_val = {'efficiency': sample_weight_val_eff, 'power': sample_weight_val_power}

# ==================================================
# STEP 5: TRAINING
# ==================================================
cb = [
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    callbacks.ModelCheckpoint("best_fnn_model.keras", save_best_only=True)
]

history = model.fit(
    X_train,
    {'efficiency': y_eff_train, 'power': y_pow_train},
    sample_weight=sample_weight_train,
    validation_data=(X_val, {'efficiency': y_eff_val, 'power': y_pow_val}, sample_weight_val),
    epochs=100,
    batch_size=64,
    callbacks=cb,
    verbose=1
)

# ==================================================
# STEP 6: EVALUATION
# ==================================================
y_eff_pred_probs, y_pow_pred = model.predict(X_test)
y_eff_pred_class = np.argmax(y_eff_pred_probs, axis=1)

acc = accuracy_score(y_eff_test, y_eff_pred_class)
prec = precision_score(y_eff_test, y_eff_pred_class, average='macro', zero_division=0)
rec = recall_score(y_eff_test, y_eff_pred_class, average='macro', zero_division=0)
f1 = f1_score(y_eff_test, y_eff_pred_class, average='macro', zero_division=0)

rmse = np.sqrt(mean_squared_error(y_pow_test, y_pow_pred))
mae = mean_absolute_error(y_pow_test, y_pow_pred)
r2 = r2_score(y_pow_test, y_pow_pred)
mape = np.mean(np.abs((y_pow_test - y_pow_pred.flatten()) / (y_pow_test + 1e-6))) * 100

cm = confusion_matrix(y_eff_test, y_eff_pred_class)
labels = ["Low", "Medium", "High"]
report = classification_report(y_eff_test, y_eff_pred_class, target_names=labels, zero_division=0)

# ==================================================
# STEP 8: BMS LOAD MANAGEMENT (POWER-BASED DECISION)
# ==================================================
predicted_power = float(np.mean(y_pow_pred))
battery_voltage = float(df["Battery Voltage (V)"].iloc[-1])

if predicted_power < 20:
    bms_status = "⚠️ LOW SOLAR INPUT — Lights OFF, Battery discharging."
elif predicted_power < 70:
    bms_status = "🟡 MODERATE INPUT — Partial load operation recommended."
else:
    bms_status = "✅ HIGH SOLAR INPUT — Full load can remain ON."

# ==================================================
# STEP 9: PERFORMANCE SUMMARY
# ==================================================
print("\n📊 EFFICIENCY CLASSIFICATION METRICS")
print("----------------------------------")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}\n")
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
# STEP 10: VISUALIZATION
# ==================================================
plt.figure(figsize=(10, 5))
plt.plot(history.history['efficiency_accuracy'], label='Train Acc')
plt.plot(history.history['val_efficiency_accuracy'], label='Val Acc', linestyle='--')
plt.title("Efficiency Classification Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['power_mae'], label='Train MAE')
plt.plot(history.history['val_power_mae'], label='Val MAE', linestyle='--')
plt.title("Power Prediction Mean Absolute Error")
plt.xlabel("Epochs")
plt.ylabel("MAE (W)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Efficiency Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.scatter(y_pow_test, y_pow_pred, alpha=0.7)
plt.title("Power Prediction (Actual vs Predicted)")
plt.xlabel("Actual Power (W)")
plt.ylabel("Predicted Power (W)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ==================================================
# STEP 11: EXPORT TO TFLITE
# ==================================================
def representative_dataset():
    for i in range(500):
        idx = np.random.randint(0, X_train.shape[0])
        yield [X_train[idx:idx+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open("optimized_fnn_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Exported: optimized_fnn_int8.tflite for ESP32-S3 deployment")
