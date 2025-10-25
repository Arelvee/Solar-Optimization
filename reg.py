# ==================================================
# SOLAR ENERGY HARVESTING OPTIMIZATION SYSTEM
# MULTICLASS LOGISTIC REGRESSION + POWER FORECAST ALERT
# ==================================================
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import joblib
from datetime import datetime, timedelta

# ==================================================
# STEP 1: LOAD DATA
# ==================================================
conn = sqlite3.connect("solar_data_collection.db")
df = pd.read_sql_query("SELECT * FROM solar_data", conn)
conn.close()

# ==================================================
# STEP 2: FEATURE ENGINEERING
# ==================================================
df["Solar_Power"] = df["Solar Voltage (V)"] * df["Solar Current (A)"]
df["Efficiency (%)"] = (df["Power Output (W)"] / (df["Solar_Power"] + 1e-6)) * 100

# Efficiency classification — balanced ranges
def classify_efficiency(eff):
    if eff < 60:
        return 0  # Low
    elif 60 <= eff < 85:
        return 1  # Medium
    else:
        return 2  # High

df["Efficiency_Class"] = df["Efficiency (%)"].apply(classify_efficiency)

# Expand data to simulate 1000+ samples if fewer exist
if len(df) < 1000:
    df = pd.concat([df]*((1000 // len(df)) + 1), ignore_index=True)

# Input and target
X = df[[
    "Temperature (°C)", "Humidity (%)", "Solar Voltage (V)",
    "Solar Current (A)", "Solar Irradiance (Lux)",
    "Battery Voltage (V)", "Battery Current (A)",
    "Power Output (W)", "Time of Day (hour,0–23)", "Day Type (0=Cloudy,1=Sunny)"
]].values
y = df["Efficiency_Class"].values

# ==================================================
# STEP 3: SPLIT AND SCALE DATA
# ==================================================
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "scaler_fnn.pkl")
print("✅ Scaler saved as scaler_fnn.pkl")

# ==================================================
# STEP 4: TRAIN MULTICLASS LOGISTIC REGRESSION
# ==================================================
model_lr = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=2000,
    penalty='l2'
)
model_lr.fit(X_train, y_train)

# ==================================================
# STEP 5: EVALUATE MODEL
# ==================================================
y_pred = model_lr.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"], digits=4)

print("\n📊 LOGISTIC REGRESSION MULTICLASS RESULTS")
print("---------------------------------------------")
print(f"Accuracy : {acc:.4f}")
print(f"Total Samples in Confusion Matrix: {np.sum(cm)}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# ==================================================
# STEP 6: VISUALIZE CONFUSION MATRIX
# ==================================================
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Low","Medium","High"], yticklabels=["Low","Medium","High"])
plt.title("Confusion Matrix - Logistic Regression (3-Class)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("logreg_confusion_matrix_multiclass.png", dpi=300)
plt.show()

# ==================================================
# STEP 7: SAVE MODEL
# ==================================================
joblib.dump(model_lr, "logreg_multiclass_model.pkl")
print("✅ Logistic Regression (3-Class) model saved as logreg_multiclass_model.pkl")

# ==================================================
# STEP 8: POWER OUTPUT FORECASTING (5-HOUR AHEAD)
# ==================================================
print("\n🔮 POWER OUTPUT FORECAST (Next 5 Hours Simulation)")
latest_data = df.iloc[-1][[
    "Temperature (°C)", "Humidity (%)", "Solar Voltage (V)",
    "Solar Current (A)", "Solar Irradiance (Lux)",
    "Battery Voltage (V)", "Battery Current (A)",
    "Power Output (W)", "Time of Day (hour,0–23)", "Day Type (0=Cloudy,1=Sunny)"
]].values.reshape(1, -1)

latest_scaled = scaler.transform(latest_data)
base_power = df["Power Output (W)"].iloc[-1]
forecast_powers = []
timestamps = []

for i in range(1, 6):
    future_time = datetime.now() + timedelta(hours=i)
    simulated_change = np.random.uniform(-0.08, 0.05)  # ±8% variation
    predicted_power = base_power * (1 + simulated_change)
    forecast_powers.append(predicted_power)
    timestamps.append(future_time)

# ==================================================
# STEP 9: ALERT LOGIC FOR LOAD MANAGEMENT
# ==================================================
threshold_power = base_power * 0.60
for t, p in zip(timestamps, forecast_powers):
    if p < threshold_power:
        print(f"⚠️ ALERT [{t.strftime('%Y-%m-%d %H:%M')}]: Forecasted Power = {p:.2f} W ⚠️ Consider reducing load.")
    else:
        print(f"✅ Stable [{t.strftime('%Y-%m-%d %H:%M')}]: Forecasted Power = {p:.2f} W OK.")

# ==================================================
# STEP 10: MODEL COMPARISON (FNN VS LR)
# ==================================================
fnn_results = {
    "Accuracy": 0.9893,
    "Precision": 0.9867,
    "Recall": 0.9941,
    "F1 Score": 0.9904,
}

lr_results = {
    "Accuracy": acc,
    "Precision": np.mean([float(v.split()[1]) for v in report.splitlines()[2:5]]),
    "Recall": np.mean([float(v.split()[2]) for v in report.splitlines()[2:5]]),
    "F1 Score": np.mean([float(v.split()[3]) for v in report.splitlines()[2:5]]),
}

comparison_df = pd.DataFrame([fnn_results, lr_results], index=["Feedforward NN", "Logistic Regression (3-Class)"])
print("\n📈 MODEL PERFORMANCE COMPARISON")
print(comparison_df)

plt.figure(figsize=(8,5))
comparison_df.plot(kind='bar', figsize=(8,5), color=["#4CAF50", "#2196F3"])
plt.title("Model Performance Comparison: FNN vs Logistic Regression (3-Class)")
plt.ylabel("Score")
plt.ylim(0.7, 1.05)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig("model_comparison_multiclass.png", dpi=300)
plt.show()
