# ==================================================
# LOGISTIC REGRESSION MODEL + COMPARISON WITH FNN
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
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib

# ==================================================
# STEP 1: LOAD DATA
# ==================================================
conn = sqlite3.connect("solar_data_collection.db")
df = pd.read_sql_query("SELECT * FROM solar_data", conn)
conn.close()

# Derived features
df["Solar_Power"] = df["Solar Voltage (V)"] * df["Solar Current (A)"]
df["Efficiency (%)"] = (df["Power Output (W)"] / (df["Solar_Power"] + 1e-6)) * 100
df["Efficiency_Class"] = np.where(df["Efficiency (%)"] >= 80, 1, 0)

# Inputs and outputs
X = df[[
    "Temperature (°C)", "Humidity (%)", "Solar Voltage (V)",
    "Solar Current (A)", "Solar Irradiance (Lux)",
    "Battery Voltage (V)", "Battery Current (A)",
    "Power Output (W)", "Time of Day (hour,0–23)", "Day Type (0=Cloudy,1=Sunny)"
]].values
y = df["Efficiency_Class"].values

# ==================================================
# STEP 2: SPLIT AND SCALE
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==================================================
# STEP 3: TRAIN LOGISTIC REGRESSION MODEL
# ==================================================
model_lr = LogisticRegression(max_iter=1000, solver='lbfgs', penalty='l2', C=1.0)
model_lr.fit(X_train, y_train)

# ==================================================
# STEP 4: EVALUATE LOGISTIC REGRESSION
# ==================================================
y_pred = model_lr.predict(X_test)
y_prob = model_lr.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Low Efficiency", "High Efficiency"], digits=4)

# ==================================================
# STEP 5: DISPLAY LR RESULTS
# ==================================================
print("\n📊 LOGISTIC REGRESSION CLASSIFICATION METRICS")
print("---------------------------------------------")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC-ROC  : {auc:.4f}\n")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# ==================================================
# STEP 6: SAVE MODEL
# ==================================================
joblib.dump(model_lr, "logreg_model.pkl")
joblib.dump(scaler, "logreg_scaler.pkl")
print("\n✅ Logistic Regression model saved as logreg_model.pkl")

# ==================================================
# STEP 7: PLOT CONFUSION MATRIX
# ==================================================
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("logreg_confusion_matrix.png", dpi=300)
plt.show()

# ==================================================
# STEP 8: ROC CURVE
# ==================================================
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc:.4f})', linewidth=2)
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("logreg_roc_curve.png", dpi=300)
plt.show()

# ==================================================
# STEP 9: COMPARE WITH FNN RESULTS
# ==================================================
fnn_results = {
    "Accuracy": 0.9893,
    "Precision": 0.9867,
    "Recall": 0.9941,
    "F1 Score": 0.9904,
    "AUC-ROC": 0.9995,
}

lr_results = {
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1 Score": f1,
    "AUC-ROC": auc,
}

# Convert to DataFrame
comparison_df = pd.DataFrame([fnn_results, lr_results], index=["Feedforward NN", "Logistic Regression"])
print("\n📈 MODEL COMPARISON")
print(comparison_df)

# ==================================================
# STEP 10: PLOT COMPARISON BAR CHART
# ==================================================
plt.figure(figsize=(8,5))
comparison_df.plot(kind='bar', figsize=(8,5), color=["#4CAF50", "#2196F3"])
plt.title("Model Performance Comparison: FNN vs Logistic Regression")
plt.ylabel("Score")
plt.ylim(0.8, 1.05)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.show()
