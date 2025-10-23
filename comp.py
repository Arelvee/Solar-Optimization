import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error,
    mean_absolute_error, r2_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, Model

# -------------------------------
# 1. Load Dataset from SQLite
# -------------------------------
conn = sqlite3.connect("solar_data_collection.db")
df = pd.read_sql_query("SELECT * FROM solar_data", conn)  # <-- adjust table name
conn.close()

# Convert to numeric (in case of type mismatch)
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# Compute Solar Power and Efficiency
df["Solar_Power"] = df["Current"] * df["Voltage"]
df["Efficiency"] = (df["Solar_Power"] / (df["Irradiance"] * df["Area"])) * 100
df["Efficient"] = (df["Efficiency"] >= 80).astype(int)

features = ["Irradiance", "Temperature", "Voltage", "Current", "Area"]
target_eff = "Efficient"
target_pow = "Solar_Power"

X = df[features]
y_eff = df[target_eff]
y_pow = df[target_pow]

# Split dataset
X_train, X_test, y_eff_train, y_eff_test, y_pow_train, y_pow_test = train_test_split(
    X, y_eff, y_pow, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 2. Logistic Regression (Baseline)
# -------------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_eff_train)

y_eff_pred_lr = lr.predict(X_test)
y_eff_pred_prob_lr = lr.predict_proba(X_test)[:, 1]

# Metrics
acc = accuracy_score(y_eff_test, y_eff_pred_lr)
prec = precision_score(y_eff_test, y_eff_pred_lr)
rec = recall_score(y_eff_test, y_eff_pred_lr)
f1 = f1_score(y_eff_test, y_eff_pred_lr)
auc = roc_auc_score(y_eff_test, y_eff_pred_prob_lr)

print("\n📊 LOGISTIC REGRESSION - EFFICIENCY CLASSIFICATION METRICS")
print("----------------------------------------------------------")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC-ROC  : {auc:.4f}")

cm_lr = confusion_matrix(y_eff_test, y_eff_pred_lr)
print("\nConfusion Matrix:\n", cm_lr)
print("\nClassification Report:\n", classification_report(y_eff_test, y_eff_pred_lr))

# -------------------------------
# 3. Feedforward Neural Network
# -------------------------------
inputs = tf.keras.Input(shape=(X_train.shape[1],))
x = layers.Dense(64, activation="relu")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.BatchNormalization()(x)

eff_out = layers.Dense(1, activation="sigmoid", name="efficiency")(x)
pow_out = layers.Dense(1, activation="linear", name="power")(x)

model = Model(inputs=inputs, outputs=[eff_out, pow_out])
model.compile(optimizer="adam",
              loss={"efficiency": "binary_crossentropy", "power": "mse"},
              metrics={"efficiency": ["accuracy"], "power": ["mae"]})

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
]

history = model.fit(
    X_train,
    {"efficiency": y_eff_train, "power": y_pow_train},
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=0
)

# Predictions
eff_pred_fnn_prob, pow_pred_fnn = model.predict(X_test)
y_eff_pred_fnn = (eff_pred_fnn_prob > 0.5).astype(int)

# Metrics (FNN)
acc_fnn = accuracy_score(y_eff_test, y_eff_pred_fnn)
prec_fnn = precision_score(y_eff_test, y_eff_pred_fnn)
rec_fnn = recall_score(y_eff_test, y_eff_pred_fnn)
f1_fnn = f1_score(y_eff_test, y_eff_pred_fnn)
auc_fnn = roc_auc_score(y_eff_test, eff_pred_fnn_prob)

rmse_fnn = np.sqrt(mean_squared_error(y_pow_test, pow_pred_fnn))
mae_fnn = mean_absolute_error(y_pow_test, pow_pred_fnn)
r2_fnn = r2_score(y_pow_test, pow_pred_fnn)
mape_fnn = np.mean(np.abs((y_pow_test - pow_pred_fnn) / y_pow_test)) * 100

# -------------------------------
# 4. Performance Comparison
# -------------------------------
lr_scores = [acc, prec, rec, f1, auc]
fnn_scores = [acc_fnn, prec_fnn, rec_fnn, f1_fnn, auc_fnn]
labels = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]

plt.figure(figsize=(8, 5))
x = np.arange(len(labels))
width = 0.35
plt.bar(x - width/2, lr_scores, width, label="Logistic Regression", alpha=0.8)
plt.bar(x + width/2, fnn_scores, width, label="FNN", alpha=0.8)
plt.xticks(x, labels)
plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.title("Performance Comparison: FNN vs Logistic Regression")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# -------------------------------
# 5. Confusion Matrix Plot
# -------------------------------
plt.figure(figsize=(5,4))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["Low", "High"], yticklabels=["Low", "High"])
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# -------------------------------
# 6. Summary Table
# -------------------------------
summary_df = pd.DataFrame({
    "Metric": labels,
    "FNN": fnn_scores,
    "Logistic Regression": lr_scores
})
print("\n📊 MODEL COMPARISON SUMMARY")
print("----------------------------------------------------------")
print(summary_df.to_string(index=False))

print("\n🔋 POWER PREDICTION (FNN ONLY)")
print("----------------------------------------------------------")
print(f"RMSE : {rmse_fnn:.4f}")
print(f"MAE  : {mae_fnn:.4f}")
print(f"R²   : {r2_fnn:.4f}")
print(f"MAPE : {mape_fnn:.2f}%")

# -------------------------------
# 7. Training Curves (Optional)
# -------------------------------
plt.figure(figsize=(8, 4))
plt.plot(history.history['efficiency_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_efficiency_accuracy'], label='Val Accuracy')
plt.title("FNN Training Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
