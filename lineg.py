import matplotlib.pyplot as plt
import numpy as np

# Metrics for comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
fnn_values = [0.9893, 0.9867, 0.9941, 0.9904, 0.9995]
lr_values = [0.9634, 0.9541, 0.9703, 0.9621, 0.9876]

x = np.arange(len(metrics))

plt.figure(figsize=(10,6))
plt.plot(x, fnn_values, marker='o', label='FNN', linewidth=2)
plt.plot(x, lr_values, marker='s', label='Logistic Regression', linewidth=2, linestyle='--')

plt.xticks(x, metrics)
plt.ylim(0.94, 1.00)
plt.title('FNN vs Logistic Regression Model Comparison', fontsize=14)
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
