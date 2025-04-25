import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion matrix values from LUMED KAN
data = [
    [92.22, 4.68, 1.63, 1.53, 0, 0, 0],
    [3.19, 90.53, 3.68, 2.98, 1.72, 1.02, 0],
    [0.38, 3.98, 87.48, 3.78, 2.98, 1.72, 0],
    [0.32, 1.12, 2.68, 90.72, 3.98, 2.92, 0.62],
    [0.25, 1.15, 2.48, 5.08, 90.19, 2.92, 0],
    [1.84, 2.04, 2.68, 2.78, 4.08, 88.16, 2.62],
    [0, 0.87, 1.98, 1.88, 2.58, 5.02, 89.51],
]

# Convert to numpy array
conf_matrix = np.array(data)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Greens", xticklabels=range(7), yticklabels=range(7),
            annot_kws={"size": 14})  # Increase font size for annotations

# Increase the font size for axis labels and title
plt.xlabel("Predicted Labels", fontsize=16)
plt.ylabel("True Labels", fontsize=16)
# plt.title("Confusion Matrix")
plt.show()