import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Simulated 3D point cloud (small sample for understanding)
xyz_sample = np.array([
    [10, 50, 5],   # Point 1
    [12, 52, 6],   # Point 2
    [15, 55, 8],   # Point 3
    [18, 58, 10],  # Point 4
    [20, 60, 12],  # Point 5
])

# Step 1: Compute the Mean of Each Dimension (X, Y, Z)
mean_x = np.mean(xyz_sample[:, 0])  # Mean of X
mean_y = np.mean(xyz_sample[:, 1])  # Mean of Y
mean_z = np.mean(xyz_sample[:, 2])  # Mean of Z

mean_vector = np.array([mean_x, mean_y, mean_z])

# Step 2: Center the Data by Subtracting the Mean
xyz_centered = xyz_sample - mean_vector

# Step 3: Compute the Covariance Matrix
cov_matrix = np.cov(xyz_centered.T)  # NumPy computes covariance correctly

# Create a heatmap of the covariance matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=["X", "Y", "Z"], yticklabels=["X", "Y", "Z"])

plt.title("Covariance Matrix Heatmap")
plt.xlabel("Dimensions")
plt.ylabel("Dimensions")
plt.show()
