# Re-import necessary libraries since execution state was reset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the original coordinate system (X, Y, Z axes)
original_axes = np.array([
    [1, 0, 0],  # X-axis
    [0, 1, 0],  # Y-axis
    [0, 0, 1]   # Z-axis
])

# Example eigenvectors from PCA (columns represent PC1, PC2, PC3)
eigenvectors = np.array([
    [0.8, -0.5,  0.2],  # PC1 (Slope direction - New X-axis)
    [0.6,  0.7, -0.3],  # PC2 (Secondary variation - New Y-axis)
    [0.0,  0.4,  0.9]   # PC3 (Surface normal - New Z-axis)
])

# Define the origin
origin = np.array([0, 0, 0])

# Create a 3D figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the original coordinate system (X, Y, Z)
ax.quiver(*origin, *original_axes[:, 0], color='gray', linestyle='dashed', linewidth=1.5, label="Original X-axis")
ax.quiver(*origin, *original_axes[:, 1], color='gray', linestyle='dashed', linewidth=1.5, label="Original Y-axis")
ax.quiver(*origin, *original_axes[:, 2], color='gray', linestyle='dashed', linewidth=1.5, label="Original Z-axis")

# Plot the new PCA coordinate system (Principal Components)
ax.quiver(*origin, *eigenvectors[:, 0], color='red', linewidth=2, label="PC1 (New X-axis, Main Slope)")
ax.quiver(*origin, *eigenvectors[:, 1], color='green', linewidth=2, label="PC2 (New Y-axis, Secondary Variation)")
ax.quiver(*origin, *eigenvectors[:, 2], color='blue', linewidth=2, label="PC3 (New Z-axis, Surface Normal)")

# Labels and view settings
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel("Original X-axis")
ax.set_ylabel("Original Y-axis")
ax.set_zlabel("Original Z-axis")
ax.set_title("Old and New Coordinate Systems (PCA Eigenvectors)")
ax.legend()

# Show the plot
plt.show()
