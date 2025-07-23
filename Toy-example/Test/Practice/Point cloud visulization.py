import laspy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# Load the LAS file (Change file path as needed)
las_file_path = "C:/Users/golzardm/Documents/Dataset-Slope-LiDAR-Embankment-SLidE/Data/2024-02/TerryRoad_Feb2024_GE_ReSampled.laz"  # 🔹 Change this to your .las file path
las = laspy.read(las_file_path)

# Extract X, Y, Z coordinates
xyz = np.vstack((las.x, las.y, las.z)).T

#  3D Scatter Plot using Matplotlib
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, c=xyz[:, 2], cmap='jet', alpha=0.5)

# Formatting
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("3D Point Cloud Visualization")

plt.show()


