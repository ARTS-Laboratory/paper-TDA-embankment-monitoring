import laspy
import numpy as np
import matplotlib.pyplot as plt

# Load the LAZ file
laz_file_path = "C:/Users/golzardm/Documents/Dataset-Slope-LiDAR-Embankment-SLidE/Data/2024-02/TerryRoad_Feb2024_GE_ReSampled.laz"  # 🔹 Updated to .laz file
las = laspy.read(laz_file_path)

# Extract X, Y, Z coordinates
xyz = np.vstack((las.x, las.y, las.z)).T

# Print the total number of points
print(f"Total number of points: {len(xyz)}")
#%%
# Randomly select a subset of points (e.g., 10,000 points)
num_points = 10000
if len(xyz) > num_points:
    indices = np.random.choice(len(xyz), num_points, replace=False)
    xyz_subset = xyz[indices]
else:
    xyz_subset = xyz

# 3D Scatter Plot using Matplotlib
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with terrain colormap for earth-like appearance
ax.scatter(xyz_subset[:, 0], xyz_subset[:, 1], xyz_subset[:, 2], s=1, c=xyz_subset[:, 2], cmap='terrain', alpha=0.6)

# Formatting
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("3D Point Cloud Visualization of Earth Embankment")

plt.show()