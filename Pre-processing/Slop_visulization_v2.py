import laspy
import numpy as np
import matplotlib.pyplot as plt

# Load the LAZ file
laz_path = "C:/Users/golzardm/Documents/Dataset-Slope-LiDAR-Embankment-SLidE/Data/2021-06/laz/2021-06.laz"
#laz_path = "C:/Users/hp zbook g5/Documents/GitHub/Dataset-Slope-LiDAR-Embankment-SLidE/Data/2024-02/TerryRoad_Feb2024_GE_ReSampled.laz"

las = laspy.read(laz_path)

# Extract X, Y, Z coordinates
xyz = np.vstack((las.x, las.y, las.z)).T

# Print the total number of points and Z range for debugging
print(f"Total number of points: {len(xyz)}")
print(f"Z range (min, max): ({np.min(xyz[:, 2])}, {np.max(xyz[:, 2])})")

# Randomly select a subset of points (e.g., 10,000 points)
num_points = 18000
if len(xyz) > num_points:
    indices = np.random.choice(len(xyz), num_points, replace=False)
    xyz_subset = xyz[indices]
else:
    xyz_subset = xyz

# Debug: Print Z range of subset
print(f"Subset Z range (min, max): ({np.min(xyz_subset[:, 2])}, {np.max(xyz_subset[:, 2])})")
#%%
# Remove the adjacent road (filter by elevation)
z_threshold = 92  # the value should be  slightly more than data in z direction in plot
road_mask = xyz_subset[:, 2] <= z_threshold  # Use <= to include points at exactly 89
xyz_cleaned = xyz_subset[~road_mask]

# Debug: Print number of points before and after removal
print(f"Points before road removal: {len(xyz_subset)}")
print(f"Points after road removal: {len(xyz_cleaned)}")

# 3D Scatter Plot using Matplotlib
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with terrain colormap for earth-like appearance
ax.scatter(xyz_cleaned[:, 0], xyz_cleaned[:, 1], xyz_cleaned[:, 2], s=1, c=xyz_cleaned[:, 2], cmap='terrain', alpha=0.6)

# Formatting
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("3D Point Cloud Visualization of Earth Embankment (Road Removed)")

plt.show()