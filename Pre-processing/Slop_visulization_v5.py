import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

# Load the LAZ file
laz_file_path = "C:/Users/golzardm/Documents/Dataset-Slope-LiDAR-Embankment-SLidE/Data/2021-06/laz/2021-06.laz"
las = laspy.read(laz_file_path)

# Extract X, Y, Z coordinates
xyz = np.vstack((las.x, las.y, las.z)).T

# Print the total number of points and ranges for debugging
print(f"Total number of points: {len(xyz)}")
print(f"X range (min, max): ({np.min(xyz[:, 0])}, {np.max(xyz[:, 0])})")
print(f"Y range (min, max): ({np.min(xyz[:, 1])}, {np.max(xyz[:, 1])})")
print(f"Z range (min, max): ({np.min(xyz[:, 2])}, {np.max(xyz[:, 2])})")

# Remove the adjacent road (filter by elevation)
z_threshold = 92  # Retain road removal threshold
road_mask = xyz[:, 2] <= z_threshold
xyz_road_removed = xyz[~road_mask]
print(f"Points after road removal: {len(xyz_road_removed)}")

# Define the four points for the river/strip (replace with your picked coordinates)
point1 = np.array([711410.123, 30.456, 89.789])  # Replace with your P1
point2 = np.array([711430.789, 30.123, 91.234])  # Replace with your P2
point3 = np.array([711410.456, 40.789, 92.345])  # Replace with your P3
point4 = np.array([711430.234, 40.123, 94.567])  # Replace with your P4

# Calculate the bounding box from the four points
x_min = min(point1[0], point2[0], point3[0], point4[0])
x_max = max(point1[0], point2[0], point3[0], point4[0])
y_min = min(point1[1], point2[1], point3[1], point4[1])
y_max = max(point1[1], point2[1], point3[1], point4[1])
z_min = min(point1[2], point2[2], point3[2], point4[2])  # Optional Z range
z_max = max(point1[2], point2[2], point3[2], point4[2])  # Optional Z range

print(f"Masking region: X=({x_min}, {x_max}), Y=({y_min}, {y_max}), Z=({z_min}, {z_max})")

# Create a mask to remove the river/strip region
river_mask = (xyz_road_removed[:, 0] >= x_min) & (xyz_road_removed[:, 0] <= x_max) & \
             (xyz_road_removed[:, 1] >= y_min) & (xyz_road_removed[:, 1] <= y_max) & \
             (xyz_road_removed[:, 2] >= z_min) & (xyz_road_removed[:, 2] <= z_max)

# Apply the mask to remove the river/strip
xyz_cleaned = xyz_road_removed[~river_mask]

# Print points before and after river/strip removal
print(f"Points before river/strip removal: {len(xyz_road_removed)}")
print(f"Points after river/strip removal: {len(xyz_cleaned)}")

# Randomly select a subset of points (e.g., 10,000 points) from cleaned data
num_points = 10000
if len(xyz_cleaned) > num_points:
    indices = np.random.choice(len(xyz_cleaned), num_points, replace=False)
    xyz_subset = xyz_cleaned[indices]
else:
    xyz_subset = xyz_cleaned

# 3D Scatter Plot using Matplotlib
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with terrain colormap for earth-like appearance
scatter = ax.scatter(xyz_subset[:, 0], xyz_subset[:, 1], xyz_subset[:, 2], s=1, c=xyz_subset[:, 2], cmap='terrain', alpha=0.6)

# Formatting
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("3D Point Cloud Visualization of Earth Embankment (Road and River/Strip Removed)")

plt.show()

