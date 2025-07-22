import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

# Load the LAZ file
laz_file_path = "C:/Users/golzardm/Documents/Dataset-Slope-LiDAR-Embankment-SLidE/Data/2021-06/laz/2021-06.laz"
las = laspy.read(laz_file_path)

# Extract X, Y, Z coordinates
xyz = np.vstack((las.x, las.y, las.z)).T

# Print the total number of points and Z range for debugging
print(f"Total number of points: {len(xyz)}")
print(f"Z range (min, max): ({np.min(xyz[:, 2])}, {np.max(xyz[:, 2])})")

# Randomly select a subset of points (e.g., 10,000 points)
num_points = 10000
if len(xyz) > num_points:
    indices = np.random.choice(len(xyz), num_points, replace=False)
    xyz_subset = xyz[indices]
else:
    xyz_subset = xyz

# Debug: Print Z range of subset
print(f"Subset Z range (min, max): ({np.min(xyz_subset[:, 2])}, {np.max(xyz_subset[:, 2])})")

# Remove the adjacent road (filter by elevation)
z_threshold = 92  # Set based on plot, but verify with raw data
road_mask = xyz_subset[:, 2] <= z_threshold  # Use <= to include points at exactly 89
xyz_cleaned = xyz_subset[~road_mask]

# Debug: Print number of points before and after removal
print(f"Points before road removal: {len(xyz_subset)}")
print(f"Points after road removal: {len(xyz_cleaned)}")

# 3D Scatter Plot using Matplotlib
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with terrain colormap for earth-like appearance
scatter = ax.scatter(xyz_cleaned[:, 0], xyz_cleaned[:, 1], xyz_cleaned[:, 2], s=1, c=xyz_cleaned[:, 2], cmap='terrain', alpha=0.6)

# Formatting
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("3D Point Cloud Visualization of Earth Embankment (Road Removed) - Click for Coordinates")

# Function to handle mouse click and get coordinates
def onclick(event):
    if ax.contains(event)[0]:  # Check if click is within axes
        x, y = event.xdata, event.ydata
        # Approximate Z from projection
        z = proj3d.proj_transform(x, y, 0, ax.get_proj())[2]
        # Find the nearest point in the cleaned data
        distances = np.sqrt((xyz_cleaned[:, 0] - x)**2 + (xyz_cleaned[:, 1] - y)**2 + (xyz_cleaned[:, 2] - z)**2)
        nearest_idx = np.argmin(distances)
        nearest_point = xyz_cleaned[nearest_idx]
        print(f"Clicked near: X={nearest_point[0]:.3f}, Y={nearest_point[1]:.3f}, Z={nearest_point[2]:.3f}")

# Connect the click event
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()