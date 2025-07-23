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

# Randomly select a subset of points (e.g., 10,000 points)
num_points = 10000
if len(xyz) > num_points:
    indices = np.random.choice(len(xyz), num_points, replace=False)
    xyz_subset = xyz[indices]
else:
    xyz_subset = xyz

# Remove the adjacent road (filter by elevation)
z_threshold = 92
road_mask = xyz_subset[:, 2] <= z_threshold
xyz_cleaned = xyz_subset[~road_mask]

# Print points before and after road removal
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
ax.set_title("3D Point Cloud Visualization - Click to Pick River/Strip Points")

# Function to handle mouse click and get coordinates
picked_points = []  # List to store picked coordinates
def onclick(event):
    if ax.contains(event)[0]:
        x, y = event.xdata, event.ydata
        z = proj3d.proj_transform(x, y, 0, ax.get_proj())[2]
        distances = np.sqrt((xyz_cleaned[:, 0] - x)**2 + (xyz_cleaned[:, 1] - y)**2 + (xyz_cleaned[:, 2] - z)**2)
        nearest_idx = np.argmin(distances)
        nearest_point = xyz_cleaned[nearest_idx]
        picked_points.append(nearest_point)
        print(f"Picked point {len(picked_points)}: X={nearest_point[0]:.3f}, Y={nearest_point[1]:.3f}, Z={nearest_point[2]:.3f}")
        if len(picked_points) == 4:  # Stop after 4 points
            print("Four points picked. Update code with these coordinates and rerun.")
            plt.close()

# Connect the click event
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()