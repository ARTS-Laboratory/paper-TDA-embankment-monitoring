import laspy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Load the LAS file
las = laspy.read("C:/Users/GOLZARDM/Documents/paper-TDA-embankment-monitoring/Toy-example/Data/surface_with_smooth_circular_cavity_40.las")

# Extract X, Y, Z coordinates
xyz = np.vstack((las.x, las.y, las.z)).T

# Perform PCA
pca = PCA(n_components=3)
pc_values = pca.fit_transform(xyz)

# Find the cavity region (deepest part of PC3)
cavity_threshold = np.percentile(pc_values[:, 2], 10)  # Keep only bottom 2% (deepest points)
cavity_points = xyz[pc_values[:, 2] < cavity_threshold]

# Save only cavity points to LAS format
output_file = "cavity_only.las"
header = laspy.LasHeader(point_format=las.header.point_format.id, version=las.header.version)
filtered_las = laspy.LasData(header)

filtered_las.x, filtered_las.y, filtered_las.z = cavity_points[:, 0], cavity_points[:, 1], cavity_points[:, 2]
filtered_las.write(output_file)

# Enable interactive mode for zooming/panning in Spyder
plt.ion()

# Create Two 2D Scatter Plots (Top View & Side View)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Top-down view (XY plane)
ax[0].scatter(cavity_points[:, 0], cavity_points[:, 1], s=2)
ax[0].set_title("Top View (XY Plane)")
ax[0].set_xlabel("X Coordinate")
ax[0].set_ylabel("Y Coordinate")
ax[0].grid(True)

# Side view (XZ plane)
ax[1].scatter(cavity_points[:, 0], cavity_points[:, 2], s=2)
ax[1].set_title("Side View (XZ Plane)")
ax[1].set_xlabel("X Coordinate")
ax[1].set_ylabel("Z Coordinate")
ax[1].grid(True)

plt.show()

# Create Matplotlib 3D Plot (Like the Uploaded One)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for 3D visualization
sc = ax.scatter(cavity_points[:, 0], cavity_points[:, 1], cavity_points[:, 2], 
                c=cavity_points[:, 2], cmap='jet', s=2)

ax.set_title("3D Plot of Cavities & Abnormalities")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()

# Create Open3D 3D Interactive Visualization
o3d_pc = o3d.geometry.PointCloud()
o3d_pc.points = o3d.utility.Vector3dVector(cavity_points)
o3d.visualization.draw_geometries([o3d_pc])

print(f"Cavity-only point cloud saved to {output_file}")
