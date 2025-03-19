import laspy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

#  Load the LAS file (Change file path as needed)
las_file_path = "C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Toy-example/Data/slope_with_abnormalities.las"  # 🔹 Change this to your .las file path
las = laspy.read(las_file_path)

# 🔹Extract X, Y, Z coordinates
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

# 🔹 Open3D Interactive Visualization
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# Visualize the point cloud interactively
o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Viewer")
