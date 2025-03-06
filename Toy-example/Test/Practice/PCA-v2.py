import laspy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# load point cloud
las = laspy.read("C:/Users/GOLZARDM/Documents/paper-TDA-embankment-monitoring/Toy-example/Data/surface_with_smooth_circular_hump_50.las")

# turn the point cloud into array
xyz = np.vstack((las.x, las.y, las.z)).T

# Performing the PCA for 3D 
pca = PCA(n_components=3)
pc_values = pca.fit_transform(xyz) # tins line transfer data into the center points and find the covariance matrix
                                   # the covariance matrix decomposed to fine eignvectors and eignvalues(pc1,pc2 and pc3)
                                   # the point cloud projected onto new principle axies
                                   
# extractin the cavity region 
cavity_threshold = np.percentile(pc_values[:, 2], 90)    # extract the third PC (pc3) with 90% of pc3 from bottom 
cavity_points = xyz[pc_values[:, 2] > cavity_threshold]  # only the points belong to the humps extracted

# Save the abnormalities to LAS format
output_file = "cavity_only.las"
header = laspy.LasHeader(point_format=las.header.point_format.id, version=las.header.version)
filtered_las = laspy.LasData(header)

filtered_las.x, filtered_las.y, filtered_las.z = cavity_points[:, 0], cavity_points[:, 1], cavity_points[:, 2]
filtered_las.write(output_file)

# interactive mode for plotting
plt.ion()

# 2D Scatter Plots (Top View & Side View)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Top-down view (XY plane)
ax[0].scatter(cavity_points[:, 0], cavity_points[:, 1], s=2)
ax[0].set_title("Top View ")
ax[0].set_xlabel("X Coordinate")
ax[0].set_ylabel("Y Coordinate")
ax[0].grid(True)

# Side view (XZ plane)
ax[1].scatter(cavity_points[:, 0], cavity_points[:, 2], s=2)
ax[1].set_title("Side View ")
ax[1].set_xlabel("X direction")
ax[1].set_ylabel("Z direction")
ax[1].grid(True)

plt.show()

# 3D Plot 
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for 3D visualization
sc = ax.scatter(cavity_points[:, 0], cavity_points[:, 1], cavity_points[:, 2], 
                c=cavity_points[:, 2], cmap='jet', s=2)

ax.set_title("3D Plot ")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()

# Interactive plot
o3d_pc = o3d.geometry.PointCloud()
o3d_pc.points = o3d.utility.Vector3dVector(cavity_points)
o3d.visualization.draw_geometries([o3d_pc])

print(f"Cavity-only point cloud saved to {output_file}")

