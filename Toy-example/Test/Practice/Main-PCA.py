import numpy as np
import matplotlib.pyplot as plt
import laspy
import open3d as o3d
from sklearn.decomposition import PCA

# load the LAS file
las = laspy.read("C:/Users/GOLZARDM/Documents/paper-TDA-embankment-monitoring/Toy-example/Data/Complex abnormalities.las")


xyz = np.vstack((las.x, las.y, las.z)).T

# PCA performing in this step
pca = PCA(n_components=3)
pc_values = pca.fit_transform(xyz)

# mean & standard deveiation calculated
mean_pc3 = np.mean(pc_values[:, 2])
std_pc3 = np.std(pc_values[:, 2])

print("Mean PC3 Value:", mean_pc3)

# flat surface removing
flat_surface_mask = (pc_values[:, 2] >= (mean_pc3 - 1.5 * std_pc3)) & (pc_values[:, 2] <= (mean_pc3 + 1.5 * std_pc3))
non_surface_points = xyz[~flat_surface_mask]  

#  cavities (Low PC3) and humps (High PC3) identified
cavity_points = xyz[pc_values[:, 2] < (mean_pc3 - 1.5 * std_pc3)]
hump_points = xyz[pc_values[:, 2] > (mean_pc3 + 1.5 * std_pc3)]

#  PC3 values for color mapping normalized between 0 and 1
pc3_min, pc3_max = np.min(pc_values[:, 2]), np.max(pc_values[:, 2])
normalized_pc3 = (pc_values[:, 2] - pc3_min) / (pc3_max - pc3_min)  

#### vibrant colormap like plasma, turbo, jet
cavity_colors = plt.cm.plasma(normalized_pc3[pc_values[:, 2] < (mean_pc3 - 1.5 * std_pc3)])
hump_colors = plt.cm.plasma(normalized_pc3[pc_values[:, 2] > (mean_pc3 + 1.5 * std_pc3)])

# 2D Scatter plots (Top View & Side View) with stronger colors
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

### Top-down view (in X-Y plane
ax[0].scatter(cavity_points[:, 0], cavity_points[:, 1], s=8, c=cavity_colors, label="Cavities")  # Increased point size
ax[0].scatter(hump_points[:, 0], hump_points[:, 1], s=8, c=hump_colors, label="Humps")  # Increased point size
ax[0].set_title("Top View ")
ax[0].set_xlabel("X Coordinate")
ax[0].set_ylabel("Y Coordinate")
ax[0].grid(True)
ax[0].legend()

# Side view in X-Z plane
ax[1].scatter(cavity_points[:, 0], cavity_points[:, 2], s=8, c=cavity_colors, label="Cavities")  # Increased point size
ax[1].scatter(hump_points[:, 0], hump_points[:, 2], s=8, c=hump_colors, label="Humps")  # Increased point size
ax[1].set_title("Side View ")
ax[1].set_xlabel("X direction")
ax[1].set_ylabel("Z direction")
ax[1].grid(True)
ax[1].legend()

#### allowing the script to continue while showing plots foe avoiding clash
plt.show(block=False)

#  3D Plot using the Matplotlib with Vibrant Colors
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# scatter plot 
sc = ax.scatter(cavity_points[:, 0], cavity_points[:, 1], cavity_points[:, 2], 
                c=cavity_colors, s=8, label="cavities")  # Increased point size
sc2 = ax.scatter(hump_points[:, 0], hump_points[:, 1], hump_points[:, 2], 
                 c=hump_colors, s=8, label="humps")  # Increased point size

ax.set_title("3D Plot")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

#### allowing the script to continue while showing plots for avoiding clash
plt.show(block=False)

###### in this section the the filtered data in this case cavities and humps or all together saved and prepared for TDA  
# save the LAS file before launching Open3D visualization.
output_file = "abnormalities_only.las"
header = laspy.LasHeader(point_format=las.header.point_format.id, version=las.header.version)
filtered_las = laspy.LasData(header)

filtered_las.x, filtered_las.y, filtered_las.z = non_surface_points[:, 0], non_surface_points[:, 1], non_surface_points[:, 2]
filtered_las.write(output_file)

print(f"Filtered point cloud (cavities & humps) saved to {output_file}")

# interactive 3D plot using Open3D with color mapping
o3d_pc = o3d.geometry.PointCloud()
all_abnormal_points = np.vstack((cavity_points, hump_points))
o3d_pc.points = o3d.utility.Vector3dVector(all_abnormal_points)

# convert colors to open3D format (Nx3) and enhance contrast
all_colors = np.vstack((cavity_colors[:, :3], hump_colors[:, :3])) ** 1.5  # brightness adjustment
all_colors = np.clip(all_colors, 0, 1)  #  values within range (0-1)
o3d_pc.colors = o3d.utility.Vector3dVector(all_colors)

# fix open3D any viewport issue & allow proper visualization
vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=600)  # ensure window size is set
vis.add_geometry(o3d_pc)  # add the point cloud
vis.run()  # run the visualization
vis.destroy_window()  # close the window properly after visualization
