# Import required libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import laspy
import open3d as o3d
from sklearn.decomposition import PCA

# ✅ Apply LaTeX Formatting for Matplotlib
plt.rcParams.update({'text.usetex': True})  # Enable LaTeX rendering
plt.rcParams.update({'font.family': 'serif'})  # Use serif fonts
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})  
plt.rcParams.update({'font.size': 10})  # Standard font size
plt.rcParams.update({'mathtext.rm': 'serif'})  # Use serif fonts for math text
plt.rcParams.update({'mathtext.fontset': 'custom'})  # Use custom math fonts

# Load the LAS file
las = laspy.read("C:/Users/GOLZARDM/Documents/paper-TDA-embankment-monitoring/Toy-example/Data/Complex abnormalities.las")

# Extract X, Y, Z coordinates
xyz = np.vstack((las.x, las.y, las.z)).T

# Perform PCA
pca = PCA(n_components=3)
pc_values = pca.fit_transform(xyz)

# Calculate statistics
mean_pc3 = np.mean(pc_values[:, 2])
std_pc3 = np.std(pc_values[:, 2])

print("Mean PC3 Value:", mean_pc3)

# Remove Flat Surface
flat_surface_mask = (pc_values[:, 2] >= (mean_pc3 - 1.5 * std_pc3)) & (pc_values[:, 2] <= (mean_pc3 + 1.5 * std_pc3))
non_surface_points = xyz[~flat_surface_mask]  # Remove flat surface

# Identify Cavities (Low PC3) and Humps (High PC3)
cavity_points = xyz[pc_values[:, 2] < (mean_pc3 - 1.5 * std_pc3)]
hump_points = xyz[pc_values[:, 2] > (mean_pc3 + 1.5 * std_pc3)]

# Normalize PC3 values for color mapping (0 to 1 scale)
pc3_min, pc3_max = np.min(pc_values[:, 2]), np.max(pc_values[:, 2])
normalized_pc3 = (pc_values[:, 2] - pc3_min) / (pc3_max - pc3_min)

# Choose a vibrant colormap
cavity_colors = plt.cm.plasma(normalized_pc3[pc_values[:, 2] < (mean_pc3 - 1.5 * std_pc3)])
hump_colors = plt.cm.plasma(normalized_pc3[pc_values[:, 2] > (mean_pc3 + 1.5 * std_pc3)])

# ✅ 2D Scatter Plots (Top View & Side View) with LaTeX formatting
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Top-down view (XY plane)
ax[0].scatter(cavity_points[:, 0], cavity_points[:, 1], s=8, c=cavity_colors, label=r'\textbf{Cavities}')
ax[0].scatter(hump_points[:, 0], hump_points[:, 1], s=8, c=hump_colors, label=r'\textbf{Humps}')
ax[0].set_title(r'\textbf{Top View}', y=1.02)
ax[0].set_xlabel(r'\textbf{X Coordinate}')
ax[0].set_ylabel(r'\textbf{Y Coordinate}')
ax[0].grid(True)
ax[0].legend()

# Side view (XZ plane)
ax[1].scatter(cavity_points[:, 0], cavity_points[:, 2], s=8, c=cavity_colors, label=r'\textbf{Cavities}')
ax[1].scatter(hump_points[:, 0], hump_points[:, 2], s=8, c=hump_colors, label=r'\textbf{Humps}')
ax[1].set_title(r'\textbf{Side View}', y=1.02)
ax[1].set_xlabel(r'\textbf{X direction}')
ax[1].set_ylabel(r'\textbf{Z direction}')
ax[1].grid(True)
ax[1].legend()

plt.pause(0.001)

# ✅ 3D Scatter Plot in Matplotlib with LaTeX Formatting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(cavity_points[:, 0], cavity_points[:, 1], cavity_points[:, 2], 
                c=cavity_colors, s=8, label=r'\textbf{Cavities}')
sc2 = ax.scatter(hump_points[:, 0], hump_points[:, 1], hump_points[:, 2], 
                 c=hump_colors, s=8, label=r'\textbf{Humps}')

ax.set_title(r'\textbf{3D Plot with Strong Colors}', y=1.02)
ax.set_xlabel(r'\textbf{X}')
ax.set_ylabel(r'\textbf{Y}')
ax.set_zlabel(r'\textbf{Z}')
ax.legend()

plt.pause(0.001)

# Save the LAS file before launching Open3D visualization
output_file = "abnormalities_only.las"
header = laspy.LasHeader(point_format=las.header.point_format.id, version=las.header.version)
filtered_las = laspy.LasData(header)

filtered_las.x, filtered_las.y, filtered_las.z = non_surface_points[:, 0], non_surface_points[:, 1], non_surface_points[:, 2]
filtered_las.write(output_file)

print(f"Filtered point cloud (cavities & humps) saved to {output_file}")

# Interactive 3D Plot using Open3D (LaTeX not applicable in Open3D)
o3d_pc = o3d.geometry.PointCloud()
all_abnormal_points = np.vstack((cavity_points, hump_points))
o3d_pc.points = o3d.utility.Vector3dVector(all_abnormal_points)

# Convert colors to Open3D format (Nx3)
all_colors = np.vstack((cavity_colors[:, :3], hump_colors[:, :3])) ** 1.5  
all_colors = np.clip(all_colors, 0, 1)  
o3d_pc.colors = o3d.utility.Vector3dVector(all_colors)

vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=600)
vis.add_geometry(o3d_pc)

# Open3D Non-Blocking Mode
for _ in range(100):
    vis.poll_events()
    vis.update_renderer()

vis.destroy_window()

# Covariance Heatmaps Before and After PCA with LaTeX Formatting
cov_matrix_before_pca = np.cov(xyz.T)
cov_matrix_after_pca = np.cov(pc_values.T)

# Covariance Matrix Before PCA
plt.figure(figsize=(6, 5))
sns.heatmap(cov_matrix_before_pca, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=[r'\textbf{X}', r'\textbf{Y}', r'\textbf{Z}'],
            yticklabels=[r'\textbf{X}', r'\textbf{Y}', r'\textbf{Z}'])

plt.title(r'\textbf{Covariance Matrix Before PCA}')
plt.xlabel(r'\textbf{Dimensions}')
plt.ylabel(r'\textbf{Dimensions}')
plt.show()

# Covariance Matrix After PCA
plt.figure(figsize=(6, 5))
sns.heatmap(cov_matrix_after_pca, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=[r'\textbf{PC1}', r'\textbf{PC2}', r'\textbf{PC3}'],
            yticklabels=[r'\textbf{PC1}', r'\textbf{PC2}', r'\textbf{PC3}'])

plt.title(r'\textbf{Covariance Matrix After PCA}')
plt.xlabel(r'\textbf{Principal Components}')
plt.ylabel(r'\textbf{Principal Components}')
plt.show()


