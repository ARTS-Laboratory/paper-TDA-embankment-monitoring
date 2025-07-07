import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import laspy
import open3d as o3d
from sklearn.decomposition import PCA

# LaTeX Formatting 
plt.rcParams.update({'text.usetex': True})  
plt.rcParams.update({'font.family': 'serif'})  
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})  
plt.rcParams.update({'font.size': 8})  
plt.rcParams.update({'mathtext.rm': 'serif'})  
plt.rcParams.update({'mathtext.fontset': 'custom'})  

# loading the LAS file
las = laspy.read("C:/Users/GOLZARDM/Documents/paper-TDA-embankment-monitoring/Toy-example/Data/surface_with_smooth_circular_hump_50.las")
xyz = np.vstack((las.x, las.y, las.z)).T

# PCA performing in this step
pca = PCA(n_components=3)
pc_values = pca.fit_transform(xyz)

# mean & standard deviation calculation
mean_pc3 = np.mean(pc_values[:, 2])
std_pc3 = np.std(pc_values[:, 2])

#  here the flat surface removing
flat_surface_mask = (pc_values[:, 2] >= (mean_pc3 - 1.5 * std_pc3)) & (pc_values[:, 2] <= (mean_pc3 + 1.5 * std_pc3))
non_surface_points = xyz[~flat_surface_mask]

# cavities with low PC3 and humps with high PC3 detection
cavity_points = xyz[pc_values[:, 2] < (mean_pc3 - 1.5 * std_pc3)]
hump_points = xyz[pc_values[:, 2] > (mean_pc3 + 1.5 * std_pc3)]

# PC3 values for color mapping normalized between 0 and 1
pc3_min, pc3_max = np.min(pc_values[:, 2]), np.max(pc_values[:, 2])
normalized_pc3 = (pc_values[:, 2] - pc3_min) / (pc3_max - pc3_min)

# vibrant colormap like plasma, turbo, jet
cavity_colors = plt.cm.plasma(normalized_pc3[pc_values[:, 2] < (mean_pc3 - 1.5 * std_pc3)])
hump_colors = plt.cm.plasma(normalized_pc3[pc_values[:, 2] > (mean_pc3 + 1.5 * std_pc3)])

# Plot 1: 2D Scatter Plot in the top View - X-Y plane
plt.figure(figsize=(6.5, 4), dpi=300)
plt.scatter(cavity_points[:, 0], cavity_points[:, 1], s=8, c=cavity_colors, label="Cavities")
plt.scatter(hump_points[:, 0], hump_points[:, 1], s=8, c=hump_colors, label="Humps")
plt.title("Top View (X-Y Plane)", fontsize=10)
plt.xlabel("X Coordinate", fontsize=8)
plt.ylabel("Y Coordinate", fontsize=8)
plt.grid(True)
plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), framealpha=1, fontsize=8, frameon=True)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout(pad=0)
plt.show()

# 2D Scatter Plot in the side View - X-Z plane
plt.figure(figsize=(6.5, 4), dpi=300)
plt.scatter(cavity_points[:, 0], cavity_points[:, 2], s=8, c=cavity_colors, label="Cavities")
plt.scatter(hump_points[:, 0], hump_points[:, 2], s=8, c=hump_colors, label="Humps")
plt.title("Side View (X-Z Plane)", fontsize=10)
plt.xlabel("X Direction", fontsize=8)
plt.ylabel("Z Direction", fontsize=8)
plt.grid(True)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout(pad=0)
plt.show()

# 3D Scatter Plot
fig = plt.figure(figsize=(6.5, 4), dpi=300)
ax = fig.add_subplot(111, projection='3d')
points = np.vstack((cavity_points, hump_points))
colors = np.concatenate((cavity_colors[:, 0], hump_colors[:, 0]))
sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, cmap='viridis', s=8)
ax.set_xlabel("X", fontsize=8)
ax.set_ylabel("Y", fontsize=8)
ax.set_zlabel("Z", fontsize=8)
cb = plt.colorbar(sc, ax=ax, pad=0.1)
cb.set_label("Feature Value", fontsize=8)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='z', labelsize=8)
plt.savefig("3d_plot.png", bbox_inches='tight', pad_inches=0)
plt.show()

ax.set_xlim(np.min(xyz[:, 0]), np.max(xyz[:, 0]))
ax.set_ylim(np.min(xyz[:, 1]), np.max(xyz[:, 1]))
ax.set_zlim(np.min(xyz[:, 2]), np.max(xyz[:, 2]))

# save the filtered data for TDA
output_file = "abnormalities_only.las"
header = laspy.LasHeader(point_format=las.header.point_format.id, version=las.header.version)
filtered_las = laspy.LasData(header)
filtered_las.x, filtered_las.y, filtered_las.z = non_surface_points[:, 0], non_surface_points[:, 1], non_surface_points[:, 2]
filtered_las.write(output_file)
print(f"Filtered point cloud (cavities & humps) saved to {output_file}")

# covariance Heatmaps Before and After PCA with modifications
cov_matrix_before_pca = np.cov(xyz.T)
cov_matrix_after_pca = np.cov(pc_values.T)

# round to integers for display
rounded_before = np.round(cov_matrix_before_pca).astype(int)
rounded_after = np.round(cov_matrix_after_pca).astype(int)

# covariance Matrix Before PCA
fig = plt.figure(figsize=(2, 2), dpi=350)
heatmap_before = sns.heatmap(
    rounded_before, annot=True, fmt="d", cmap="viridis",
    annot_kws={"fontfamily": "serif", "fontsize": 8},
    xticklabels=[r'X', r'Y', r'Z'],
    yticklabels=[r'X', r'Y', r'Z']
)
plt.xlabel(r'dimensions', fontsize=8)
plt.ylabel(r'dimensions', fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
cb = heatmap_before.collections[0].colorbar
cb.set_label("covariance Value", fontsize=8)
cb.ax.tick_params(labelsize=8)  
plt.tight_layout(pad=0.2)
plt.show()

# covariance Matrix After PCA
fig = plt.figure(figsize=(2, 2), dpi=350)
heatmap_after = sns.heatmap(
    rounded_after, annot=True, fmt="d", cmap="viridis",
    annot_kws={"fontfamily": "serif", "fontsize": 8},
    xticklabels=[r'PC1', r'PC2', r'PC3'],
    yticklabels=[r'PC1', r'PC2', r'PC3']
)
plt.xlabel(r'principal components', fontsize=8)
plt.ylabel(r'principal components', fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
cb = heatmap_after.collections[0].colorbar
cb.set_label("covariance value", fontsize=8)
cb.ax.tick_params(labelsize=8)  
plt.tight_layout(pad=0.2)
plt.show()

