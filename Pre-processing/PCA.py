import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import laspy
from sklearn.decomposition import PCA

# LaTeX-style plot settings
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})

# Load the LAS file
las = laspy.read("C:/Users/GOLZARDM/Documents/paper-TDA-embankment-monitoring/Toy-example/Data/road_removed_with_cavity.las")
xyz = np.vstack((las.x, las.y, las.z)).T

# PCA transformation
pca = PCA(n_components=3)
pc_values = pca.fit_transform(xyz)

# Threshold factor
threshold_factor = 1.5  # Adjust this to control outlier strictness

# Compute mean and std
mean_pc3 = np.mean(pc_values[:, 2])
std_pc3 = np.std(pc_values[:, 2])

# Masking surface and outliers
flat_surface_mask = (pc_values[:, 2] >= (mean_pc3 - threshold_factor * std_pc3)) & \
                     (pc_values[:, 2] <= (mean_pc3 + threshold_factor * std_pc3))
non_surface_points = xyz[~flat_surface_mask]

# Separate cavity and hump points
cavity_points = xyz[pc_values[:, 2] < (mean_pc3 - threshold_factor * std_pc3)]
hump_points = xyz[pc_values[:, 2] > (mean_pc3 + threshold_factor * std_pc3)]

# Normalize PC3 values
pc3_min, pc3_max = np.min(pc_values[:, 2]), np.max(pc_values[:, 2])
normalized_pc3 = (pc_values[:, 2] - pc3_min) / (pc3_max - pc3_min)

cavity_colors = plt.cm.plasma(normalized_pc3[pc_values[:, 2] < (mean_pc3 - threshold_factor * std_pc3)])
hump_colors = plt.cm.plasma(normalized_pc3[pc_values[:, 2] > (mean_pc3 + threshold_factor * std_pc3)])

# 2D Top View
plt.figure(figsize=(6.5, 4))
plt.scatter(cavity_points[:, 0], cavity_points[:, 1], s=8, c=cavity_colors, label="Cavities")
plt.scatter(hump_points[:, 0], hump_points[:, 1], s=8, c=hump_colors, label="Humps")
plt.title("Top View (X-Y Plane)", fontsize=10)
plt.xlabel("X Coordinate", fontsize=8)
plt.ylabel("Y Coordinate", fontsize=8)
plt.grid(True)
plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), framealpha=1, fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout(pad=0)
plt.show()

# 2D Side View
plt.figure(figsize=(6.5, 4))
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

# 3D Plot with Subsampling for Speed
points = np.vstack((cavity_points, hump_points))
colors = np.concatenate((cavity_colors[:, 0], hump_colors[:, 0]))

# Subsample for faster rendering
max_plot_points = 8000
if points.shape[0] > max_plot_points:
    indices = np.random.choice(points.shape[0], max_plot_points, replace=False)
    points = points[indices]
    colors = colors[indices]

fig = plt.figure(figsize=(6.5, 4))  # Avoid high dpi for interactivity
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, cmap='viridis', s=2)

ax.set_xlabel("X", fontsize=8)
ax.set_ylabel("Y", fontsize=8)
ax.set_zlabel("Z", fontsize=8)
cb = plt.colorbar(sc, ax=ax, pad=0.1)
cb.set_label("Feature Value", fontsize=8)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='z', labelsize=8)
plt.tight_layout()
plt.show()

# Save filtered points
output_file = "abnormalities_only.las"
header = laspy.LasHeader(point_format=las.header.point_format.id, version=las.header.version)
filtered_las = laspy.LasData(header)
filtered_las.x = non_surface_points[:, 0]
filtered_las.y = non_surface_points[:, 1]
filtered_las.z = non_surface_points[:, 2]
filtered_las.write(output_file)
print(f"Filtered point cloud saved to: {output_file}")

# Covariance Heatmaps
cov_matrix_before_pca = np.cov(xyz.T)
cov_matrix_after_pca = np.cov(pc_values.T)

fig = plt.figure(figsize=(3.5, 4))
sns.heatmap(cov_matrix_before_pca, annot=True, fmt=".2f", cmap="coolwarm",
            annot_kws={"fontfamily": "serif", "fontsize": 12},
            xticklabels=['X', 'Y', 'Z'],
            yticklabels=['X', 'Y', 'Z'])
plt.xlabel("dimensions", fontsize=10)
plt.ylabel("dimensions", fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout(pad=0.2)
plt.show()

fig = plt.figure(figsize=(3.5, 4))
sns.heatmap(cov_matrix_after_pca, annot=True, fmt=".2f", cmap="coolwarm",
            annot_kws={"fontfamily": "serif", "fontsize": 12},
            xticklabels=['PC1', 'PC2', 'PC3'],
            yticklabels=['PC1', 'PC2', 'PC3'])
plt.xlabel("principal components", fontsize=10)
plt.ylabel("principal components", fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout(pad=0.2)
plt.show()
