import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import laspy
import open3d as o3d
from sklearn.decomposition import PCA

# Apply LaTeX Formatting for Matplotlib Plots
plt.rcParams.update({'text.usetex': True})  
plt.rcParams.update({'font.family': 'serif'})  
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})  
plt.rcParams.update({'font.size': 8})  
plt.rcParams.update({'mathtext.rm': 'serif'})  
plt.rcParams.update({'mathtext.fontset': 'custom'})  

# Load the LAS file
las = laspy.read("C:/Users/GOLZARDM/Documents/paper-TDA-embankment-monitoring/Toy-example/Data/Complex abnormalities.las")
xyz = np.vstack((las.x, las.y, las.z)).T

# PCA performing in this step
pca = PCA(n_components=3)
pc_values = pca.fit_transform(xyz)

# Mean & standard deviation calculated
mean_pc3 = np.mean(pc_values[:, 2])
std_pc3 = np.std(pc_values[:, 2])

# Flat surface removing
flat_surface_mask = (pc_values[:, 2] >= (mean_pc3 - 1.5 * std_pc3)) & (pc_values[:, 2] <= (mean_pc3 + 1.5 * std_pc3))
non_surface_points = xyz[~flat_surface_mask]

# Cavities (Low PC3) and humps (High PC3) identified
cavity_points = xyz[pc_values[:, 2] < (mean_pc3 - 1.5 * std_pc3)]
hump_points = xyz[pc_values[:, 2] > (mean_pc3 + 1.5 * std_pc3)]

# PC3 values for color mapping normalized between 0 and 1
pc3_min, pc3_max = np.min(pc_values[:, 2]), np.max(pc_values[:, 2])
normalized_pc3 = (pc_values[:, 2] - pc3_min) / (pc3_max - pc3_min)

# Vibrant colormap like plasma, turbo, jet
cavity_colors = plt.cm.plasma(normalized_pc3[pc_values[:, 2] < (mean_pc3 - 1.5 * std_pc3)])
hump_colors = plt.cm.plasma(normalized_pc3[pc_values[:, 2] > (mean_pc3 + 1.5 * std_pc3)])

# Plot 1: 2D Scatter Plot (Top View - X-Y plane)
plt.figure(figsize=(6.5, 4), dpi=300)  # High DPI and shorter plot size
plt.scatter(cavity_points[:, 0], cavity_points[:, 1], s=8, c=cavity_colors, label="Cavities")
plt.scatter(hump_points[:, 0], hump_points[:, 1], s=8, c=hump_colors, label="Humps")
plt.title("Top View (X-Y Plane)", fontsize=10)
plt.xlabel("X Coordinate", fontsize=8)
plt.ylabel("Y Coordinate", fontsize=8)
plt.grid(True)
# Adjusted legend: smaller font size, transparent background, reduced spacing
plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), framealpha=1, fontsize=8, frameon=True)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout(pad=0)
plt.show()

# Plot 2: 2D Scatter Plot (Side View - X-Z plane)
plt.figure(figsize=(6.5, 4), dpi=300)  # High DPI and shorter plot size
plt.scatter(cavity_points[:, 0], cavity_points[:, 2], s=8, c=cavity_colors, label="Cavities")
plt.scatter(hump_points[:, 0], hump_points[:, 2], s=8, c=hump_colors, label="Humps")
plt.title("Side View (X-Z Plane)", fontsize=10)
plt.xlabel("X Direction", fontsize=8)
plt.ylabel("Z Direction", fontsize=8)
plt.grid(True)
# Adjusted legend: smaller font size, transparent background, reduced spacing
#plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), framealpha=1, fontsize=8, frameon=True)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout(pad=0)
plt.show()

# Plot 3: 3D Scatter Plot
fig = plt.figure(figsize=(6.5, 4), dpi=300)  # High DPI and shorter plot size
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(cavity_points[:, 0], cavity_points[:, 1], cavity_points[:, 2], c=cavity_colors, s=8, label="Cavities")
sc2 = ax.scatter(hump_points[:, 0], hump_points[:, 1], hump_points[:, 2], c=hump_colors, s=8, label="Humps")
ax.set_title("3D Scatter Plot", fontsize=10)
ax.set_xlabel("X", fontsize=8)
ax.set_ylabel("Y", fontsize=8)
ax.set_zlabel("Z", fontsize=8)
# Adjusted legend: smaller font size, transparent background, reduced spacing
#plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), framealpha=1, fontsize=8, frameon=True)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='z', labelsize=8)
#plt.tight_layout(pad=0.1)
ax.set_box_aspect([1, 1, 1])  # Equal scaling for all three dimensions
plt.show()

# Save the filtered data for TDA
output_file = "abnormalities_only.las"
header = laspy.LasHeader(point_format=las.header.point_format.id, version=las.header.version)
filtered_las = laspy.LasData(header)

filtered_las.x, filtered_las.y, filtered_las.z = non_surface_points[:, 0], non_surface_points[:, 1], non_surface_points[:, 2]
filtered_las.write(output_file)

print(f"Filtered point cloud (cavities & humps) saved to {output_file}")

# Covariance Heatmaps Before and After PCA with LaTeX Formatting
cov_matrix_before_pca = np.cov(xyz.T)
cov_matrix_after_pca = np.cov(pc_values.T)

# Covariance Matrix Before PCA
fig = plt.figure(figsize=(6.5, 4), dpi=300)  # High DPI and shorter plot size
sns.heatmap(cov_matrix_before_pca, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=[r'\textbf{X}', r'\textbf{Y}', r'\textbf{Z}'],
            yticklabels=[r'\textbf{X}', r'\textbf{Y}', r'\textbf{Z}'])

plt.title(r'\textbf{Covariance Matrix Before PCA}')
plt.xlabel(r'\textbf{Dimensions}')
plt.ylabel(r'\textbf{Dimensions}')
plt.tight_layout(pad=0)
plt.show()

# Covariance Matrix After PCA
fig = plt.figure(figsize=(6.5, 4), dpi=300)  # High DPI and shorter plot size
sns.heatmap(cov_matrix_after_pca, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=[r'\textbf{PC1}', r'\textbf{PC2}', r'\textbf{PC3}'],
            yticklabels=[r'\textbf{PC1}', r'\textbf{PC2}', r'\textbf{PC3}'])

plt.title(r'\textbf{Covariance Matrix After PCA}')
plt.xlabel(r'\textbf{Principal Components}')
plt.ylabel(r'\textbf{Principal Components}')
plt.tight_layout(pad=0)
plt.show()

