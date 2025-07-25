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
#las = laspy.read("C:/Users/hp zbook g5/Documents/GitHub/paper-TDA-embankment-monitoring/Toy-example/Data/surface_with_smooth_circular_cavity_50.las")
las = laspy.read("C:/Users/GOLZARDM/Documents/paper-TDA-embankment-monitoring/Toy-example/Data/road_removed_with_cavity.las")
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
#fig = plt.figure(figsize=(6.5, 4), dpi=300)  # High DPI and shorter plot size
fig = plt.figure(figsize=(6.5, 4))
ax = fig.add_subplot(111, projection='3d')

# Combine cavity and hump points for a single plot with viridis coloring
points = np.vstack((cavity_points, hump_points))
colors = np.concatenate((cavity_colors[:, 0], hump_colors[:, 0]))

# Scatter plot with viridis colormap
sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, cmap='viridis', s=2)
ax.set_xlabel("X", fontsize=8)
ax.set_ylabel("Y", fontsize=8)
ax.set_zlabel("Z", fontsize=8)

# Colorbar for viridis color representation
cb = plt.colorbar(sc, ax=ax, pad=0.1)
cb.set_label("Feature Value", fontsize=8)

# Adjusted legend: smaller font size, transparent background, reduced spacing
#ax.legend(fontsize=8)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='z', labelsize=8)

# Save and display the plot
plt.savefig("3d_plot.png", bbox_inches='tight', pad_inches=0)
plt.show()

# Adjusted axis limits to center the object
ax.set_xlim(np.min(xyz[:, 0]), np.max(xyz[:, 0]))
ax.set_ylim(np.min(xyz[:, 1]), np.max(xyz[:, 1]))
ax.set_zlim(np.min(xyz[:, 2]), np.max(xyz[:, 2]))

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
fig = plt.figure(figsize=(3.5, 4), dpi=300)  # High DPI and shorter plot size
sns.heatmap(cov_matrix_before_pca, annot=True, fmt=".2f", cmap="coolwarm",
            annot_kws={"fontfamily": "serif", "fontsize": 12},  # Ensure serif font
            xticklabels=[r'X', r'Y', r'Z'],
            yticklabels=[r'X', r'Y', r'Z'])
plt.xlabel(r'dimensions', fontsize=10)  # X-axis label with font size 10
plt.ylabel(r'dimensions', fontsize=10)  # Y-axis label with font size 10
# Set the font size of x and y axis ticks
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout(pad=0.2)
plt.show()

# Covariance Matrix After PCA
fig = plt.figure(figsize=(3.5, 4), dpi=300)  # High DPI and shorter plot size
sns.heatmap(cov_matrix_after_pca, annot=True, fmt=".2f", cmap="coolwarm",
            annot_kws={"fontfamily": "serif", "fontsize": 12},  # Ensure serif font
            xticklabels=[r'PC1', r'PC2', r'PC3'],
            yticklabels=[r'PC1', r'PC2', r'PC3'])
plt.xlabel(r'principal components', fontsize=10)  # X-axis label with font size 10
plt.ylabel(r'principal components', fontsize=10)  # Y-axis label with font size 10
# Set the font size of x and y axis ticks
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout(pad=0.2)
plt.show()


