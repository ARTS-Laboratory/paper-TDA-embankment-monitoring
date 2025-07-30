import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import os

# ----------------------------
# Load LAZ file using laspy
# ----------------------------
laz_file_path = "C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/cleaned_slope.las"

las = laspy.read(laz_file_path)
xyz = np.vstack((las.x, las.y, las.z)).T
print(f" Total number of points: {len(xyz)}")

# ----------------------------
# Print coordinate bounds
# ----------------------------
x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
print(f"X range: {x_min:.2f} to {x_max:.2f}")
print(f"Y range: {y_min:.2f} to {y_max:.2f}")
print(f"Z range: {z_min:.2f} to {z_max:.2f}")

# ----------------------------
# Set cavity/hump parameters
# ----------------------------
ab_center_coords = (711413.5, 308255.2)   # Customize this to your real-world center
ab_radius = 3.0                           # Radius of abnormality
amplitude = -2                            # Negative for cavity, positive for hump
print(f" Using cavity center: {ab_center_coords}")
print(f" Radius: {ab_radius:.2f}, Amplitude: {amplitude}")

# ----------------------------
# Apply Gaussian deformation
# ----------------------------
dx = xyz[:, 0] - ab_center_coords[0]
dy = xyz[:, 1] - ab_center_coords[1]
dist_sq = dx**2 + dy**2

dz = amplitude * np.exp(-dist_sq / (2 * ab_radius**2))
xyz[:, 2] += dz

dz_min, dz_max = dz.min(), dz.max()
print(f" dz range: min = {dz_min:.3f}, max = {dz_max:.3f}")

# ----------------------------
# Subsample for plotting (max 30,000 points)
# ----------------------------
num_points = 30000
if len(xyz) > num_points:
    indices = np.random.choice(len(xyz), num_points, replace=False)
    xyz_subset = xyz[indices]
else:
    xyz_subset = xyz

# ----------------------------
# Plot using matplotlib
# ----------------------------
z_color_min = xyz[:, 2].min()
z_color_max = xyz[:, 2].max()
norm = colors.Normalize(vmin=z_color_min, vmax=z_color_max)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    xyz_subset[:, 0], xyz_subset[:, 1], xyz_subset[:, 2],
    s=1, c=xyz_subset[:, 2], cmap='viridis', norm=norm, alpha=0.6
)

# Label axes
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("3D Point Cloud with Cavity/Hump Abnormality")

# Match colorbar to Z
ax.set_zlim(z_color_min, z_color_max)
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label("Elevation (Z)", rotation=270, labelpad=15)

# Equal aspect ratio
ax.set_box_aspect([
    np.ptp(xyz_subset[:, 0]),
    np.ptp(xyz_subset[:, 1]),
    np.ptp(xyz_subset[:, 2]) * 3  # Stretch Z for clarity
])

# View angle
ax.view_init(elev=30, azim=135)
plt.tight_layout()
plt.show()

# ----------------------------
# Save modified LAS file
# ----------------------------
las.X = ((xyz[:, 0] - las.header.offsets[0]) / las.header.scales[0]).astype(np.int32)
las.Y = ((xyz[:, 1] - las.header.offsets[1]) / las.header.scales[1]).astype(np.int32)
las.Z = ((xyz[:, 2] - las.header.offsets[2]) / las.header.scales[2]).astype(np.int32)

output_path = os.path.join(os.getcwd(), "clean_slope_abnormality_added.las")
las.write(output_path)
print(f" Saved modified LAS file to:\n{output_path}")
