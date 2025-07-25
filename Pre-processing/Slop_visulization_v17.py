import laspy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os

# ----------------------------
# Load LAZ file
# ----------------------------
laz_file_path = "C:/Users/golzardm/Documents/Dataset-Slope-LiDAR-Embankment-SLidE/Data/2021-06/laz/2021-06.laz"
las = laspy.read(laz_file_path)
xyz = np.vstack((las.x, las.y, las.z)).T
print(f"Total number of points: {len(xyz)}")

# ----------------------------
# Road removal
# ----------------------------
z_threshold = 92
mask = xyz[:, 2] > z_threshold
xyz = xyz[mask]
print(f"Points after road removal: {len(xyz)}")

# ----------------------------
# Abnormality parameters
# ----------------------------
ab_center_coords = (711412, 308250.2)
ab_radius = 5.0
amplitude = -2.0
dx = xyz[:, 0] - ab_center_coords[0]
dy = xyz[:, 1] - ab_center_coords[1]
dist_sq = dx**2 + dy**2
dz = amplitude * np.exp(-dist_sq / (2 * ab_radius**2))
xyz[:, 2] += dz

# ----------------------------
# Subsample for visualization
# ----------------------------
num_points = 30000
xyz_subset = xyz[np.random.choice(len(xyz), num_points, replace=False)] if len(xyz) > num_points else xyz

# ----------------------------
# Plot
# ----------------------------
z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
norm = colors.Normalize(vmin=z_min, vmax=z_max)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(xyz_subset[:, 0], xyz_subset[:, 1], xyz_subset[:, 2],
                s=1, c=xyz_subset[:, 2], cmap='viridis', norm=norm, alpha=0.6)

ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("3D Point Cloud with Road Removed and Cavity Applied")
ax.set_zlim(z_min, z_max)
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label("Elevation (Z)", rotation=270, labelpad=15)
ax.set_box_aspect([
    np.ptp(xyz_subset[:, 0]),
    np.ptp(xyz_subset[:, 1]),
    np.ptp(xyz_subset[:, 2]) * 3
])
ax.view_init(elev=30, azim=135)
plt.tight_layout()
plt.show()

# ----------------------------
# Save filtered + modified data to new LAS file
# ----------------------------
from laspy import LasHeader, LasData

header = LasHeader(point_format=las.header.point_format, version=las.header.version)
header.offsets = np.min(xyz, axis=0)
header.scales = las.header.scales  # Reuse same scale

new_las = LasData(header)
new_las.X = ((xyz[:, 0] - header.offsets[0]) / header.scales[0]).astype(np.int32)
new_las.Y = ((xyz[:, 1] - header.offsets[1]) / header.scales[1]).astype(np.int32)
new_las.Z = ((xyz[:, 2] - header.offsets[2]) / header.scales[2]).astype(np.int32)

output_path = os.path.join(os.getcwd(), "road_removed_with_cavity.las")
new_las.write(output_path)
print(f"Saved filtered and modified LAS to:\n{output_path}")
