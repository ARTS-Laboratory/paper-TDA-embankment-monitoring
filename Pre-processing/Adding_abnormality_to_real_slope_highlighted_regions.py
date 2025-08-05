import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os

# ==== Step 1: Load LAS point cloud ====
laz_file_path = "C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/cleaned_slope.las"
las = laspy.read(laz_file_path)
xyz = np.vstack((las.x, las.y, las.z)).T
print(f"Total number of points: {len(xyz)}")

# ==== Step 2: Add artificial abnormality (hump or cavity) ====
ab_center_coords = (711414.5, 308255.2)
ab_radius = 2.0
amplitude = 2.0

dx = xyz[:, 0] - ab_center_coords[0]
dy = xyz[:, 1] - ab_center_coords[1]
dist_sq = dx**2 + dy**2
dz = amplitude * np.exp(-dist_sq / (2 * ab_radius**2))
xyz[:, 2] += dz

# ==== Step 3: Separate normal and abnormal points ====
dz_threshold = 0.01  # Only significant bumps are considered abnormal
abnormal_mask = dz > dz_threshold

xyz_abnormal = xyz[abnormal_mask]
xyz_normal = xyz[~abnormal_mask]

# ==== Step 4: Grid and interpolate separately ====
grid_size = 250
x_min, x_max = np.min(xyz[:, 0]), np.max(xyz[:, 0])
y_min, y_max = np.min(xyz[:, 1]), np.max(xyz[:, 1])
xi = np.linspace(x_min, x_max, grid_size)
yi = np.linspace(y_min, y_max, grid_size)
X_grid, Y_grid = np.meshgrid(xi, yi)

# Interpolate full Z grid first for LAS saving
Z_grid_full = griddata((xyz[:, 0], xyz[:, 1]), xyz[:, 2], (X_grid, Y_grid), method='linear')
Z_masked_full = np.ma.masked_invalid(Z_grid_full)

# Interpolate separately for visualizing normal and abnormal parts
Z_ab = griddata((xyz_abnormal[:, 0], xyz_abnormal[:, 1]), xyz_abnormal[:, 2], (X_grid, Y_grid), method='linear')
Z_normal = griddata((xyz_normal[:, 0], xyz_normal[:, 1]), xyz_normal[:, 2], (X_grid, Y_grid), method='linear')

Z_ab = np.ma.masked_invalid(Z_ab)
Z_normal = np.ma.masked_invalid(Z_normal)

# ==== Step 5: Plot ====
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot blue normal surface
ax.plot_wireframe(X_grid, Y_grid, Z_normal, color='blue', linewidth=0.4, rstride=2, cstride=2)

# Plot red abnormality
ax.plot_wireframe(X_grid, Y_grid, Z_ab, color='red', linewidth=0.6, rstride=2, cstride=2)

ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("Only Abnormality Area in Red (dz-based separation)")
ax.set_box_aspect([np.ptp(xi), np.ptp(yi), np.ptp(xyz[:, 2]) * 3])
ax.view_init(elev=30, azim=135)
plt.tight_layout()
plt.show()

# ==== Step 6: Save interpolated mesh as LAS ====
X_flat = X_grid.flatten()
Y_flat = Y_grid.flatten()
Z_flat = Z_masked_full.filled(np.nan).flatten()

valid = ~np.isnan(Z_flat)
grid_points = np.vstack((X_flat[valid], Y_flat[valid], Z_flat[valid])).T

header = laspy.LasHeader(point_format=3, version="1.2")
new_las = laspy.LasData(header)
new_las.x = grid_points[:, 0]
new_las.y = grid_points[:, 1]
new_las.z = grid_points[:, 2]

output_path = os.path.join(os.getcwd(), "wireframe_mesh_colored.las")
new_las.write(output_path)
print(f"✅ Saved wireframe mesh to: {output_path}")
