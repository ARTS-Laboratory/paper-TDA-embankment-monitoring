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

# ==== Step 2: Add artificial abnormality (optional) ====
ab_center_coords = (711414.5, 308255.2)
ab_radius = 1
amplitude = -1
dx = xyz[:, 0] - ab_center_coords[0]
dy = xyz[:, 1] - ab_center_coords[1]
dist_sq = dx**2 + dy**2
dz = amplitude * np.exp(-dist_sq / (2 * ab_radius**2))
xyz[:, 2] += dz

# ==== Step 3: Grid and interpolate ====
x_min, x_max = np.min(xyz[:, 0]), np.max(xyz[:, 0])
y_min, y_max = np.min(xyz[:, 1]), np.max(xyz[:, 1])
grid_size = 250
xi = np.linspace(x_min, x_max, grid_size)
yi = np.linspace(y_min, y_max, grid_size)
X_grid, Y_grid = np.meshgrid(xi, yi)

Z_grid = griddata((xyz[:, 0], xyz[:, 1]), xyz[:, 2], (X_grid, Y_grid), method='linear')
Z_masked = np.ma.masked_invalid(Z_grid)

# ==== Step 4: Plot wireframe ====
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X_grid, Y_grid, Z_masked, color='blue', linewidth=0.4, rstride=2, cstride=2)

ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("Clean Wireframe Mesh of Levee Slope (Masked NaNs)")
ax.set_box_aspect([np.ptp(xi), np.ptp(yi), np.ptp(xyz[:, 2]) * 3])
ax.view_init(elev=30, azim=135)
plt.tight_layout()
plt.show()

# ==== Step 5: Save interpolated grid as LAS ====
X_flat = X_grid.flatten()
Y_flat = Y_grid.flatten()
Z_flat = Z_masked.filled(np.nan).flatten()

valid = ~np.isnan(Z_flat)
grid_points = np.vstack((X_flat[valid], Y_flat[valid], Z_flat[valid])).T

header = laspy.LasHeader(point_format=3, version="1.2")
new_las = laspy.LasData(header)
new_las.x = grid_points[:, 0]
new_las.y = grid_points[:, 1]
new_las.z = grid_points[:, 2]

output_path = os.path.join(os.getcwd(), "wireframe_mesh_grid.las")
new_las.write(output_path)

print(f"✅ Saved wireframe grid to: {output_path}")
