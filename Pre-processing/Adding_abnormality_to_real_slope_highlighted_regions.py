import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import griddata
import os

# ==== Step 1: Load LAS point cloud ====
laz_file_path = "C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/cleaned_slope.las"
las = laspy.read(laz_file_path)
xyz = np.vstack((las.x, las.y, las.z)).T
print(f"Total number of points: {len(xyz)}")

# ==== Step 2: Add artificial abnormality (hump or cavity) ====
ab_center_coords = (711414.5, 308255.2)  # X, Y
ab_radius = 1.5
amplitude = 1.0          # < 0 makes a cavity, > 0 a hump

dx = xyz[:, 0] - ab_center_coords[0]
dy = xyz[:, 1] - ab_center_coords[1]
dist_sq = dx**2 + dy**2
dz = amplitude * np.exp(-dist_sq / (2 * ab_radius**2))  # Gaussian
xyz[:, 2] += dz

# ==== Step 3: Separate normal and abnormal points (works for hump or cavity) ====
dz_threshold = 0.02 * abs(amplitude)     # scale threshold with amplitude
abnormal_mask = np.abs(dz) > dz_threshold

xyz_abnormal = xyz[abnormal_mask]
xyz_normal   = xyz[~abnormal_mask]

print(f"Abnormal pts: {xyz_abnormal.shape[0]}, Normal pts: {xyz_normal.shape[0]}")

# ==== Step 4: Grid and interpolate (with safety for small sets) ====
grid_size = 300
x_min, x_max = np.min(xyz[:, 0]), np.max(xyz[:, 0])
y_min, y_max = np.min(xyz[:, 1]), np.max(xyz[:, 1])
xi = np.linspace(x_min, x_max, grid_size)
yi = np.linspace(y_min, y_max, grid_size)
X_grid, Y_grid = np.meshgrid(xi, yi)

# Full grid for export
Z_grid_full = griddata((xyz[:, 0], xyz[:, 1]), xyz[:, 2], (X_grid, Y_grid), method='linear')
Z_masked_full = np.ma.masked_invalid(Z_grid_full)

def safe_griddata(pts, values, Xg, Yg):
    if pts.shape[0] < 3:
        # Not enough points to triangulate — return all-NaN grid
        return np.full_like(Xg, np.nan, dtype=float)
    return griddata((pts[:, 0], pts[:, 1]), values, (Xg, Yg), method='linear')

# Interpolate the two parts separately
Z_ab     = safe_griddata(xyz_abnormal, xyz_abnormal[:, 2], X_grid, Y_grid)
Z_normal = safe_griddata(xyz_normal,   xyz_normal[:, 2],   X_grid, Y_grid)

Z_ab     = np.ma.masked_invalid(Z_ab)
Z_normal = np.ma.masked_invalid(Z_normal)

# ==== Step 5: Plot ====
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# Blue background
ax.plot_wireframe(X_grid, Y_grid, Z_normal, color='blue', linewidth=0.4, rstride=2, cstride=2)
# Red abnormality
ax.plot_wireframe(X_grid, Y_grid, Z_ab,     color='red',  linewidth=0.6, rstride=2, cstride=2)

ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("Wireframe with Abnormality Highlighted (hump or cavity)")
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
