import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os

# Load LAS file
laz_file_path = "C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/cleaned_slope.las"
las = laspy.read(laz_file_path)
xyz = np.vstack((las.x, las.y, las.z)).T
print(f"Total number of points: {len(xyz)}")

# Add abnormality (optional)
ab_center_coords = (711417.5, 308255.2)
ab_radius = 2
amplitude = 2
dx = xyz[:, 0] - ab_center_coords[0]
dy = xyz[:, 1] - ab_center_coords[1]
dist_sq = dx**2 + dy**2
dz = amplitude * np.exp(-dist_sq / (2 * ab_radius**2))
xyz[:, 2] += dz

# Build grid (limit to data bounds)
x_min, x_max = np.min(xyz[:, 0]), np.max(xyz[:, 0])
y_min, y_max = np.min(xyz[:, 1]), np.max(xyz[:, 1])
grid_size = 250
xi = np.linspace(x_min, x_max, grid_size)
yi = np.linspace(y_min, y_max, grid_size)
X_grid, Y_grid = np.meshgrid(xi, yi)

# Interpolate (use 'linear' to avoid overshoots from 'cubic')
Z_grid = griddata((xyz[:, 0], xyz[:, 1]), xyz[:, 2], (X_grid, Y_grid), method='linear')

# Mask invalid values to avoid plotting gaps
Z_masked = np.ma.masked_invalid(Z_grid)

# Plot
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X_grid, Y_grid, Z_masked, color='blue', linewidth=0.4, rstride=2, cstride=2)

# Labels and view
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("Clean Wireframe Mesh of Levee Slope (Masked NaNs)")

# Add vertical exaggeration
ax.set_box_aspect([
    np.ptp(xi),                  # X
    np.ptp(yi),                  # Y
    np.ptp(xyz[:, 2]) * 3        # Z stretched
])

ax.view_init(elev=30, azim=135)
plt.tight_layout()
plt.show()
