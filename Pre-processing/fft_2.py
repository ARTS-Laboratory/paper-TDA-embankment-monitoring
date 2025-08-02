import numpy as np
import matplotlib.pyplot as plt
import laspy

# Load preprocessed and already-gridded data
las = laspy.read("C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/abnormality_hump_wireframe.las")
points = np.vstack((las.x, las.y, las.z)).T

# Assuming you already have gridded X, Y, Z
# Just extract the bounding box and reshape
grid_size = 250
x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
x_grid = np.linspace(x_min, x_max, grid_size)
y_grid = np.linspace(y_min, y_max, grid_size)
xv, yv = np.meshgrid(x_grid, y_grid)

# Assuming Z is already interpolated — no interpolation here
# Just reshape Z from LAS directly if stored or passed separately
# If you had saved Z_grid from an earlier step, just load and plot it

# Load Z_grid (if precomputed)
# Example placeholder (you should load this from file if available)
from scipy.interpolate import griddata
z_grid = griddata((points[:, 0], points[:, 1]), points[:, 2], (xv, yv), method='linear')
z_grid = np.nan_to_num(z_grid, nan=np.nanmean(z_grid))

# Plot the already-prepared heightmap
fig, ax = plt.subplots(figsize=(10, 6))

cmap = plt.get_cmap('terrain')
elev_map = ax.imshow(z_grid, cmap=cmap, extent=(x_min, x_max, y_min, y_max), origin='lower')

contour_levels = np.linspace(z_grid.min(), z_grid.max(), 15)
contours = ax.contour(xv, yv, z_grid, levels=contour_levels, colors='k', linewidths=0.6)
ax.clabel(contours, inline=True, fontsize=8)

ax.set_title("Slope Heightmap with Contour Overlays")
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
fig.colorbar(elev_map, ax=ax, label='Elevation (m)')

plt.tight_layout()
plt.show()

