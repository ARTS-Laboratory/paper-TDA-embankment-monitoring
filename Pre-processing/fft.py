import numpy as np
import matplotlib.pyplot as plt
import laspy
from scipy.interpolate import griddata

# Step 1: Load your LiDAR point cloud
las = laspy.read("C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/abnormality_hump_wireframe.las")
points = np.vstack((las.x, las.y, las.z)).T

# Step 2: Interpolate to a grid
grid_res = 0.25  # meters
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()
x_grid = np.arange(x_min, x_max, grid_res)
y_grid = np.arange(y_min, y_max, grid_res)
xv, yv = np.meshgrid(x_grid, y_grid)
z_grid = griddata((points[:, 0], points[:, 1]), points[:, 2], (xv, yv), method='linear')
z_grid = np.nan_to_num(z_grid, nan=np.nanmean(z_grid))

# Step 3: Plot the heightmap with contour lines
fig, ax = plt.subplots(figsize=(10, 6))

# Background color shading
cmap = plt.get_cmap('terrain')
elev_map = ax.imshow(z_grid, cmap=cmap, extent=(x_min, x_max, y_min, y_max), origin='lower')

# Contour lines
contour_levels = np.linspace(z_grid.min(), z_grid.max(), 15)
contours = ax.contour(xv, yv, z_grid, levels=contour_levels, colors='k', linewidths=0.6)
ax.clabel(contours, inline=True, fontsize=8)

# Labels
ax.set_title("Slope Heightmap with Contour Overlays")
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
fig.colorbar(elev_map, ax=ax, label='Elevation (m)')

plt.tight_layout()
plt.show()
