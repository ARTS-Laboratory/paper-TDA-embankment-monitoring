import numpy as np
import matplotlib.pyplot as plt
import laspy
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from scipy.interpolate import griddata
import os

# ==== Step 1: Load LiDAR Point Cloud ====
las = laspy.read("C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/Slope_hump_4.las")
points = np.vstack((las.x, las.y, las.z)).T

# ==== Step 2: Generate Heightmap ====
grid_res = 0.25
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()
x_grid = np.arange(x_min, x_max, grid_res)
y_grid = np.arange(y_min, y_max, grid_res)
xv, yv = np.meshgrid(x_grid, y_grid)
z_grid = griddata((points[:, 0], points[:, 1]), points[:, 2], (xv, yv), method='linear')
z_grid = np.nan_to_num(z_grid, nan=np.nanmean(z_grid))

# ==== Step 3: Interactive Polygon Selection ====
selected_polygon = []

def onselect(verts):
    selected_polygon.append(verts)
    plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
cmap = plt.get_cmap('terrain')
elev_map = ax.imshow(z_grid, cmap=cmap, extent=(x_min, x_max, y_min, y_max), origin='lower')
contour_levels = np.linspace(z_grid.min(), z_grid.max(), 15)
contours = ax.contour(xv, yv, z_grid, levels=contour_levels, colors='k', linewidths=0.5)
ax.clabel(contours, inline=True, fontsize=8)

#ax.set_title("Draw a polygon and press Enter or close the window")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
fig.colorbar(elev_map, ax=ax, label="Elevation (m)")

selector = PolygonSelector(ax, onselect, useblit=True)
plt.show(block=True)

# ==== Step 4: Extract and Save Selected Region ====
if selected_polygon:
    path = Path(selected_polygon[0])
    inside = path.contains_points(points[:, :2])
    selected = points[inside]

    if selected.shape[0] == 0:
        print(" No points found inside the selected polygon.")
    else:
        new_las = laspy.LasData(las.header)  # Corrected: no `.copy()` needed
        new_las.points = las.points[inside]  # Keep original point format
        output_path = os.path.join(os.getcwd(), "selected_region_1.las")
        new_las.write(output_path)
        print(f" Saved {selected.shape[0]} points to: {output_path}")

        # 3D Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(selected[:, 0], selected[:, 1], selected[:, 2],
                   c=selected[:, 2], cmap='terrain', s=1)
        ax.set_title("3D View of Selected Region")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()
else:
    print("No polygon was selected.")
