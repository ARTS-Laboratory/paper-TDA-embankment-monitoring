import numpy as np
import matplotlib.pyplot as plt
import laspy
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from scipy.interpolate import griddata
import os

# ==== Settings ====
las_path = r"C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/Slope_hump_4.las"
grid_res = 0.25
ROTATE_CW_90 = True  # rotate the map 90° clockwise for visualization/selection

# ==== 1) Load ====
las = laspy.read(las_path)
pts = np.vstack((las.x, las.y, las.z)).T  # original XYZ

# ==== 2) Optional: rotate XY about the dataset center ====
def rotate_xy_cw90(xy, cx, cy):
    # 90° CW rotation about (cx, cy): (x', y') = ( cx + (y-cy),  cy - (x-cx) )
    x, y = xy[:, 0], xy[:, 1]
    xr = cx + (y - cy)
    yr = cy - (x - cx)
    return np.column_stack([xr, yr])

if ROTATE_CW_90:
    cx = 0.5 * (pts[:, 0].min() + pts[:, 0].max())
    cy = 0.5 * (pts[:, 1].min() + pts[:, 1].max())
    xy_rot = rotate_xy_cw90(pts[:, :2], cx, cy)
    pts_rot = np.column_stack([xy_rot, pts[:, 2]])  # (x_rot, y_rot, z)
else:
    pts_rot = pts.copy()

# ==== 3) Grid & interpolate in the (possibly rotated) frame ====
x_min, x_max = pts_rot[:, 0].min(), pts_rot[:, 0].max()
y_min, y_max = pts_rot[:, 1].min(), pts_rot[:, 1].max()

x_grid = np.arange(x_min, x_max, grid_res)
y_grid = np.arange(y_min, y_max, grid_res)
xv, yv = np.meshgrid(x_grid, y_grid)

z_grid = griddata((pts_rot[:, 0], pts_rot[:, 1]), pts_rot[:, 2], (xv, yv), method='linear')
z_grid = np.nan_to_num(z_grid, nan=np.nanmean(z_grid))

# ==== 4) Interactive polygon on the rotated frame ====
selected_polygon = []

def onselect(verts):
    selected_polygon.append(verts)
    plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_facecolor("#8fd18f")
cmap = plt.get_cmap('terrain')

# Use pcolormesh so XY coordinates are respected
im = ax.pcolormesh(xv, yv, z_grid, cmap=cmap, shading="auto")
fig.colorbar(im, ax=ax, label="Elevation (m)")

levels = np.linspace(z_grid.min(), z_grid.max(), 15)
cs = ax.contour(xv, yv, z_grid, levels=levels, colors='k', linewidths=0.5)
ax.clabel(cs, inline=True, fontsize=8)

ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")

selector = PolygonSelector(ax, onselect, useblit=True)
plt.show(block=True)

# ==== 5) Extract indices via rotated XY, save original XYZ ====
if selected_polygon:
    path = Path(selected_polygon[0])
    # test inclusion on the rotated XY:
    inside = path.contains_points(pts_rot[:, :2])
    sel = pts[inside]  # save original (unrotated) points to LAS

    if sel.shape[0] == 0:
        print("No points found inside the selected polygon.")
    else:
        new_las = laspy.LasData(las.header)
        new_las.points = las.points[inside]
        out_path = os.path.join(os.getcwd(), "selected_region_1.las")
        new_las.write(out_path)
        print(f"Saved {sel.shape[0]} points to: {out_path}")

        # 3D preview of ORIGINAL geometry
        fig = plt.figure(figsize=(8, 6))
        ax3 = fig.add_subplot(111, projection="3d")
        sc = ax3.scatter(sel[:, 0], sel[:, 1], sel[:, 2], c=sel[:, 2], cmap="terrain", s=1)
        fig.colorbar(sc, ax=ax3, shrink=0.6, label="Elevation (m)")
        ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
        ax3.set_title("3D View of Selected Region")
        plt.tight_layout(); plt.show()
else:
    print("No polygon was selected.")
