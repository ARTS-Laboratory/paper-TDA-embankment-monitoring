import os
import numpy as np
import matplotlib.pyplot as plt
import laspy
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from scipy.interpolate import griddata

# ================== Settings ==================
LAS_PATH   = r"C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/Slope_cavity_2.las"
GRID_RES   = 0.25
THETA_DEG  = 80.0  # + clockwise, - counterclockwise (any angle)
OUT_NAME   = "selected_region_1.las"
# ==============================================

# ---- Load point cloud ----
las = laspy.read(LAS_PATH)
pts = np.vstack((las.x, las.y, las.z)).T  # (N,3)

# ---- Rotate XY by arbitrary angle about a pivot (dataset center by default) ----
def rotate_xy(xy, theta_deg, pivot):
    """Rotate 2D points by theta_deg about pivot=(cx, cy). +theta = clockwise."""
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    # clockwise rotation matrix
    R = np.array([[ c,  s],
                  [-s,  c]])
    out = (xy - pivot) @ R.T + pivot
    return out

cx = 0.5 * (pts[:, 0].min() + pts[:, 0].max())
cy = 0.5 * (pts[:, 1].min() + pts[:, 1].max())
pivot = np.array([cx, cy])

xy_rot = rotate_xy(pts[:, :2], THETA_DEG, pivot)
pts_rot = np.column_stack([xy_rot, pts[:, 2]])  # (x_rot, y_rot, z)

# ---- Build grid in rotated frame & interpolate ----
x_min, x_max = pts_rot[:, 0].min(), pts_rot[:, 0].max()
y_min, y_max = pts_rot[:, 1].min(), pts_rot[:, 1].max()
xg = np.arange(x_min, x_max, GRID_RES)
yg = np.arange(y_min, y_max, GRID_RES)
xv, yv = np.meshgrid(xg, yg)

z_grid = griddata((pts_rot[:, 0], pts_rot[:, 1]), pts_rot[:, 2], (xv, yv), method="linear")

# ---- Plot (white background, NaNs white) + interactive polygon ----
selected_polygon = []

def onselect(verts):
    selected_polygon.append(verts)
    plt.close()

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_facecolor("white")

cmap = plt.get_cmap("terrain").copy()
cmap.set_bad(color="white")  # NaNs shown as white

im = ax.pcolormesh(xv, yv, z_grid, cmap=cmap, shading="auto")
fig.colorbar(im, ax=ax, label="Elevation (m)")

z_for_contour = np.ma.masked_invalid(z_grid)
levels = np.linspace(np.nanmin(z_grid), np.nanmax(z_grid), 15)
cs = ax.contour(xv, yv, z_for_contour, levels=levels, colors="k", linewidths=0.5)
ax.clabel(cs, inline=True, fontsize=8)

ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_title(f"Heightmap rotated {THETA_DEG:.1f}° (clockwise)")

selector = PolygonSelector(ax, onselect, useblit=True)
plt.show(block=True)

# ---- Convert selection (defined in rotated XY) back to original points ----
if selected_polygon:
    path = Path(selected_polygon[0])
    inside = path.contains_points(pts_rot[:, :2])  # test on rotated XY
    sel = pts[inside]  # save ORIGINAL (unrotated) XYZ to LAS

    if sel.shape[0] == 0:
        print("No points found inside the selected polygon.")
    else:
        new_las = laspy.LasData(las.header)
        new_las.points = las.points[inside]
        out_path = os.path.join(os.getcwd(), OUT_NAME)
        new_las.write(out_path)
        print(f"Saved {sel.shape[0]} points to: {out_path}")

        # 3D preview (original geometry)
        fig = plt.figure(figsize=(8, 6))
        ax3 = fig.add_subplot(111, projection="3d")
        sc = ax3.scatter(sel[:, 0], sel[:, 1], sel[:, 2], c=sel[:, 2], cmap="terrain", s=1)
        fig.colorbar(sc, ax=ax3, shrink=0.6, label="Elevation (m)")
        ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
        ax3.set_title("3D View of Selected Region")
        plt.tight_layout(); plt.show()
else:
    print("No polygon was selected.")
