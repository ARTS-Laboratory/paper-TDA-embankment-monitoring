import laspy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import Button
import os
import sys
import plotly.graph_objects as go


# -------------------- Step 1: Load Point Cloud -------------------- #
laz_path = "C:/Users/golzardm/Documents/Dataset-Slope-LiDAR-Embankment-SLidE/Data/2024-02/TerryRoad_Feb2024_GE_ReSampled.laz"
#laz_path = "C:/Users/hp zbook g5/Documents/GitHub/Dataset-Slope-LiDAR-Embankment-SLidE/Data/2024-02/TerryRoad_Feb2024_GE_ReSampled.laz"

las = laspy.read(laz_path)
xyz = np.vstack((las.x, las.y, las.z)).T

print("Loaded:", xyz.shape[0], "points")
print("Z range:", np.min(xyz[:, 2]), "to", np.max(xyz[:, 2]))

# -------------------- Step 2: Subsample -------------------- #
num_points = 10000
if xyz.shape[0] > num_points:
    idx = np.random.choice(len(xyz), num_points, replace=False)
    xyz = xyz[idx]

# -------------------- Step 3: Z-Threshold Filtering -------------------- #
z_threshold = 203
xyz = xyz[xyz[:, 2] > z_threshold]

# -------------------- Step 4: Interactive Polygon Selection -------------------- #
clicked_points = []
plot_closed = False

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        clicked_points.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, 'ro')
        fig.canvas.draw()

def ondone(event):
    global plot_closed
    plot_closed = True
    plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(xyz[:, 0], xyz[:, 1], c=xyz[:, 2], cmap='terrain', s=2)
ax.set_title("Click to Select Region to Remove (Then Click DONE)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.colorbar(sc, label='Z Elevation')
plt.axis("equal")

# Add DONE button
ax_done = plt.axes([0.8, 0.01, 0.1, 0.04])
btn_done = Button(ax_done, 'DONE')
btn_done.on_clicked(ondone)

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)  # This ensures it waits for you to close the window

# -------------------- Step 5: Polygon Cleanup -------------------- #
if not plot_closed or len(clicked_points) < 3:
    print(" You must click at least 3 points and press DONE.")
    sys.exit()

clicked_points.append(clicked_points[0])  # close polygon
polygon = np.array(clicked_points)
poly_path = Path(polygon)

xy = xyz[:, :2]
mask_inside = poly_path.contains_points(xy)
xyz_cleaned = xyz[~mask_inside]

print(f" Removed {np.sum(mask_inside)} points")
print(f" Remaining: {len(xyz_cleaned)} points")

# -------------------- Step 6: Show Cleaned Plot -------------------- #
if len(xyz_cleaned) == 0:
    print(" Cleaned point cloud is empty — not plotting or saving.")
    sys.exit()

fig2, ax2 = plt.subplots(figsize=(10, 6))
sc2 = ax2.scatter(xyz_cleaned[:, 0], xyz_cleaned[:, 1], c=xyz_cleaned[:, 2], cmap='terrain', s=2)
ax2.set_title(" Cleaned Point Cloud (Region Removed)")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_aspect('equal')
plt.colorbar(sc2, label="Z Elevation")
plt.tight_layout()
plt.show(block=True)

# -------------------- Step 7: Save Cleaned LAS File -------------------- #
output_path = "cleaned_slope.las"

#  Create new LAS with same format/version
las_cleaned = laspy.create(point_format=las.header.point_format, file_version=las.header.version)

# Use same scales and offsets
scales = las.header.scales
offsets = las.header.offsets

# Convert float XYZ to scaled integers
las_cleaned.X = ((xyz_cleaned[:, 0] - offsets[0]) / scales[0]).astype(np.int32)
las_cleaned.Y = ((xyz_cleaned[:, 1] - offsets[1]) / scales[1]).astype(np.int32)
las_cleaned.Z = ((xyz_cleaned[:, 2] - offsets[2]) / scales[2]).astype(np.int32)

# Set header scale/offset explicitly
las_cleaned.header.scales = scales
las_cleaned.header.offsets = offsets

# Save LAS file
las_cleaned.write(output_path)
print(" LAS file saved to:")
print(os.path.abspath(output_path))

