import laspy
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# -------------------- Step 1: Load Point Cloud -------------------- #
laz_path = "C:/Users/golzardm/Documents/Dataset-Slope-LiDAR-Embankment-SLidE/Data/2021-06/laz/2021-06.laz"
#laz_path = "C:/Users/hp zbook g5/Documents/GitHub/Dataset-Slope-LiDAR-Embankment-SLidE/Data/2024-02/TerryRoad_Feb2024_GE_ReSampled.laz"

las = laspy.read(laz_path)
xyz = np.vstack((las.x, las.y, las.z)).T

print("Loaded:", xyz.shape[0], "points")
print("Z range:", np.min(xyz[:, 2]), "to", np.max(xyz[:, 2]))

# -------------------- Step 2: Optional Subsampling for Plotting -------------------- #
plot_xyz = xyz
num_plot_points = 15000
if xyz.shape[0] > num_plot_points:
    idx = np.random.choice(len(xyz), num_plot_points, replace=False)
    plot_xyz = xyz[idx]

# -------------------- Step 3: Plot Full Point Cloud -------------------- #
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(plot_xyz[:, 0], plot_xyz[:, 1], plot_xyz[:, 2], s=1, c=plot_xyz[:, 2], cmap='terrain')
ax.set_title("Full 3D Point Cloud (Before Removal)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()

# -------------------- Step 3.5: Interactive 3D Viewer to Get X,Y,Z -------------------- #
import plotly.express as px
import pandas as pd

# Create a DataFrame from the subsampled data
df = pd.DataFrame(plot_xyz, columns=["X", "Y", "Z"])

fig = px.scatter_3d(df, x="X", y="Y", z="Z", color="Z", opacity=0.7, height=800)
fig.update_traces(marker=dict(size=2))
fig.update_layout(
    title="🎯 Hover over any point to get X, Y, Z values (for bounding box definition)",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
    )
)
fig.show()
#%%

# -------------------- Step 4: Define Bounding Box -------------------- #
# ⚠️ CHANGE these values based on the region you want to remove
xmin, xmax = 2501949.3888, 2502030.2765
ymin, ymax = 11728981.5164, 11729105.6866
zmin, zmax = 196.5182, 202.0784

print(f"🚫 Removing all points in box: X=[{xmin}, {xmax}], Y=[{ymin}, {ymax}], Z=[{zmin}, {zmax}]")

# -------------------- Step 5: Apply Bounding Box Filter -------------------- #
mask_inside_box = (
    (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
    (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax) &
    (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
)

xyz_cleaned = xyz[~mask_inside_box]

print(f"✅ Removed {np.sum(mask_inside_box)} points")
print(f"✅ Remaining: {len(xyz_cleaned)} points")

# -------------------- Step 6: Plot Cleaned Cloud -------------------- #
plot_cleaned = xyz_cleaned
if len(plot_cleaned) > num_plot_points:
    idx = np.random.choice(len(plot_cleaned), num_plot_points, replace=False)
    plot_cleaned = plot_cleaned[idx]

fig2 = plt.figure(figsize=(10, 7))
ax2 = fig2.add_subplot(111, projection='3d')
sc2 = ax2.scatter(plot_cleaned[:, 0], plot_cleaned[:, 1], plot_cleaned[:, 2], s=1, c=plot_cleaned[:, 2], cmap='terrain')
ax2.set_title("Cleaned 3D Point Cloud (After Removal)")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
plt.tight_layout()
plt.show()

# -------------------- Step 7: Save Cleaned LAS File -------------------- #
output_path = "cleaned_slope_box.las"

las_cleaned = laspy.create(point_format=las.header.point_format, file_version=las.header.version)

scales = las.header.scales
offsets = las.header.offsets

las_cleaned.X = ((xyz_cleaned[:, 0] - offsets[0]) / scales[0]).astype(np.int32)
las_cleaned.Y = ((xyz_cleaned[:, 1] - offsets[1]) / scales[1]).astype(np.int32)
las_cleaned.Z = ((xyz_cleaned[:, 2] - offsets[2]) / scales[2]).astype(np.int32)

las_cleaned.header.scales = scales
las_cleaned.header.offsets = offsets

las_cleaned.write(output_path)
print("✅ Cleaned LAS file saved to:")
print(os.path.abspath(output_path))
