import numpy as np
import laspy
from scipy.spatial import cKDTree
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Load Point Cloud ---
las = laspy.read("C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/road_removed_with_abnormality.las")
X = np.vstack((las.x, las.y, las.z)).T

# --- Subsample for faster processing ---
num_points = 15000
indices = np.random.choice(len(X), num_points, replace=False)
X_subset = X[indices]
xy = X_subset[:, :2]
z = X_subset[:, 2]

# --- MLS Surface Fit ---
def mls_fit(xy, z, k=30):
    tree = cKDTree(xy)
    z_fit = np.zeros_like(z)
    for i, pt in enumerate(xy):
        _, idx = tree.query(pt, k=k)
        A = np.c_[np.ones(k), xy[idx][:, 0], xy[idx][:, 1]]
        coeffs, *_ = np.linalg.lstsq(A, z[idx], rcond=None)
        z_fit[i] = np.dot([1, pt[0], pt[1]], coeffs)
    return z_fit

z_fit = mls_fit(xy, z)
residuals = z - z_fit

# --- Isolation Forest ---
iso = IsolationForest(contamination=0.1, random_state=0)
labels_iforest = iso.fit_predict(residuals.reshape(-1, 1))
abnormal_idx = np.where(labels_iforest == -1)[0]
abnormal_pts = X_subset[abnormal_idx]

# --- DBSCAN ---
db = DBSCAN(eps=1.2, min_samples=2).fit(abnormal_pts[:, :2])
labels_dbscan = db.labels_
num_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
print(f"Detected Clusters: {num_clusters}")

# --- Plot Abnormal Clusters Colored by Height (Z) ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    abnormal_pts[:, 0], 
    abnormal_pts[:, 1], 
    abnormal_pts[:, 2],
    c=abnormal_pts[:, 2],     # Color by Z height
    cmap='viridis',
    s=10
)

cb = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cb.set_label("Z Height (m)")

ax.set_title("MLS + Isolation Forest + DBSCAN Clusters (Colored by Z Height)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_xlim(abnormal_pts[:, 0].min()-1, abnormal_pts[:, 0].max()+1)
ax.set_ylim(abnormal_pts[:, 1].min()-1, abnormal_pts[:, 1].max()+1)
ax.set_zlim(abnormal_pts[:, 2].min()-1, abnormal_pts[:, 2].max()+1)

plt.tight_layout()
plt.show()
#%% Section 2: Filter and Save Selected Cavity Regions

import numpy as np
import matplotlib.pyplot as plt
import laspy
from pathlib import Path

# --- Define Bounding Boxes (X, Y, Z bounds) ---
bounding_boxes = [
    # Format: (xmin, xmax, ymin, ymax, zmin, zmax)
    (711402, 711425, 308239, 308260, 91.23, 95.7),  # Example: middle cavity
    # Add more if needed:
    # (x1, x2, y1, y2, z1, z2),
]

# --- Filter Points Inside Any Bounding Box ---
final_points = []
for (xmin, xmax, ymin, ymax, zmin, zmax) in bounding_boxes:
    mask = (
        (abnormal_pts[:, 0] >= xmin) & (abnormal_pts[:, 0] <= xmax) &
        (abnormal_pts[:, 1] >= ymin) & (abnormal_pts[:, 1] <= ymax) &
        (abnormal_pts[:, 2] >= zmin) & (abnormal_pts[:, 2] <= zmax)
    )
    final_points.append(abnormal_pts[mask])

# Stack all selected clusters
final_selected = np.vstack(final_points)

# --- Plot: Viridis Colormap by Z (Height) ---
plt.ion()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(final_selected[:, 0], final_selected[:, 1], final_selected[:, 2],
                c=final_selected[:, 2], cmap='viridis', s=20)

cb = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
cb.set_label("Z Height (m)")

ax.set_title("Filtered Cavity Region (Colored by Height)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim(final_selected[:, 0].min() - 1, final_selected[:, 0].max() + 1)
ax.set_ylim(final_selected[:, 1].min() - 1, final_selected[:, 1].max() + 1)
ax.set_zlim(final_selected[:, 2].min() - 1, final_selected[:, 2].max() + 1)
plt.tight_layout()
plt.show()

# --- Save to LAS ---
min_vals = final_selected.min(axis=0)
scale = 0.001
offset = min_vals - 1000

header = laspy.LasHeader(point_format=3, version="1.2")
header.x_scale = header.y_scale = header.z_scale = scale
header.x_offset = offset[0]
header.y_offset = offset[1]
header.z_offset = offset[2]

las_out = laspy.LasData(header)
las_out.x = final_selected[:, 0]
las_out.y = final_selected[:, 1]
las_out.z = final_selected[:, 2]

output_path = Path("selected_clusters_filtered.las")
las_out.write(str(output_path))

print(f"\nFiltered cavity saved as: {output_path}")
print(f" Number of points saved: {len(final_selected)}")
