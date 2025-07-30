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
iso = IsolationForest(contamination=0.03, random_state=0)
labels_iforest = iso.fit_predict(residuals.reshape(-1, 1))
abnormal_idx = np.where(labels_iforest == -1)[0]
abnormal_pts = X_subset[abnormal_idx]

# --- DBSCAN ---
db = DBSCAN(eps=1.2, min_samples=2).fit(abnormal_pts[:, :2])
labels_dbscan = db.labels_
num_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
print(f"Detected Clusters: {num_clusters}")

# --- Plot Abnormal Clusters with Coordinates ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(abnormal_pts[:, 0], abnormal_pts[:, 1], abnormal_pts[:, 2],
                     c=labels_dbscan, cmap='viridis', s=8)

ax.set_title("MLS + Isolation Forest + DBSCAN Clusters")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()

# Hover tooltip with coordinates (use Spyder console for exact coordinates)
def on_move(event):
    if event.inaxes:
        print(f"Mouse at: X={event.xdata:.2f}, Y={event.ydata:.2f}")

fig.canvas.mpl_connect('motion_notify_event', on_move)
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import laspy
from pathlib import Path

# --- Define Bounding Boxes (from manual inspection) ---
bounding_boxes = [
    # Format: (xmin, xmax, ymin, ymax)
    (711410, 711420, 308240, 308260),   # Cavity 1 (example)
    # Add more if needed
]

# --- Filter Points Inside Bounding Boxes ---
final_points = []
for (xmin, xmax, ymin, ymax) in bounding_boxes:
    mask = ((abnormal_pts[:, 0] >= xmin) & (abnormal_pts[:, 0] <= xmax) &
            (abnormal_pts[:, 1] >= ymin) & (abnormal_pts[:, 1] <= ymax))
    selected = abnormal_pts[mask]
    final_points.append(selected)

final_selected = np.vstack(final_points)
print(f"Total selected points: {len(final_selected)}")

# --- Optional 3D Plot of Selected Clusters ---
plt.ion()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(final_selected[:, 0], final_selected[:, 1], final_selected[:, 2],
           cmap='viridis', s=10)
ax.set_title("Selected Cavity Points (After Bounding Box Filter)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
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

las_out.write("selected_clusters_filtered.las")
print("✅ Filtered region saved to 'selected_clusters_filtered.las'")
