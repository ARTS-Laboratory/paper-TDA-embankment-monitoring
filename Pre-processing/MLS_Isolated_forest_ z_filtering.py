import numpy as np
import laspy
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# --- Load Point Cloud ---
las = laspy.read("C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/clean_slope_abnormality_added.las")
X = np.vstack((las.x, las.y, las.z)).T
print(f"Total points: {len(X)}")

# --- Subsample for Speed ---
num_points = 150000
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
residual_std = np.std(residuals)

# --- Filter Only Points with Large Negative Residuals (Below Surface) ---
threshold = 2.48 * residual_std
neg_idx = np.where(residuals < -threshold)[0]
X_neg = X_subset[neg_idx]
xy_neg = X_neg[:, :2]

print(f"Points with residual < -{threshold:.3f}: {len(X_neg)}")

# --- DBSCAN on Negative Residual Points ---
db = DBSCAN(eps=1.0, min_samples=5).fit(xy_neg)
labels = db.labels_

# --- Find Largest Cluster (most likely the cavity) ---
unique_labels = set(labels) - {-1}
largest_cluster = None
max_count = 0

for label in unique_labels:
    cluster_idx = np.where(labels == label)[0]
    if len(cluster_idx) > max_count:
        max_count = len(cluster_idx)
        largest_cluster = cluster_idx

# --- Extract Final Points ---
if largest_cluster is not None:
    final_cavity = X_neg[largest_cluster]
    print(f"✅ Final cavity points: {len(final_cavity)}")
else:
    final_cavity = np.empty((0, 3))
    print("⚠️ No cavity found.")

# --- Plot ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

if len(final_cavity) > 0:
    sc = ax.scatter(
        final_cavity[:, 0],
        final_cavity[:, 1],
        final_cavity[:, 2],
        c=final_cavity[:, 2],
        cmap='viridis',
        s=10
    )
    cb = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label("Z Height (m)")
else:
    ax.text2D(0.3, 0.5, "No cavity found", transform=ax.transAxes)

ax.set_title("Only Deepest Cavity (Filtered + Clustered)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.grid(False)
plt.tight_layout()
plt.show()

# --- Optional Save ---
if len(final_cavity) > 0:
    min_vals = final_cavity.min(axis=0)
    scale = 0.001
    offset = min_vals - 1000

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.x_scale = header.y_scale = header.z_scale = scale
    header.x_offset = offset[0]
    header.y_offset = offset[1]
    header.z_offset = offset[2]

    las_out = laspy.LasData(header)
    las_out.x = final_cavity[:, 0]
    las_out.y = final_cavity[:, 1]
    las_out.z = final_cavity[:, 2]

    output_path = Path("final_cavity_only.las")
    las_out.write(str(output_path))
    print(f"\n💾 Saved final cavity to: {output_path}")
else:
    print("⚠️ Nothing to save.")
