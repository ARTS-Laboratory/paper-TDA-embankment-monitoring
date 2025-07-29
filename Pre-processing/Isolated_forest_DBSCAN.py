#%% MLS + Isolation Forest + DBSCAN with Dominant Cavity Filtering

import numpy as np
import laspy
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score
import pandas as pd

# Load .las file
las = laspy.read("C:/Users/hp zbook g5/Documents/GitHub/paper-TDA-embankment-monitoring/Pre-processing/road_removed_with_abnormality.las")
X = np.vstack((las.x, las.y, las.z)).T
print(f"Total points: {len(X)}")

# Subsampling
num_points = 15000
indices = np.random.choice(len(X), num_points, replace=False)
X_subset = X[indices]
xy = X_subset[:, :2]
z = X_subset[:, 2]

# Step 1: MLS surface fitting
def mls_surface_fit(xy, z, k=30):
    tree = cKDTree(xy)
    z_fitted = np.zeros_like(z)
    for i, pt in enumerate(xy):
        _, idx = tree.query(pt, k=k)
        neighbors_xy = xy[idx]
        neighbors_z = z[idx]
        A = np.c_[np.ones(k), neighbors_xy[:, 0], neighbors_xy[:, 1]]
        coeffs, *_ = np.linalg.lstsq(A, neighbors_z, rcond=None)
        z_fitted[i] = np.dot([1, pt[0], pt[1]], coeffs)
    return z_fitted

z_fit = mls_surface_fit(xy, z, k=30)
residuals = z - z_fit

# Step 2: Isolation Forest on residuals
residuals_reshaped = residuals.reshape(-1, 1)
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(residuals_reshaped)
outlier_mask = iso.predict(residuals_reshaped) == -1
outlier_pts = X_subset[outlier_mask]

# Step 3: DBSCAN on detected outliers
db = DBSCAN(eps=2, min_samples=2).fit(outlier_pts[:, :2])
labels = db.labels_
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Clusters found: {num_clusters}")

# Step 4: Keep only dominant cavity (largest cluster)
if num_clusters > 0:
    largest_label = pd.Series(labels[labels != -1]).value_counts().idxmax()
    final_mask = labels == largest_label
    final_pts = outlier_pts[final_mask]
    final_labels = labels[final_mask]
else:
    final_pts = outlier_pts
    final_labels = labels

# Step 5: Visualization in Spyder
plt.ion()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(final_pts[:, 0], final_pts[:, 1], final_pts[:, 2],
           c=final_labels, cmap='tab20', s=3)
ax.set_title("MLS + Isolation Forest + DBSCAN (Dominant Cavity)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()

# Step 6: Save to LAS
min_vals = final_pts.min(axis=0)
scale = 0.001
offset = min_vals - 1000

header = laspy.LasHeader(point_format=3, version="1.2")
header.x_scale = header.y_scale = header.z_scale = scale
header.x_offset = offset[0]
header.y_offset = offset[1]
header.z_offset = offset[2]

las_out = laspy.LasData(header)
las_out.x = final_pts[:, 0]
las_out.y = final_pts[:, 1]
las_out.z = final_pts[:, 2]

output_path = Path("dominant_cavity_mls_iforest_dbscan.las")
las_out.write(str(output_path))
print(f"Saved clustered cavity to: '{output_path}'")

# Step 7: Evaluation
z_true = z
z_pred = z_fit
r2 = r2_score(z_true, z_pred)
print("\nMLS Surface Fit Evaluation:")
print(f"R² Score: {r2:.5f}")
print(f"Residual Std: {np.std(residuals):.5f}")
