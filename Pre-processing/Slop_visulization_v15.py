#%% MLS + DBSCAN for Slope Clustering

import numpy as np
import pandas as pd
import laspy
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score

# Load point cloud
las = laspy.read("C:/Users/hp zbook g5/Documents/GitHub/Dataset-Slope-LiDAR-Embankment-SLidE/Data/2021-06/laz/2021-06.laz")
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

z_fit = mls_surface_fit(xy, z)
residuals = z - z_fit
std_res = np.std(residuals)
threshold = 3 * std_res
abnormal_idx = np.where(np.abs(residuals) > threshold)[0]
abnormal_pts = X_subset[abnormal_idx]

# Step 2: DBSCAN clustering
db = DBSCAN(eps=1.2, min_samples=2).fit(abnormal_pts[:, :2])
labels = db.labels_
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Step 3: Plot in Spyder (interactive)
plt.ion()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(abnormal_pts[:, 0], abnormal_pts[:, 1], abnormal_pts[:, 2],
           c=labels, cmap='tab20', s=3)
ax.set_title(f"MLS + DBSCAN ({num_clusters} Clusters)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()

# Step 4: Save .LAS
min_vals = abnormal_pts.min(axis=0)
scale = 0.001
offset = min_vals - 1000
header = laspy.LasHeader(point_format=3, version="1.2")
header.x_scale = header.y_scale = header.z_scale = scale
header.x_offset = offset[0]
header.y_offset = offset[1]
header.z_offset = offset[2]

las_out = laspy.LasData(header)
las_out.x = abnormal_pts[:, 0]
las_out.y = abnormal_pts[:, 1]
las_out.z = abnormal_pts[:, 2]
las_out.write("abnormalities_mls_dbscan.las")
print("Saved as 'abnormalities_mls_dbscan.las'")

# Reconstruct z_true and z_pred from the subset
z_true = z
z_pred = z_fit
r2 = r2_score(z_true, z_pred)

print("\nMLS Surface Fit Evaluation:")
print(f"R² Score: {r2:.5f}")
print(f"Residual Std: {std_res:.5f}")