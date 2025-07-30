#%% RANSAC AND DBSCAN 

from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import DBSCAN
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import laspy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load point cloud from .las file
las = laspy.read("C:/Users/golzardm/Documents/Dataset-Slope-LiDAR-Embankment-SLidE/Data/2021-06/laz/2021-06.laz")
X = np.vstack((las.x, las.y, las.z)).T  # shape: (N, 3)

print(f"Total number of points: {len(X)}")

#%% Optionally downsample (keep your subset if needed)
num_points = 15000
if len(X) > num_points:
    indices = np.random.choice(len(X), num_points, replace=False)
    X_subset = X[indices]
else:
    X_subset = X

#%% RANSAC surface fitting
xy = X[:, :2]
z_true = X[:, 2]

degree = 2
poly = PolynomialFeatures(degree=degree)
temp_threshold = 1.0
model = make_pipeline(poly, RANSACRegressor(LinearRegression(), residual_threshold=temp_threshold))
model.fit(xy, z_true)

# Residuals
z_pred = model.predict(xy)
residuals = z_true - z_pred
std_residual = np.std(residuals)

# Thresholds
residual_threshold = 1.5 * std_residual
threshold = 2.5 * std_residual
print(f"Adaptive residual_threshold = {residual_threshold:.3f}")
print(f"Adaptive threshold = {threshold:.3f}")

# Refit with adjusted residual threshold
model = make_pipeline(poly, RANSACRegressor(LinearRegression(), residual_threshold=residual_threshold))
model.fit(xy, z_true)
z_pred = model.predict(xy)
residuals = z_true - z_pred

# Abnormality detection
abnormal_indices = np.where(np.abs(residuals) > threshold)[0]
abnormal_points = X[abnormal_indices]

# DBSCAN clustering
db = DBSCAN(eps=1.2, min_samples=2).fit(abnormal_points[:, :2])
labels = db.labels_
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Detected clusters: {num_clusters}")

# Keep only clustered points (exclude noise, i.e., label -1)
cluster_mask = labels != -1
clustered_points = abnormal_points[cluster_mask]
cluster_labels = labels[cluster_mask]

#%% 3D Plot of clustered points (Spyder interactive)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(clustered_points[:, 0], clustered_points[:, 1], clustered_points[:, 2],
                     c=cluster_labels, cmap='tab10', s=1)

ax.set_title(f"DBSCAN Clusters of Abnormal Regions ({num_clusters} clusters)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.tight_layout()
plt.show()

#%% Save clustered points to LAS file
min_vals = clustered_points.min(axis=0)
scale = 0.001
offset = min_vals - 1000

header = laspy.LasHeader(point_format=3, version="1.2")
header.x_scale = header.y_scale = header.z_scale = scale
header.x_offset = offset[0]
header.y_offset = offset[1]
header.z_offset = offset[2]

las_out = laspy.LasData(header)
las_out.x = clustered_points[:, 0]
las_out.y = clustered_points[:, 1]
las_out.z = clustered_points[:, 2]

output_path = Path("clustered_abnormalities.las")
las_out.write(str(output_path))
print(f" Clustered abnormalities saved to: '{output_path}'")

#%% Evaluation
r2 = r2_score(z_true, z_pred)
print("\nPolynomial Fit Evaluation:")
print(f"Degree: {degree}")
print(f"R² Score: {r2:.5f}")
print(f"Residual Std: {std_residual:.5f}")
