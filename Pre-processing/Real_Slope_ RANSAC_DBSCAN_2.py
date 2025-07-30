#%% RANSAC AND DBSCAN 

from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import DBSCAN
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pandas as pd
import laspy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#  point cloud from .las file
las = laspy.read("C:/Users/golzardm/Documents/Dataset-Slope-LiDAR-Embankment-SLidE/Data/2021-06/laz/2021-06.laz")
X = np.vstack((las.x, las.y, las.z)).T  # shape: (N, 3)

# Print the total number of points
print(f"Total number of points: {len(X)}")
#%%
# Randomly select a subset of points (set for num_points a value)
num_points = 15000
if len(X) > num_points:
    indices = np.random.choice(len(X), num_points, replace=False)
    X_subset = X[indices]
else:
    X_subset = X

#%%
# Extracting X, Y, Z
xy = X[:, :2]
z_true = X[:, 2]

# Step 1: Fit initial model with temp residual_threshold
degree = 2
poly = PolynomialFeatures(degree=degree)
temp_threshold = 1.0  # Safe starting point
model = make_pipeline(poly, RANSACRegressor(LinearRegression(), residual_threshold=temp_threshold))
model.fit(xy, z_true)

# Step 2: Predict and compute residuals
z_pred = model.predict(xy)
residuals = z_true - z_pred
std_residual = np.std(residuals)

# Step 3: Use standard deviation to tune thresholds
residual_threshold = 1.5 * std_residual
threshold = 2.5 * std_residual
print(f"Adaptive residual_threshold = {residual_threshold:.3f}")
print(f"Adaptive threshold for abnormality = {threshold:.3f}")

# Step 4: Refit the model with adaptive residual_threshold
model = make_pipeline(poly, RANSACRegressor(LinearRegression(), residual_threshold=residual_threshold))
model.fit(xy, z_true)

# Step 5: Recompute predictions and residuals
z_pred = model.predict(xy)
residuals = z_true - z_pred

# Step 6: Abnormality detection
abnormal_indices = np.where(np.abs(residuals) > threshold)[0]
abnormal_points = X[abnormal_indices]

# Step 7: Apply DBSCAN
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=1.2, min_samples=2).fit(abnormal_points[:, :2])
labels = db.labels_
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Step 8: Visualization
import matplotlib.pyplot as plt
plt.rcParams.update({
    'image.cmap': 'viridis',
    'font.serif': [
        'Times New Roman', 'Times', 'DejaVu Serif', 'Bitstream Vera Serif',
        'Computer Modern Roman', 'New Century Schoolbook', 'Century Schoolbook L',
        'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Palatino',
        'Charter', 'serif'
    ],
    'font.family': 'serif',
    'font.size': 14,
})

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(abnormal_points[::2, 0], abnormal_points[::2, 1], abnormal_points[::2, 2],
                     c=np.abs(residuals[abnormal_indices])[::2], cmap='terrain', s=0.3)

cb = plt.colorbar(scatter, ax=ax, shrink=0.2, pad=0.1)

cb.set_label("Residual Magnitude")
ax.set_title(f"Detected Abnormalities via Residuals + DBSCAN Clustering ({num_clusters} clusters)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()

# Step 9: Save abnormalities to LAS
from pathlib import Path
import laspy

min_vals = abnormal_points.min(axis=0)
scale = 0.001
offset = min_vals - 1000

header = laspy.LasHeader(point_format=3, version="1.2")
header.x_scale = header.y_scale = header.z_scale = scale
header.x_offset = offset[0]
header.y_offset = offset[1]
header.z_offset = offset[2]

las_out = laspy.LasData(header)
las_out.x = abnormal_points[:, 0]
las_out.y = abnormal_points[:, 1]
las_out.z = abnormal_points[:, 2]

output_path = Path("abnormalities_ransac_dbscan.las")
las_out.write(str(output_path))
print(f"Abnormalities saved to '{output_path}'")

# Step 10: Final evaluation
from sklearn.metrics import r2_score

r2 = r2_score(z_true, z_pred)
print("\nPolynomial Fit Evaluation:")
print(f"Degree: {degree}")
print(f"R² Score: {r2:.5f}")
print(f"Residual Std: {std_residual:.5f}")



