
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import laspy
from scipy.ndimage import gaussian_filter

# grid creating
x = np.linspace(0, 40, 60)  # length
y = np.linspace(0, 50, 60)  # width
x, y = np.meshgrid(x, y)

# creating curved slope
a, b, c = 0.1, 0.03, -0.002  # slope + mild curvature
z_base = a * x + b * y + c * (x**2 + y**2)

# adding abnormalities   magnitude, location, diameter
z = z_base \
    + 1.5 * np.exp(-((x - 25)**2 + (y - 30)**2) / 9) \
    + 1.2 * np.exp(-((x - 30)**2 + (y - 10)**2) / 2) \
    - 1.8 * np.exp(-((x - 15)**2 + (y - 10)**2) / 3) \
    - 1.0 * np.exp(-((x - 20)**2 + (y - 40)**2) / 2)

# adding roughness using gaussian-filtered noise
np.random.seed(42)    # random noise pattern generator(e.g., 42 random number)
noise = np.random.randn(*z.shape)
roughness = gaussian_filter(noise, sigma=2)  # sigma control smoothness of bumps (e.g., Lower = sharper bumps & Higher = smoother hills)
z += 0.4 * roughness                         # Control amplitude of roughness (e.g., Lower = almost flat & Higher = dramatic unevenness)

# slope is saved as numPy array
X = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
np.save("synthetic_curved_surface.npy", X)

# Plot: mesh surface 
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='terrain', edgecolor='k', alpha=0.6)
ax.set_title("Synthetic Embankment-Like Surface with Abnormalities and Noise")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()

# Plot: point cloud visualization 
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 2], cmap='viridis', s=4)
ax.set_title("Point Cloud View of Synthetic Surface with Noise")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()

# saving as .las point cloud 
header = laspy.LasHeader(point_format=3, version="1.2")
header.x_scale = header.y_scale = header.z_scale = 0.001
header.x_offset = header.y_offset = header.z_offset = 0.0

las = laspy.LasData(header)
las.x = X[:, 0]
las.y = X[:, 1]
las.z = X[:, 2]
las.write("synthetic_surface.las")

print(" Synthetic noisy surface saved as both .npy and .las files.")



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

#  point cloud from .las file
las = laspy.read("C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/synthetic_surface.las")

X = np.vstack((las.x, las.y, las.z)).T  # Shape: (N, 3)

# Extracting X, Y, Z
xy = X[:, :2]
z_true = X[:, 2]

#  fit a smooth surface using polynomial regression (RANSAC for robustness)
degree = 2
poly = PolynomialFeatures(degree=degree)
model = make_pipeline(poly, RANSACRegressor(LinearRegression(), residual_threshold=0.1))
model.fit(xy, z_true)

#  Predict smooth surface and compute residuals
z_pred = model.predict(xy)
residuals = z_true - z_pred

#  threshold adjustment the residuals to detect abnormalities
threshold = 0.35  # Sensitivity threshold  (Lower=More sensitive → detects tiny bumps, even noise might be flagged & Higher=Less sensitive → detects only larger anomalies like humps or cavities)
abnormal_indices = np.where(np.abs(residuals) > threshold)[0]
abnormal_points = X[abnormal_indices]

#  apply DBSCAN clustering on abnormal regions
db = DBSCAN(eps=1.0, min_samples=5).fit(abnormal_points[:, :2])
labels = db.labels_
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Plot detected abnormalities with residual magnitude
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
scatter = ax.scatter(abnormal_points[:, 0], abnormal_points[:, 1], abnormal_points[:, 2],
                     c=np.abs(residuals[abnormal_indices]), cmap='viridis', s=10)
cb = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
cb.set_label("Residual Magnitude")
ax.set_title(f"Detected Abnormalities via Residuals + DBSCAN Clustering ({num_clusters} clusters)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()

# saving abnormalities to LAS file
header = laspy.LasHeader(point_format=3, version="1.2")
header.x_scale = header.y_scale = header.z_scale = 0.001
header.x_offset = header.y_offset = header.z_offset = 0.0

las_out = laspy.LasData(header)
las_out.x = abnormal_points[:, 0]
las_out.y = abnormal_points[:, 1]
las_out.z = abnormal_points[:, 2]
las_out.write("abnormalities_ransac_dbscan.las")
print(" Abnormalities saved to 'abnormalities_ransac_dbscan.las'")

# Print evaluation
r2 = r2_score(z_true, z_pred)
std_residual = np.std(residuals)

print("\nPolynomial Fit Evaluation:")
print(f"Degree: {degree}")
print(f"R² Score: {r2:.5f}")
print(f"Residual Std: {std_residual:.5f}")

