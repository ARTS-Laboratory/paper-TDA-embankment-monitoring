# Section 1: Generate and Visualize Mildly Curved Embankment-Like Surface with Abnormalities

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 

# Create grid for a gently curved embankment surface
x = np.linspace(0, 40, 80)  # length
y = np.linspace(0, 50, 60)  # width
x, y = np.meshgrid(x, y)

# Gently sloped surface with slight curvature (like windshield)
a, b, c = 0.1, 0.03, -0.002  # slope + mild curvature
z_base = a * x + b * y + c * (x**2 + y**2)

# Add a hump and a cavity
z = z_base + \
    2.5 * np.exp(-((x - 25)**2 + (y - 30)**2) / 3) - \
    1.3 * np.exp(-((x - 15)**2 + (y - 10)**2) / 6)
    
X = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
np.save("synthetic_curved_surface.npy", X)  # Save for next section

# Plot the surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='terrain', edgecolor='k', alpha=0.6)
ax.set_title("Synthetic Embankment-Like Surface with Hump and Cavity")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()

#%% step 2

from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import DBSCAN
from sklearn.pipeline import make_pipeline
import pandas as pd



# Load the synthetic surface
X = np.load("C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/synthetic_curved_surface.npy")

x, y, z = X[:, 0], X[:, 1], X[:, 2]

# Step 1: Fit a smooth surface using polynomial regression (RANSAC for robustness)
poly = PolynomialFeatures(degree=2)
model = make_pipeline(poly, RANSACRegressor(residual_threshold=0.1))
model.fit(X[:, :2], z)

# Step 2: Predict the smooth surface and calculate residuals
z_pred = model.predict(X[:, :2])
residuals = z - z_pred

# Step 3: Threshold the residuals to detect abnormalities
threshold = 0.25  # Sensitivity threshold
abnormal_indices = np.where(np.abs(residuals) > threshold)[0]
abnormal_points = X[abnormal_indices]

# Step 4: Apply clustering (DBSCAN) to the abnormal points to isolate connected regions
db = DBSCAN(eps=1.0, min_samples=5).fit(abnormal_points[:, :2])
labels = db.labels_

# Show how many clusters were found
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Plot detected abnormalities with cluster labels
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(abnormal_points[:, 0], abnormal_points[:, 1], abnormal_points[:, 2], 
                     c=labels, cmap='tab10', s=6)
ax.set_title(f"Detected Abnormalities via Residuals + DBSCAN Clustering ({num_clusters} clusters)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()
