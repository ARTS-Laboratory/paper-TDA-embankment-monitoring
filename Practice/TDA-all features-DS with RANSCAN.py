import laspy
import numpy as np
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import NumberOfPoints, Amplitude, PersistenceEntropy
from gtda.plotting import plot_diagram
from sklearn.linear_model import RANSACRegressor

class tda:
    def __init__(self, homo_dim=1, fts='entropy') -> None:
        """
        homo_dim: homology dimensions, integer value.
        fts: features to return, options: 'entropy', 'numofpoints', 'amp'
        """
        self.homology_dimensions = list(range(homo_dim + 1))
        print("Homology dimensions:", self.homology_dimensions)
        self.persistence = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=self.homology_dimensions,
            n_jobs=-1)
        self.fts = fts
        if fts == 'entropy':
            from gtda.diagrams import PersistenceEntropy
            self.persistence_entropy = PersistenceEntropy()
        elif fts == 'numofpoints':
            from gtda.diagrams import NumberOfPoints
            self.NumOfPoint = NumberOfPoints()
        elif fts == 'amp':
            self.metrics = ["bottleneck", "wasserstein", "landscape", "persistence_image", "betti", "heat"]

    def ransac_tda(self, pcd, residual_threshold=1.0, max_trials=1000):
        """
        Downsamples/filters the point cloud using RANSAC to fit a plane model.
        pcd: numpy array of shape (N, 3)
        Returns the filtered point cloud (inliers).
        """
        print("Applying RANSAC to filter point cloud...")
        ransac = RANSACRegressor(residual_threshold=residual_threshold, max_trials=max_trials)
        ransac.fit(pcd[:, :2], pcd[:, 2])  # Fit z = f(x, y)
        inlier_mask = ransac.inlier_mask_
        filtered_pcd = pcd[inlier_mask]
        print(f"RANSAC filtered point cloud shape: {filtered_pcd.shape}")
        return filtered_pcd

    def forward(self, pcd):
        """
        pcd: list of point cloud arrays. For a single point cloud, pass [pcd].
        Computes persistence diagrams and extracts TDA features.
        """
        print("Computing persistence diagrams on point cloud(s)...")
        self.diag = self.persistence.fit_transform(pcd)
        print("Persistence diagrams computed.")
        
        # Plot the persistence diagram only for the 'entropy' extractor
        if self.fts == 'entropy':
            print("Plotting the persistence diagram...")
            plot_diagram(self.diag[0])
            plt.title("Persistence Diagram")
            plt.xlim(0, 10)
            plt.ylim(0, 10)
            plt.show()
        
        # Features extraction
        if self.fts == 'entropy':
            features = self.persistence_entropy.fit_transform(self.diag)
            print("Extracted persistence entropy features.")
            return features
        elif self.fts == 'numofpoints':
            features = self.NumOfPoint.fit_transform(self.diag)
            print("Extracted number-of-points features.")
            return features
        elif self.fts == 'amp':
            rslt = []
            from gtda.diagrams import Amplitude
            for m in self.metrics:
                AMP = Amplitude(metric=m)
                amp = AMP.fit_transform(self.diag)
                rslt.append(amp)
            features = np.hstack(rslt) if rslt else np.array([])
            print("Extracted amplitude features.")
            return features

    def save_homology_dimensions(self, diagram_index=0, filename_prefix="homology"):
        """
        Saves the H0 and H1 persistence diagram data as CSV files.
        """
        diag = self.diag[diagram_index]
        H0 = diag[diag[:, 2] == 0]
        H1 = diag[diag[:, 2] == 1]
        np.savetxt(f"{filename_prefix}_H0.csv", H0, delimiter=",", header="birth,death,dimension", comments='')
        np.savetxt(f"{filename_prefix}_H1.csv", H1, delimiter=",", header="birth,death,dimension", comments='')
        print(f"Saved H0 persistence diagram to {filename_prefix}_H0.csv")
        print(f"Saved H1 persistence diagram to {filename_prefix}_H1.csv")

    def __call__(self, pcd):
        return self.forward(pcd)

###########################################
# Process the point cloud from a LAS file and extract TDA features

import laspy

# Set the file path to your LAS file
file_path = "C:/Users/GOLZARDM/.spyder-py3/surface_with_smooth_circular_cavity_20.las"
print("Opening LAS file:", file_path)
with laspy.open(file_path) as f:
    las = f.read()
print("LAS file read successfully.")

x = las.x
y = las.y
z = las.z
point_cloud = np.vstack((x, y, z)).T
print("Point cloud shape:", point_cloud.shape)

# Downsample to 10% of the data using random sampling
data_size = point_cloud.shape[0]
sample_size = int(0.1 * data_size)
print(f"Original data size: {data_size} points. Reducing to {sample_size} points.")
point_cloud = point_cloud[np.random.choice(data_size, sample_size, replace=False)]
print("Downsampled point cloud shape:", point_cloud.shape)

# Use RANSAC to further filter the point cloud (using the new ransac_tda method)
# Initialize a tda instance (here we use persistence entropy extractor as an example)
print("Initializing TDA feature extractor (persistence entropy)...")
my_tda_entropy = tda(homo_dim=1, fts='entropy')
filtered_point_cloud = my_tda_entropy.ransac_tda(point_cloud, residual_threshold=1.0, max_trials=1000)

# Optionally, visualize the filtered point cloud in 3D using a colormap:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(filtered_point_cloud[:, 0], filtered_point_cloud[:, 1], filtered_point_cloud[:, 2],
                c=filtered_point_cloud[:, 2], cmap='viridis', s=1)
fig.colorbar(sc, ax=ax, label='Elevation')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Scatter Plot of Filtered Point Cloud (RANSAC Inliers)')
plt.show()

# Extract TDA features (this will also plot the persistence diagram)
features_entropy = my_tda_entropy([filtered_point_cloud])
print("Extracted persistence entropy features:")
print(features_entropy)

# Save the homology diagrams (H0 and H1) to CSV files for later analysis
print("Saving homology diagrams for H0 and H1...")
my_tda_entropy.save_homology_dimensions(diagram_index=0, filename_prefix="point_cloud")

# You can also initialize and extract other TDA features as needed:
print("Initializing TDA feature extractor (number of points)...")
my_tda_num = tda(homo_dim=1, fts='numofpoints')
features_num = my_tda_num([filtered_point_cloud])
print("Extracted number-of-points features:")
print(features_num)

print("Initializing TDA feature extractor (amplitude)...")
my_tda_amp = tda(homo_dim=1, fts='amp')
features_amp = my_tda_amp([filtered_point_cloud])
print("Extracted amplitude features:")
print(features_amp)

