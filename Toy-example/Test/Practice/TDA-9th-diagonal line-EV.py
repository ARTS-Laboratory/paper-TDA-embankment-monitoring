import laspy
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import NumberOfPoints, Amplitude, PersistenceEntropy
from gtda.plotting import plot_diagram
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class tda:
    def __init__(self, homo_dim=1, fts='entropy', max_edge_length=50.0) -> None:
        """
        homo_dim: homology dimensions, integer value.
        fts: feature extraction method, options: 'entropy', 'numofpoints', or 'amp'.
        max_edge_length: maximum edge length for Vietoris-Rips filtration.
        """
        self.homology_dimensions = list(range(homo_dim + 1))
        print("Homology dimensions:", self.homology_dimensions)
        self.persistence = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=self.homology_dimensions,
            max_edge_length=max_edge_length,  # Adjust this parameter
            n_jobs=-1)
        self.fts = fts
        if fts == 'entropy':
            self.persistence_entropy = PersistenceEntropy()
        elif fts == 'numofpoints':
            self.NumOfPoint = NumberOfPoints()
        elif fts == 'amp':
            self.metrics = ["bottleneck", "wasserstein", "landscape", "persistence_image", "betti", "heat"]

    def forward(self, pcd):
        """
        pcd: a list of point clouds (each should be an (N x 3) NumPy array). 
             For a single point cloud, call as: forward([point_cloud])
        Returns: a 2D NumPy array of extracted TDA features.
        """
        print("Computing persistence diagrams on point cloud(s)...")
        self.diag = self.persistence.fit_transform(pcd)
        print("Persistence diagrams computed.")
        
        # Debug: Print the persistence diagram for the first point cloud
        print("Persistence diagram for first point cloud:")
        print(self.diag[0])
        
        # Plot the persistence diagram
        print("Plotting the persistence diagram...")
        plt.figure(figsize=(8, 6))
        plot_diagram(self.diag[0])
        
        # Add diagonal line explicitly
        max_persistence = np.max(self.diag[0][:, 1]) if len(self.diag[0]) > 0 else 10
        plt.plot([0, max_persistence], [0, max_persistence], color='red', linestyle='--', label='Diagonal')
        
        plt.title("Persistence Diagram")
        plt.xlabel("Birth")
        plt.ylabel("Death")
        plt.xlim(0, max_persistence)
        plt.ylim(0, max_persistence)
        plt.legend()
        plt.show()
        
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

    def __call__(self, pcd):
        return self.forward(pcd)

# -----------------------------
# Read the LAS file and convert it to an (N x 3) NumPy array
# -----------------------------
file_path = "C:/Users/GOLZARDM/.spyder-py3/PC of surface_with_smooth_circular_hump.las"
print("Opening LAS file:", file_path)
with laspy.open(file_path) as f:
    las = f.read()
print("LAS file read successfully.")

x = las.x
y = las.y
z = las.z
point_cloud = np.vstack((x, y, z)).T
print("Point cloud shape:", point_cloud.shape)

# -----------------------------
# Visualize the point cloud
# -----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, c='b', alpha=0.5)
ax.set_title("3D Point Cloud")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# -----------------------------
# Downsample the point cloud for performance (if needed)
# -----------------------------
sample_size = 5000  # Adjust as needed
if point_cloud.shape[0] > sample_size:
    point_cloud = point_cloud[np.random.choice(point_cloud.shape[0], sample_size, replace=False)]
    print("Downsampled point cloud shape:", point_cloud.shape)

# -----------------------------
# Add synthetic features (e.g., noise or a hole) for testing
# -----------------------------
# Add a small amount of noise
noise = np.random.normal(0, 0.1, point_cloud.shape)  # Adjust noise level as needed
point_cloud_noisy = point_cloud + noise

# -----------------------------
# Compute TDA features using the tda class
# -----------------------------
print("Initializing TDA feature extractor...")
my_tda = tda(homo_dim=1, fts='entropy', max_edge_length=50.0)  # Adjust max_edge_length as needed

print("Computing TDA features...")
features = my_tda([point_cloud_noisy])  # Use the noisy point cloud
print("Extracted TDA features:")
print(features)
