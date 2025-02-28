import laspy
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import NumberOfPoints, Amplitude, PersistenceEntropy
from gtda.plotting import plot_diagram
import numpy as np
import matplotlib.pyplot as plt

class tda:
    def __init__(self, homo_dim=1, fts='entropy') -> None:
        """
        homo_dim: homology dimensions, integer value.
        fts: feature extraction method, options: 'entropy', 'numofpoints', or 'amp'.
        """
        self.homology_dimensions = list(range(homo_dim + 1))
        print("Homology dimensions:", self.homology_dimensions)
        self.persistence = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=self.homology_dimensions,
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
        
        # Plot the persistence diagram and set the axis scale to 10
        print("Plotting the persistence diagram...")
        plot_diagram(self.diag[0])
        plt.title("Persistence Diagram")
        plt.xlim(0, 10)
        plt.ylim(0, 10)
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

# Print the first 10 rows (points) of the point cloud matrix
print("First 10 points of the point cloud:")
print(point_cloud[:10])

# -----------------------------
# Downsample the point cloud for performance (if needed)
# -----------------------------
sample_size = 5000  # Adjust as needed
if point_cloud.shape[0] > sample_size:
    point_cloud = point_cloud[np.random.choice(point_cloud.shape[0], sample_size, replace=False)]
    print("Downsampled point cloud shape:", point_cloud.shape)

# -----------------------------
# Compute TDA features using the tda class
# -----------------------------
print("Initializing TDA feature extractor...")
# To extract persistence entropy features:
my_tda_entropy = tda(homo_dim=1, fts='entropy')
print("Computing TDA features...")
# The TDA class expects a list of point clouds, so we wrap our point_cloud in a list.
features_entropy = my_tda_entropy([point_cloud])
print("Extracted persistence entropy features:")
print(features_entropy)

# To extract number-of-points features:
my_tda_num = tda(homo_dim=1, fts='numofpoints')
features_num = my_tda_num([point_cloud])
print("Extracted number-of-points features:")
print(features_num)

# To extract amplitude features:
my_tda_amp = tda(homo_dim=1, fts='amp')
features_amp = my_tda_amp([point_cloud])
print("Extracted amplitude features:")
print(features_amp)


