import laspy
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import NumberOfPoints, Amplitude, PersistenceEntropy
import numpy as np

class tda:
    def __init__(self, homo_dim=1, fts='entropy') -> None:
        """
        homo_dim: highest homology dimension to compute.
        fts: feature extraction method, options: 'entropy', 'numofpoints', or 'amp'.
        """
        # Create list of homology dimensions; e.g., homo_dim=1 gives [0, 1]
        self.homology_dimensions = list(range(homo_dim + 1))
        print("Homology dimensions:", self.homology_dimensions)
        
        # Initialize the persistence calculator with a maximum edge length to limit computation
        self.persistence = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=self.homology_dimensions,
            max_edge_length=50,  # Adjust this value based on your data's scale
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
            for m in self.metrics:
                from gtda.diagrams import Amplitude
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
file_path = "C:/Users/GOLZARDM/.spyder-py3/PC of surface_with_smooth_circular_hump.las"  # Update path if needed

print("Opening LAS file:", file_path)
with laspy.open(file_path) as f:
    las = f.read()
print("LAS file read successfully.")

x = las.x
y = las.y
z = las.z
point_cloud = np.vstack((x, y, z)).T  # Now point_cloud is an (N x 3) array
print("Point cloud shape:", point_cloud.shape)

# -----------------------------
# Downsample the point cloud for performance (if needed)
# -----------------------------
sample_size = 5000  # Reduced sample size for faster computation
if point_cloud.shape[0] > sample_size:
    point_cloud = point_cloud[np.random.choice(point_cloud.shape[0], sample_size, replace=False)]
    print("Downsampled point cloud shape:", point_cloud.shape)

# -----------------------------
# Compute TDA features using the tda class
# -----------------------------
print("Initializing TDA feature extractor...")
my_tda = tda(homo_dim=1, fts='entropy')  # Use 'entropy' for persistence entropy features

print("Computing TDA features...")
# Wrap the point cloud in a list because the TDA class expects a list of point clouds
features = my_tda([point_cloud])
print("Extracted TDA features:")
print(features)


