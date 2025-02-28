import laspy
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import NumberOfPoints, Amplitude, PersistenceEntropy
import numpy as np

class tda:
    def __init__(self, homo_dim=1, fts='entropy') -> None:
        """
        homo_dim: homology dimensions, integer value.
        fts: feature extraction method, options: 'entropy', 'numofpoints', 'amp'.
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
        Returns: 2D NumPy array of extracted TDA features.
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
                AMP = Amplitude(metric=m)
                amp = AMP.fit_transform(self.diag)
                rslt.append(amp)
            features = np.hstack(rslt) if rslt else np.array([])
            print("Extracted amplitude features.")
            return features

    def __call__(self, pcd):
        return self.forward(pcd)


# read the las file

file_path = "C:/Users/GOLZARDM/.spyder-py3/PC of surface_with_smooth_circular_hump.las"  

print("Opening LAS file:", file_path)
with laspy.open(file_path) as f:
    las = f.read()
print("LAS file read successfully.")

x = las.x
y = las.y
z = las.z
point_cloud = np.vstack((x, y, z)).T  #  point_cloud in (N x 3) array
print("Point cloud shape:", point_cloud.shape)

# Optional: Downsample the point cloud if it's too large (uncomment if needed)
# sample_size = 10000
# if point_cloud.shape[0] > sample_size:
#     point_cloud = point_cloud[np.random.choice(point_cloud.shape[0], sample_size, replace=False)]
#     print("Downsampled point cloud shape:", point_cloud.shape)

# -----------------------------
# TDA feature computing via tda class
# -----------------------------
print("Initializing TDA feature extractor...")
my_tda = tda(homo_dim=1, fts='entropy')  # You can change fts to 'numofpoints' or 'amp' as needed

# Wrap the point cloud in a list (the TDA class expects a list of point clouds)
print("Computing TDA features...")
features = my_tda([point_cloud])
print("Extracted TDA features:")
print(features)



