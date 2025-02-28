import laspy
import numpy as np
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, NumberOfPoints, Amplitude
#from gtda.plotting import plot_diagram  # PD plotting deactivated, CSV files will be used later

class tda:
    def __init__(self, homo_dim=1, fts='all') -> None:
        """
        Initialize the TDA feature extractor.
        
        Parameters:
            homo_dim (int): Maximum homology dimension to compute.
            fts (str): Feature type to extract. Options:
                       'entropy' for persistence entropy,
                       'numofpoints' for the number of points,
                       'amp' for amplitude features,
                       'all' for concatenating all of these features.
        """
        self.homology_dimensions = list(range(homo_dim + 1))
        print("Homology dimensions:", self.homology_dimensions)
        self.persistence = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=self.homology_dimensions,
            n_jobs=-1,
            max_edge_length=1e9  # Use a very large value to avoid cutoff of features
        )
        self.fts = fts
        # Initialize feature extractors
        self.persistence_entropy = PersistenceEntropy()
        self.NumOfPoint = NumberOfPoints()
        self.metrics = ["bottleneck", "wasserstein", "landscape", "persistence_image", "betti", "heat"]
        self.diag = None

    def random_sampling_consensus(self, pcd, m=50, K=10):
        """
        Implements a consensus procedure using random sampling.
        
        For K iterations, randomly sample a subset of m points from the full point cloud,
        compute the persistence diagram on that subset, extract all TDA features (persistence entropy,
        number-of-points, and amplitude), and return the median feature vector over K iterations.
        
        Parameters:
            pcd (np.ndarray): Full point cloud (shape (N, 3)).
            m (int): Number of points to sample per iteration.
            K (int): Number of random iterations.
        
        Returns:
            median_features (np.ndarray): Median feature vector (consensus) over K iterations.
        """
        features_list = []
        for i in range(K):
            print(f"Iteration {i+1}/{K}:")
            # Randomly sample m points (if available)
            if pcd.shape[0] > m:
                idx = np.random.choice(pcd.shape[0], size=m, replace=False)
                subset = pcd[idx, :]
            else:
                subset = pcd
            # Compute persistence diagram on the subset (wrap it in a list)
            diag = self.persistence.fit_transform([subset])
            # Extract features for this iteration:
            feat_entropy = self.persistence_entropy.fit_transform(diag)
            feat_num = self.NumOfPoint.fit_transform(diag)
            amps = []
            from gtda.diagrams import Amplitude
            for metric in self.metrics:
                AMP = Amplitude(metric=metric)
                amp = AMP.fit_transform(diag)
                amps.append(amp)
            feat_amp = np.hstack(amps) if amps else np.array([])
            iteration_features = np.hstack((feat_entropy, feat_num, feat_amp))
            print(f"  Computed features: {iteration_features}")
            features_list.append(iteration_features)
        features_array = np.vstack(features_list)
        median_features = np.median(features_array, axis=0)
        print("Median consensus features:", median_features)
        return median_features

    def forward(self, pcd_list):
        """
        pcd_list: list of point cloud arrays. For a single point cloud, pass [pcd].
        
        Computes persistence diagrams on the given point cloud(s) and extracts TDA features.
        """
        print("Computing persistence diagrams on point cloud(s)...")
        self.diag = self.persistence.fit_transform(pcd_list)
        print("Persistence diagrams computed.")
        
        features_entropy = self.persistence_entropy.fit_transform(self.diag)
        features_num = self.NumOfPoint.fit_transform(self.diag)
        amps = []
        from gtda.diagrams import Amplitude
        for metric in self.metrics:
            AMP = Amplitude(metric=metric)
            amp = AMP.fit_transform(self.diag)
            amps.append(amp)
        features_amp = np.hstack(amps) if amps else np.array([])

        if self.fts == 'entropy':
            print("Extracted persistence entropy features.")
            return features_entropy
        elif self.fts == 'numofpoints':
            print("Extracted number-of-points features.")
            return features_num
        elif self.fts == 'amp':
            print("Extracted amplitude features.")
            return features_amp
        elif self.fts == 'all':
            all_features = np.hstack((features_entropy, features_num, features_amp))
            print("Extracted all features (entropy, number-of-points, amplitude).")
            return all_features

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

    def __call__(self, pcd_list):
        return self.forward(pcd_list)

###############################################
# Main Code: Process point cloud from LAS file and extract TDA features using random sampling consensus
###############################################
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
print("Original point cloud shape:", point_cloud.shape)

# Use the full point cloud for the random sampling consensus procedure.
print("Using full point cloud for random sampling consensus...")
my_tda = tda(homo_dim=1, fts='all')  # Extract all features

# Run the random sampling consensus: sample m points and repeat K times.
m = 50  # Change the sample size here to 50 points per iteration.
K = 10  # Number of iterations remains unchanged.
consensus_features = my_tda.random_sampling_consensus(point_cloud, m=m, K=K)
print("Consensus features from random sampling:", consensus_features)

# IMPORTANT: Compute the persistence diagram on the full point cloud first,
# so that self.diag is populated before saving.
_ = my_tda([point_cloud])

# Save the persistence diagram data (H0 and H1) as CSV files for later analysis.
print("Saving homology diagrams for H0 and H1...")
my_tda.save_homology_dimensions(diagram_index=0, filename_prefix="point_cloud")
