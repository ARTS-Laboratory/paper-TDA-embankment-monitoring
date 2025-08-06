import os
import csv
import laspy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import math
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, NumberOfPoints, Amplitude
#from gtda.plotting import plot_diagram  # PD plotting is deactivated; CSV files will be used later
#from sklearn.linear_model import RANSACRegressor  # Not used in consensus now

# Define the TDA feature extractor class.
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
##################
    def random_sampling_consensus(self, pcd, m=None, K=None):
        """
        Performs random sampling consensus TDA:

        - Dynamically adjusts `m` and `K` based on available points.
        - Repeats K times:
          1. Randomly sample `m` points (subset) from the point cloud.
          2. Compute the persistence diagram on that subset.
          3. Extract TDA features (entropy, number-of-points, amplitude).
        - Returns all iteration features, and writes out iteration_features.csv
        - Writes median_consensus_features.csv for that file.
        """
        
        # Dynamically adjust `m` and `K` based on dataset size
        #m = min(500, int(1 * pcd.shape[0]))  # Use 80% of available points, max 500
        #K = max(3, min(10, int(0.05 * pcd.shape[0])))  # Adjust K dynamically
        
        # Sampling parameters 
        m_percentage = 1  # Use 30% of data
        K = 1    # Number of iterations.
        print(f"Using m={m} (sample size) and K={K} (iterations) for TDA.")
        
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
        # Remove existing file if it exists
        if os.path.exists("iteration_features.csv"):
            os.remove("iteration_features.csv")
        np.savetxt("iteration_features.csv", features_array, delimiter=",", 
                   header="All iteration feature vectors", comments='')
        median_features = np.median(features_array, axis=0)
        print("Median consensus features:", median_features)
        if os.path.exists("median_consensus_features.csv"):
            os.remove("median_consensus_features.csv")
        np.savetxt("median_consensus_features.csv", median_features.reshape(1, -1), delimiter=",", 
                   header="Median consensus feature vector", comments='')
        return median_features

    def forward(self, pcd_list):
        """
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

    def save_homology_dimensions(self, diagram_index=0, filename_prefix="point_cloud"):
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



########## the main Code starts from this section
# 1- Process multiple LAS files and accumulate median consensus of features for each imported poit cloud (index)


def process_multiple_las_files():
    #  here is the directory containing the .las files.
    directory_path ="C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing"
    
  #  here you can list the filenames that you wish to process them. (this command automatically calls several inputs_
    file_list = [
        "3D_no_abnormalities.las",
        #"surface_with_smooth_circular_cavity_20.las",
        # .....  add any files
    ]

    #  creating an instance of the tda class.
    my_tda = tda(homo_dim=1, fts='all')
    
    #  creating a list to accumulate median features for each file.
    accumulated_data = []
    
    #  loop over the files in the specified order.
    for filename in file_list:
        file_path = os.path.join(directory_path, filename)
        print(f"\nNow processing file: {file_path}")
        
        # opening and reading the .las file.
        with laspy.open(file_path) as f:
            las = f.read()
        print("LAS file read successfully.")
        
        # make point clouds as array x, y, z.
        x = las.x
        y = las.y
        z = las.z
        point_cloud = np.vstack((x, y, z)).T
        print("Point cloud shape:", point_cloud.shape)
        
        ########### this part is important
        # note: if we use the PCA we have less dataset which is not logical to sample from them so, the m conditionally selected
        # note: if we use the PCA we have ledd dataset which is not logical to run them 10 times so, the k conditionally selected
        
        # random sampling TDA on this point cloud. 
        
         #m = min(500, int(1 * point_cloud.shape[0]))  # here using 100% of available points, max 500
         #K = max(3, min(10, int(0.05 * point_cloud.shape[0])))  # between 3 and 10 iterations
        
        # Show total point count
        print(f"Total available points: {point_cloud.shape[0]}")

        # === Sampling parameters ===
        m_percentage = 1  # Use 30% of data
        K = 1    # Number of iterations.
        #k_ratio = 0.05      # 5% of points for K

        # Set m (number of sampled points)
        m = int(m_percentage * point_cloud.shape[0])
        print(f"Sampling {m} points ({m_percentage * 100:.0f}% of total)")

        # Set K (iterations or neighbors)
        #K = max(3, min(10, int(k_ratio * point_cloud.shape[0])))
        #print(f"Using K = {K} (based on {k_ratio * 100:.1f}% of total)")
        
        print(f"Using m={m} (sample size) and K={K} (iterations) for file: {filename}")
        median_features = my_tda.random_sampling_consensus(point_cloud, m=m, K=K)
        
        #  computing the persistence diagram on the full point cloud (for saving homology diagrams)
        _ = my_tda([point_cloud])
        out_prefix = os.path.splitext(filename)[0]  # e.g., "Simple slop"
        my_tda.save_homology_dimensions(diagram_index=0, filename_prefix=out_prefix)
        
        #  Accumulate the median features for this file (filename + median features)
        row_data = np.concatenate(([filename], median_features))
        accumulated_data.append(row_data)
    
    #  saving the accumulated median features for all files into one CSV
    if accumulated_data:
        # determining the number of numeric features.
        numeric_length = len(accumulated_data[0]) - 1
        # defining  header with the feature names
        header_list = [
            "filename",
            "Persistence Entropy (H0)",
            "Persistence Entropy (H1)",
            "Number of Points (H0)",
            "Number of Points (H1)",
            "Bottleneck Amplitude (H0)",
            "Bottleneck Amplitude (H1)",
            "Wasserstein Amplitude (H0)",
            "Wasserstein Amplitude (H1)",
            "Landscape Amplitude (H0)",
            "Landscape Amplitude (H1)",
            "Persistence Image Amplitude (H0)",
            "Persistence Image Amplitude (H1)",
            "Betti Amplitude (H0)",
            "Betti Amplitude (H1)",
            "Heat Amplitude (H0)",
            "Heat Amplitude (H1)"
        ]
        
        output_csv = "accumulated_median_consensus_features.csv"
        with open(output_csv, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header_list)
            for row in accumulated_data:
                writer.writerow(row)
        print("\nFinal CSV saved:", output_csv)
    else:
        print("No data accumulated. Possibly no files were processed.")

# runing the main function of TDA
if __name__ == "__main__":
    process_multiple_las_files()

