import os
import laspy
import numpy as np
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, NumberOfPoints, Amplitude
import matplotlib.colors as mcolors
import matplotlib.cm as cm

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
        # We'll compute homology up to dimension = homo_dim
        # If you want H2, set homo_dim=2, etc.
        self.homology_dimensions = list(range(homo_dim + 1))
        print("Homology dimensions:", self.homology_dimensions)
        
        # Create a Vietoris-Rips persistence object
        self.persistence = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=self.homology_dimensions,
            n_jobs=-1,
            max_edge_length=1e9  # large to avoid cutting off features
        )
        self.fts = fts
        
        # Initialize feature extractors
        self.persistence_entropy = PersistenceEntropy()
        self.NumOfPoint = NumberOfPoints()
        # We'll also compute amplitude-based features with these metrics:
        self.metrics = ["bottleneck", "wasserstein", "landscape", 
                        "persistence_image", "betti", "heat"]
        
        # We store the persistence diagrams in self.diag after computing them
        self.diag = None

    def random_sampling_consensus(self, pcd, m=500, K=10):
        """
        Performs random sampling consensus TDA:
        
        - Repeats K times:
          1. Randomly sample m points (subset) from the point cloud.
          2. Compute the persistence diagram on that subset.
          3. Extract TDA features (entropy, number-of-points, amplitude).
        - Returns all iteration features, and also writes out iteration_features.csv
        - Writes median_consensus_features.csv for that file
        """
        features_list = []
        for i in range(K):
            print(f"Iteration {i+1}/{K}:")
            
            # Randomly sample m points if pcd has more than m
            if pcd.shape[0] > m:
                idx = np.random.choice(pcd.shape[0], size=m, replace=False)
                subset = pcd[idx, :]
            else:
                subset = pcd
            
            # Compute the persistence diagram on the subset
            diag = self.persistence.fit_transform([subset])
            
            # Extract features
            feat_entropy = self.persistence_entropy.fit_transform(diag)
            feat_num = self.NumOfPoint.fit_transform(diag)
            
            # Compute amplitude features for multiple metrics
            amps = []
            from gtda.diagrams import Amplitude
            for metric in self.metrics:
                AMP = Amplitude(metric=metric)
                amp = AMP.fit_transform(diag)
                amps.append(amp)
            
            # Combine amplitude features horizontally
            feat_amp = np.hstack(amps) if amps else np.array([])
            
            # Combine everything into one row
            iteration_features = np.hstack((feat_entropy, feat_num, feat_amp))
            print(f"  Computed features: {iteration_features}")
            features_list.append(iteration_features)
        
        # Stack all iteration features vertically
        features_array = np.vstack(features_list)
        
        # Write out iteration_features.csv (optional)
        np.savetxt("iteration_features.csv", features_array, delimiter=",", 
                   header="All iteration feature vectors", comments='')
        
        # Compute the median across iterations
        median_features = np.median(features_array, axis=0)
        print("Median consensus features:", median_features)
        
        # Write out the median consensus features
        np.savetxt("median_consensus_features.csv", 
                   median_features.reshape(1, -1), 
                   delimiter=",", 
                   header="Median consensus feature vector", 
                   comments='')
        
        return median_features  # Return just the median for this file

    def forward(self, pcd_list):
        """
        Given a list of point clouds (or a single point cloud in a list),
        compute the persistence diagram(s) and extract TDA features.
        """
        print("Computing persistence diagrams on point cloud(s)...")
        self.diag = self.persistence.fit_transform(pcd_list)
        print("Persistence diagrams computed.")
        
        # Now extract features from the full diagram
        features_entropy = self.persistence_entropy.fit_transform(self.diag)
        features_num = self.NumOfPoint.fit_transform(self.diag)
        
        from gtda.diagrams import Amplitude
        amps = []
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
        
        # Filter out rows for H0 and H1
        H0 = diag[diag[:, 2] == 0]
        H1 = diag[diag[:, 2] == 1]
        
        # Write them to CSV
        np.savetxt(f"{filename_prefix}_H0.csv", H0, delimiter=",", 
                   header="birth,death,dimension", comments='')
        np.savetxt(f"{filename_prefix}_H1.csv", H1, delimiter=",", 
                   header="birth,death,dimension", comments='')
        
        print(f"Saved H0 persistence diagram to {filename_prefix}_H0.csv")
        print(f"Saved H1 persistence diagram to {filename_prefix}_H1.csv")

    def __call__(self, pcd_list):
        return self.forward(pcd_list)


########
# Main code: extract TDA features from multiple .las files in a chosen order
########

def process_multiple_las_files():
    # 1) Directory containing your .las files
    directory_path = "C:/Users/GOLZARDM/Documents/GitHub/paper-TDA-embankment-monitoring/Toy-example/Test/Practice"
    
    # 2) List of filenames in the order you want them processed
    file_list = [
        "Simple slop.las",
        "surface_with_smooth_circular_cavity_20.las",
        # Add more filenames as needed...
    ]
    
    # 3) Create an instance of the tda class
    #    If you want up to H1, set homo_dim=1. For up to H2, set homo_dim=2, etc.
    my_tda = tda(homo_dim=1, fts='all')
    
    # 4) We'll store the median features for each file in this list
    accumulated_data = []
    
    # 5) Loop over the files in the specified order
    for filename in file_list:
        file_path = os.path.join(directory_path, filename)
        print(f"\nNow processing file: {file_path}")
        
        # Open and read the .las file
        with laspy.open(file_path) as f:
            las = f.read()
        print("LAS file read successfully.")
        
        # Extract x, y, z coordinates
        x = las.x
        y = las.y
        z = las.z
        point_cloud = np.vstack((x, y, z)).T
        print("Point cloud shape:", point_cloud.shape)
        
        # 6) Random sampling TDA on this point cloud
        m = 500  # sample size
        K = 10   # number of iterations
        median_features = my_tda.random_sampling_consensus(point_cloud, m=m, K=K)
        
        # 7) Optionally visualize a subset of the data
        if point_cloud.shape[0] > m:
            idx = np.random.choice(point_cloud.shape[0], size=m, replace=False)
            pcd_sampled = point_cloud[idx, :]
        else:
            pcd_sampled = point_cloud
        
        # fig = plt.figure(figsize=(8,6))
        # ax = fig.add_subplot(111, projection='3d')
        # sc = ax.scatter(pcd_sampled[:, 0], pcd_sampled[:, 1], pcd_sampled[:, 2],
        #                 c=pcd_sampled[:, 2], cmap='viridis', s=5)
        # plt.colorbar(sc, ax=ax, label='elevation')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.title(f"Randomly sampled subset of the point cloud: {filename}")
        # plt.show()
        
        # 8) Compute the persistence diagram on the full point cloud
        #    so we can save H0 and H1 for this file
        _ = my_tda([point_cloud])
        
        # 9) Save the homology dimensions for H0 and H1
        out_prefix = os.path.splitext(filename)[0]  # e.g. "Simple slop"
        my_tda.save_homology_dimensions(diagram_index=0, filename_prefix=out_prefix)
        
        # 10) Accumulate the median features for this file
        #     You might also store the filename in the row for reference
        row_data = np.concatenate(([filename], median_features))
        accumulated_data.append(row_data)
    
    # 11) At the end, write out the accumulated data to a single CSV
    # We'll assume each median_features has the same length
    # We can build a header row that includes "filename" + placeholders
    # for the number of columns in median_features
    if accumulated_data:
        # Number of columns in median_features (the first file's length minus 1 for the filename)
        # Actually the first element is the filename, so the rest are numeric
        numeric_length = len(accumulated_data[0]) - 1
        header_list = ["filename"] + [f"feature_{i+1}" for i in range(numeric_length)]
        
        # Convert to a 2D array (list of lists) for easy saving
        # Each row in accumulated_data is already a list
        # We'll save it as a CSV
        import csv
        with open("accumulated_median_consensus_features.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header_list)
            for row in accumulated_data:
                writer.writerow(row)
        print("\nFinal CSV saved: accumulated_median_consensus_features.csv")
    else:
        print("No data accumulated. Possibly no files were processed.")

# Run the main function if desired
if __name__ == "__main__":
    process_multiple_las_files()
