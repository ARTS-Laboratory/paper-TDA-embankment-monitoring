import laspy
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import NumberOfPoints, Amplitude, PersistenceEntropy
from gtda.plotting import plot_diagram     # plot PD
import numpy as np
import matplotlib.pyplot as plt

class tda:
    def __init__(self, homo_dim=1, fts='entropy') -> None:
        """
        homo_dim: homology dimensions, integer value
        fts: features need to return, options: entropy, numofpoints, amp
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
        pcd: the point cloud list to be calculated tda features. For a single point cloud, wrap it in a list, e.g., tda([pcd])
        return: 2D numpy array of extracted TDA features
        """
        print("computing persistence diagrams on point cloud(s)...")
        self.diag = self.persistence.fit_transform(pcd)
        print("persistence diagrams computed.")
        
        # Plot the persistence diagram only for the 'entropy' extractor
        if self.fts == 'entropy':
            print("Plotting the persistence diagram...")
            plot_diagram(self.diag[0])
            plt.title("Persistence Diagram")
            plt.xlim(0, 10)
            plt.ylim(0, 10)
            plt.show()
        
        # Features extraction (persistence entropy/ number of points/ amplitude (all metrics))
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

    # Method for saving homology dimensions into CSV files 
    def save_homology_dimensions(self, diagram_index=0, filename_prefix="homology"):
        """
        Saves the H0 and H1 persistence diagram data as CSV files.
        """
        diag = self.diag[diagram_index]
        # The third column indicates the homology dimension.
        H0 = diag[diag[:, 2] == 0]
        H1 = diag[diag[:, 2] == 1]
        
        # CSV files for homology dimensions (H0 and H1)
        np.savetxt(f"{filename_prefix}_H0.csv", H0, delimiter=",", header="birth,death,dimension", comments='')
        np.savetxt(f"{filename_prefix}_H1.csv", H1, delimiter=",", header="birth,death,dimension", comments='')
        print(f"Saved H0 persistence diagram to {filename_prefix}_H0.csv")
        print(f"Saved H1 persistence diagram to {filename_prefix}_H1.csv")

    def __call__(self, pcd):
        return self.forward(pcd)


# Process the point cloud using a LAS file and extract TDA features:

file_path = "C:/Users/GOLZARDM/.spyder-py3/PC of surface_with_smooth_circular_hump.las"
print("Opening LAS file:", file_path)
with laspy.open(file_path) as f:
    las = f.read()
print("LAS file read successfully.")

x = las.x
y = las.y
z = las.z
# LAS file converted to array
point_cloud = np.vstack((x, y, z)).T
print("Point cloud shape:", point_cloud.shape)

# Manual downsampling
sample_size = 9000
if point_cloud.shape[0] > sample_size:
    point_cloud = point_cloud[np.random.choice(point_cloud.shape[0], sample_size, replace=False)]
    print("Downsampled point cloud shape:", point_cloud.shape)


# TDA features extraction 

# Persistence entropy extraction (this will also plot the persistence diagram)
print("Initializing TDA feature extractor (persistence entropy)...")
my_tda_entropy = tda(homo_dim=1, fts='entropy')
features_entropy = my_tda_entropy([point_cloud])
print("Extracted persistence entropy features:")
print(features_entropy)

# Save the homology diagrams (H0 and H1) to CSV files (for later postanalysis analysis)
print("Saving homology diagrams for H0 and H1...")
my_tda_entropy.save_homology_dimensions(diagram_index=0, filename_prefix="point_cloud")


print("Initializing TDA feature extractor (number of points)...")
my_tda_num = tda(homo_dim=1, fts='numofpoints')
features_num = my_tda_num([point_cloud])
print("Extracted number-of-points features:")
print(features_num)

# Amplitude extraction:
print("Initializing TDA feature extractor (amplitude)...")
my_tda_amp = tda(homo_dim=1, fts='amp')
features_amp = my_tda_amp([point_cloud])
print("Extracted amplitude features:")
print(features_amp)


