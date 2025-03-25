import laspy
import numpy as np
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, NumberOfPoints, Amplitude
import re

# Class definition for TDA
class tda:
    def __init__(self, homo_dim=1, fts='all') -> None:
        self.homology_dimensions = list(range(homo_dim + 1))
        print("Homology dimensions:", self.homology_dimensions)
        self.persistence = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=self.homology_dimensions,
            n_jobs=-1,
            max_edge_length=1e9
        )
        self.fts = fts
        self.persistence_entropy = PersistenceEntropy()
        self.NumOfPoint = NumberOfPoints()
        self.metrics = ["bottleneck", "wasserstein", "landscape", "persistence_image", "betti", "heat"]
        self.diag = None

    def random_sampling_consensus(self, pcd, m=100, K=10):
        features_list = []
        for i in range(K):
            print(f"Iteration {i+1}/{K}:")
            if pcd.shape[0] > m:
                idx = np.random.choice(pcd.shape[0], size=m, replace=False)
                subset = pcd[idx, :]
            else:
                subset = pcd
            diag = self.persistence.fit_transform([subset])
            feat_entropy = self.persistence_entropy.fit_transform(diag)
            feat_num = self.NumOfPoint.fit_transform(diag)
            amps = []
            for metric in self.metrics:
                AMP = Amplitude(metric=metric)
                amp = AMP.fit_transform(diag)
                amps.append(amp)
            feat_amp = np.hstack(amps) if amps else np.array([])
            iteration_features = np.hstack((feat_entropy, feat_num, feat_amp))
            features_list.append(iteration_features)
        features_array = np.vstack(features_list)
        median_features = np.median(features_array, axis=0)
        return features_array

    def forward(self, pcd_list):
        self.diag = self.persistence.fit_transform(pcd_list)
        features_entropy = self.persistence_entropy.fit_transform(self.diag)
        features_num = self.NumOfPoint.fit_transform(self.diag)
        amps = []
        for metric in self.metrics:
            AMP = Amplitude(metric=metric)
            amp = AMP.fit_transform(self.diag)
            amps.append(amp)
        features_amp = np.hstack(amps) if amps else np.array([])
        all_features = np.hstack((features_entropy, features_num, features_amp))
        return all_features

# from this section the main body of program starts

# Load the LAS file
file_path = "C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Toy-example/Data/surface_with_smooth_circular_cavity_20.las"
print("Opening LAS file:", file_path)
with laspy.open(file_path) as f:
    las = f.read()

x = las.x
y = las.y
z = las.z
point_cloud = np.vstack((x, y, z)).T
print("Original point cloud shape:", point_cloud.shape)

# Use the full point cloud for the random sampling consensus procedure.
my_tda = tda(homo_dim=1, fts='all')

# sample sizes and number of iterations here can be defined
m_values = [50, 200, 500, 1000, 1500, 2500, 3500, 4500]
K = 10  # number of iterations
median_features_list = []

for m in m_values:
    print(f"\nProcessing sample size m = {m}")
    iteration_features = my_tda.random_sampling_consensus(point_cloud, m=m, K=K)
    median_features = np.median(iteration_features, axis=0)
    median_features_list.append(median_features)

# Convert to normalized median features
median_features_array = np.vstack(median_features_list)
normalized_features = (median_features_array - median_features_array.min(axis=0)) / (
    median_features_array.max(axis=0) - median_features_array.min(axis=0)
)

# fliping the columns to start from zero to one
for i in range(normalized_features.shape[1]):
    if normalized_features[0, i] > normalized_features[-1, i]:
        normalized_features[:, i] = 1 - normalized_features[:, i]

# feature labeling
feature_labels = [
    "persistence entropy (H0)", "persistence entropy (H1)",
    "number of points (H0)", "number of points(H1)",
    "bottleneck  (H0)", "bottleneck  (H1)",
    "wasserstein  (H0)", "wasserstein  (H1)",
    "landscape (H0)", "landscape (H1)",
    "persistence image (H0)", "persistence image (H1)",
    "Betti (H0)", "Betti (H1)",
    "heat (H0)", "heat (H1)"
]

# this section important: here we can select the desired features on concole by adding features nymber
print("\nAvailable features:")
for idx, feature in enumerate(feature_labels):
    print(f"{idx + 1}: {feature}")

user_input = input("\nEnter the feature numbers you want to plot (comma-separated, or type 'all' to plot all): ")

# Process user input
if user_input.lower() == 'all':
    selected_features = feature_labels
else:
    selected_indices = [int(x.strip()) - 1 for x in user_input.split(',')]
    selected_features = [feature_labels[i] for i in selected_indices if 0 <= i < len(feature_labels)]

print("\nSelected features to plot:", selected_features)

# plot each feature vs m
plt.figure(figsize=(6.5, 4), dpi=300)  # High DPI and shorter plot size

for i in range(normalized_features.shape[1]):
    label = feature_labels[i] if i < len(feature_labels) else f"Feature {i+1}"
    if label in selected_features:
        plt.plot(m_values, normalized_features[:, i], marker='o', label=label)

plt.xlabel("Sample size (m)", fontsize=8)
plt.ylabel("Normalized median feature value", fontsize=8)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), fontsize=8, frameon=True)  # Adjusted legend position
plt.grid(True)
plt.tight_layout()
plt.show()


