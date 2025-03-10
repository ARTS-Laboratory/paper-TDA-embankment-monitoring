
import laspy
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import NumberOfPoints, Amplitude, PersistenceEntropy
import numpy as np

class tda:
    def __init__(self, homo_dim=2, fts='entropy') -> None:
        """
        homo_dim: homology dimensions, integer value
        fts: features need to return, options: entropy, numofpoints, amp
        """
        self.homology_dimensions = list(range(homo_dim + 1))
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
        pcd: the point cloud list to be calculated tda features. For single point cloud, add "[]", for example, tda([pcd])
        return: 2D numpy array, features extracted from inputing point cloud list
        """
        self.diag = self.persistence.fit_transform(pcd)
        if self.fts == 'entropy':
            return self.persistence_entropy.fit_transform(self.diag)
        elif self.fts == 'numofpoints':
            return self.NumOfPoint.fit_transform(self.diag)
        elif self.fts == 'amp':
            rslt = []
            for m in self.metrics:
                AMP = Amplitude(metric=m)
                amp = AMP.fit_transform(self.diag)
                rslt.append(amp)
            return np.hstack(rslt) if rslt else np.array([])


    def __call__(self, pcd):
        return self.forward(pcd)

#### call the 
file_path = "C:/Users/golzardm/.spyder-py3/PC of surface_with_smooth_circular_hump.las"  # 

with laspy.open(file_path) as f:
    las = f.read()

x = las.x
y = las.y
z = las.z
point_cloud = np.vstack((x, y, z)).T  # make array from data point

### calculating the TDA from point cloud
my_tda = tda(homo_dim=2, fts='entropy')
features = my_tda([point_cloud])
print("Extracted TDA features:")
print(features)

