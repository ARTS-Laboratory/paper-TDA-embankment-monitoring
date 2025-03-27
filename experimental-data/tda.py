from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import NumberOfPoints, Amplitude, PersistenceEntropy
import numpy as np
from tda_utils import *





def pca_bg_removal(pcd, delta=1.5):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(pcd)
    components = pca.components_
    r_pts = pcd.dot(components.T)
    p_max, p_min = np.max(r_pts[:, 2]), np.min(r_pts[:, 2])
    if p_max < 0:
        r_pts[:, 2] -= p_max
    if p_min > 0:
        r_pts[:, 2] -= p_min
    std = np.std(r_pts[:, 2])
    high, low = delta * std, -delta * std
    hump = r_pts[r_pts[:, 2] > high]
    cavity = r_pts[r_pts[:, 2] < low]
    f_pts = np.concatenate((hump, cavity), axis=0)
    return f_pts



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
            max_edge_length=200,
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



def ransac_tda(pcd, dim=2, fts='entropy', k=10, m=500):
    """
    pcd: the point cloud list to be calculated tda features. For single point cloud, add "[]", for example, tda([pcd])
    k: the number of iterations
    m: the number of points to sample in each iteration
    return: 2D numpy array, features extracted from inputing point cloud list
    """
    tda_ = tda(homo_dim=dim, fts=fts)
    rslt = []
    for _ in range(k):
        if len(pcd) == 0:
            raise ValueError("The point cloud is empty. Cannot perform RANSAC.")
        idx = np.random.choice(len(pcd), m)
        pcd_ = [pcd[j] for j in idx]
        pcd_ = np.array(pcd_).reshape((m, 3))
        rslt.append(tda_([pcd_]))
    return np.median(rslt, axis=0)


def ransac_tda_m(pcd, dim=2, fts='entropy', k=10, m=500):
    """
    pcd: the point cloud list to be calculated tda features. For single point cloud, add "[]", for example, tda([pcd])
    k: the number of iterations
    m: the number of points to sample in each iteration
    return: 2D numpy array, features extracted from inputing point cloud list
    """
    tda_ = tda(homo_dim=dim, fts=fts)
    rslt = np.empty((len(pcd), k+1))
    for _ in range(k):
        pcd_ = []
        for p in pcd:
            idx = np.random.choice(len(p), m)
            pcd_.append(p[idx])
        # pcd_ = np.array(pcd_).reshape((m, 3))
        # rslt.append(tda_([pcd_]))
        rslt[:, _] = tda_(pcd_)[:, int(dim)]
    rslt[:, -1] = np.mean(rslt[:, :-1], axis=-1)
    return rslt


def ransac_persistence_m(pcd, dim=2, k=10, m=10000):    
    tda_ = tda(homo_dim=dim)
    max_lst, mean_lst, sum_lst, num_h2_lst, entropy_lst,  = [], [], [], [], []
    lifetimes_arr = np.empty((len(pcd), k+1))
    for _ in range(k):
        pcd_lst = []
        for points in pcd:
            ind = np.random.choice(len(points), m, replace=False)
            pcd_lst.append(points[ind].reshape((-1, 3)))
        
        entropy = tda_(pcd_lst)
        entropy_lst.append(entropy)

        diags = tda_.diag
        for i in range(diags.shape[0]):
            diag = diags[i,:]
            H2_features = diag[diag[:, 2] == int(dim)]   # Extract H2 (voids)
            lifetimes = H2_features[:, 1] - H2_features[:, 0]  # Death - Birth
            max_persistence = np.max(lifetimes)
            lifetimes_arr[i, _] = max_persistence

    lifetimes_arr[:, -1] = np.mean(lifetimes_arr[:, :-1], axis=-1)
    return lifetimes_arr



        


