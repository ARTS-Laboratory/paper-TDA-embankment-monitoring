import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import laspy
from sklearn.decomposition import PCA

# LaTeX Formatting
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})

# Load LAS file
las = laspy.read("C:/Users/GOLZARDM/Documents/paper-TDA-embankment-monitoring/Toy-example/Data/surface_with_smooth_circular_cavity_50.las")
xyz = np.vstack((las.x, las.y, las.z)).T

# PCA transformation
pca = PCA(n_components=3)
pc_values = pca.fit_transform(xyz)

# covariance matrices
cov_matrix_before_pca = np.cov(xyz.T)
cov_matrix_after_pca = np.cov(pc_values.T)
rounded_before = np.round(cov_matrix_before_pca).astype(int)
rounded_after = np.round(cov_matrix_after_pca).astype(int)

# plot matrix before PCA 
fig = plt.figure(figsize=(2, 2), dpi=350)

# matrix values (numbers)
heatmap_before = sns.heatmap(
    rounded_before, annot=True, fmt="d", cmap="viridis",
    annot_kws={"fontfamily": "serif", "fontsize": 6},  # matrix text size
    xticklabels=['X', 'Y', 'Z'],
    yticklabels=['X', 'Y', 'Z']
)

# axis labels
plt.xlabel('dimensions', fontsize=6, fontfamily='serif')  # X-label
plt.ylabel('dimensions', fontsize=6, fontfamily='serif')  # Y-label

#  axis ticks
plt.xticks(fontsize=6, fontfamily='serif')
plt.yticks(fontsize=6, fontfamily='serif')

#  colorbar including label & ticks
cb = heatmap_before.collections[0].colorbar
cb.set_label("covariance Value", fontsize=6, fontfamily='serif')
cb.ax.tick_params(labelsize=6)
for t in cb.ax.get_yticklabels():
    t.set_fontname('Times New Roman')
    t.set_fontsize(6)

plt.tight_layout(pad=0.2)
plt.show()

# Plot After PCA 
fig = plt.figure(figsize=(2, 2), dpi=350)

# matrix values (numbers)
heatmap_after = sns.heatmap(
    rounded_after, annot=True, fmt="d", cmap="viridis",
    annot_kws={"fontfamily": "serif", "fontsize": 6},  # Matrix text size
    xticklabels=['PC1', 'PC2', 'PC3'],
    yticklabels=['PC1', 'PC2', 'PC3']
)

# axis labels
plt.xlabel('principal components', fontsize=6, fontfamily='serif')
plt.ylabel('principal components', fontsize=6, fontfamily='serif')

#  axis ticks
plt.xticks(fontsize=6, fontfamily='serif')
plt.yticks(fontsize=6, fontfamily='serif')

#  colorbar including label & ticks
cb = heatmap_after.collections[0].colorbar
cb.set_label("covariance value", fontsize=6, fontfamily='serif')
cb.ax.tick_params(labelsize=6)
for t in cb.ax.get_yticklabels():
    t.set_fontname('Times New Roman')
    t.set_fontsize(6)

plt.tight_layout(pad=0.2)
plt.show()

