
# libraries
import laspy  
import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
from ripser import ripser  #  persistent homology (PH) calculation
from persim import plot_diagrams  #  persistence diagram (PD) visualization


# function to open the LAS file 
def open_las_file(file_path):
    with laspy.open(file_path) as file:
        return file.read()

# defining the file pass
file_path = "C:/Users/golzardm/.spyder-py3/PC of surface_with_smooth_circular_hump.las"

# open the .las file and extract all coordinates 
las = open_las_file(file_path)
x, y, z = las.x, las.y, las.z
point_cloud = np.vstack((x, y, z)).T  # convert to Nx3 matrix

# downsample the point cloud
sample_size = 5000  # number of sample
if point_cloud.shape[0] > sample_size:
    point_cloud = point_cloud[np.random.choice(point_cloud.shape[0], sample_size, replace=False)]

# Plot the point cloud
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, c=point_cloud[:, 2], cmap="viridis")
ax.set_title("point ploud")
plt.show()

# persistence homology using Ripser package
diagrams = ripser(point_cloud, maxdim=1)['dgms']  # Compute 0D and 1D features

#  plot persistence diagrams (PD)
plot_diagrams(diagrams, show=True, title="persistence diagrams")


