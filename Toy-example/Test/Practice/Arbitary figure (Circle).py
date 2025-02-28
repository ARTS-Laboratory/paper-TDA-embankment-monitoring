import numpy as np
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram

def generate_noisy_circle(n_points=300, noise_std=0.05):
    """
    Generate 2D points on a circle with added Gaussian noise.
    
    Parameters:
      n_points (int): Number of points.
      noise_std (float): Standard deviation of noise.
      
    Returns:
      data_2d : np.ndarray of shape (n_points, 2)
    """
    # Evenly spaced angles
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    circle = np.column_stack((np.cos(theta), np.sin(theta)))
    # Add Gaussian noise
    noise = noise_std * np.random.randn(n_points, 2)
    return circle + noise

# Generate a noisy circle in 2D and embed in 3D by adding a zero z-coordinate.
data_2d = generate_noisy_circle(n_points=300, noise_std=0.05)
data_3d = np.hstack((data_2d, np.zeros((data_2d.shape[0], 1))))
print("Noisy circle (3D) shape:", data_3d.shape)

# Visualize the noisy circle (2D projection)
plt.figure(figsize=(5,5))
plt.scatter(data_2d[:, 0], data_2d[:, 1], s=10, c=data_2d[:, 0], cmap='viridis')
plt.title("Noisy Circle (2D projection)")
plt.axis("equal")
plt.show()

# Create Vietoris-Rips Persistence object:
VR = VietorisRipsPersistence(
    metric="euclidean",
    homology_dimensions=[0, 1],  # We care about connected components (H0) and loops (H1)
    max_edge_length=2.0,         # For a circle of radius ~1, this should allow the loop to form.
    n_jobs=-1
)

# Giotto-tda expects a list of arrays if you have a single point cloud.
diagrams = VR.fit_transform([data_3d])

# Debug: Print the raw persistence diagram array and its shape.
print("Persistence diagram array:")
print(diagrams[0])
print("Shape:", diagrams[0].shape)

# Plot the persistence diagram.
plt.figure(figsize=(6,4))
plot_diagram(diagrams[0], homology_dimensions=[0, 1])
plt.title("Persistence Diagram for Noisy Circle (H0 and H1)")
plt.xlabel("Birth")
plt.ylabel("Death")
plt.show()

