# Section 1: Generate and Visualize Mildly Curved Embankment-Like Surface with Abnormalities

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create grid for a gently curved embankment surface
x = np.linspace(0, 40, 80)  # length
y = np.linspace(0, 50, 60)  # width
x, y = np.meshgrid(x, y)

# Gently sloped surface with slight curvature (like windshield)
a, b, c = 0.1, 0.03, -0.002  # slope + mild curvature
z_base = a * x + b * y + c * (x**2 + y**2)

# Add a hump and a cavity
z = z_base + \
    2.5 * np.exp(-((x - 25)**2 + (y - 30)**2) / 3) - \
    1.3 * np.exp(-((x - 15)**2 + (y - 10)**2) / 6)
    
X = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
np.save("synthetic_curved_surface.npy", X)  # Save for next section

# Plot the surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='terrain', edgecolor='k', alpha=0.6)
ax.set_title("Synthetic Embankment-Like Surface with Hump and Cavity")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()

