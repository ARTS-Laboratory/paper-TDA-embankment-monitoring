
import numpy as np
import trimesh
import matplotlib.pyplot as plt

class SurfaceWithSmoothCircularHump:
    def __init__(self, params=None):
        # Default parameters
        self.default_params = {
            'length': 100,           # Length of the surface
            'width': 100,            # Width of the surface
            'resolution': 400,       # Grid resolution
            'base_height': 0,        # Base height of the surface
            'slope_x': 0.05,         # Optional slope in the X direction
            'slope_y': 0.02,         # Optional slope in the Y direction
            'hump_radius': 15,       # Radius of the hump
            'hump_center': (50, 50), # Center of the hump (x, y)
            'hump_height': 10        # Maximum height of the hump
        }

        # Update with provided parameters
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)

    def generate_surface(self):
        """Generate a surface with an optional slope and smooth circular hump."""
        # Create coordinate grid
        x = np.linspace(0, self.params['length'], self.params['resolution'])
        y = np.linspace(0, self.params['width'], self.params['resolution'])
        self.X, self.Y = np.meshgrid(x, y)

        # Flat surface with optional slope
        self.Z = (self.params['base_height'] +
                  self.params['slope_x'] * self.X +
                  self.params['slope_y'] * self.Y)

        # Define hump parameters
        hump_x, hump_y = self.params['hump_center']
        hump_radius = self.params['hump_radius']
        hump_height = self.params['hump_height']
        distance_from_center = np.sqrt((self.X - hump_x)**2 + (self.Y - hump_y)**2)

        # Apply smooth circular hump using a paraboloid equation
        inside_hump = distance_from_center <= hump_radius
        self.Z[inside_hump] += np.clip(
            hump_height * (1 - (distance_from_center[inside_hump] / hump_radius)**2),
            0, hump_height
        )

    def save_mesh(self, filename="surface_with_smooth_circular_hump_15.obj"):
        """Save the generated mesh to a file."""
        # Generate mesh
        vertices = np.column_stack((self.X.flatten(), self.Y.flatten(), self.Z.flatten()))
        faces = []
        resolution = self.params['resolution']
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                vertex = i * resolution + j
                faces.append([vertex, vertex + 1, vertex + resolution])
                faces.append([vertex + 1, vertex + resolution + 1, vertex + resolution])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(filename)
        print(f"Mesh saved as {filename}")

    def plot_two_views(self):
        """Plot two views: original tilted side view and an additional perpendicular side view."""
        fig = plt.figure(figsize=(14, 6))

        # First view: Original tilted side view
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(self.X, self.Y, self.Z, cmap='Greens', edgecolor='red', linewidth=0.5)
        ax1.view_init(elev=20, azim=120)  # First perspective
        ax1.set_xlim(0, self.params['length'])
        ax1.set_ylim(0, self.params['width'])
        ax1.set_zlim(self.params['base_height'], self.params['base_height'] + self.params['hump_height'] + 5)
        ax1.set_box_aspect([self.params['length'], self.params['width'], self.params['hump_height']])
        ax1.set_title("Tilted Side View")
        #ax1.set_xlabel('Length')
        #ax1.set_ylabel('Width')
        ax1.set_zlabel('elevation')

        # Second view: Perpendicular side view
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(self.X, self.Y, self.Z, cmap='Greens', edgecolor='red', linewidth=0.5)
        ax2.view_init(elev=0, azim=90)  # Perpendicular side view
        ax2.set_xlim(0, self.params['length'])
        ax2.set_ylim(0, self.params['width'])
        ax2.set_zlim(self.params['base_height'], self.params['base_height'] + self.params['hump_height'] + 5)
        ax2.set_box_aspect([self.params['length'], self.params['width'], self.params['hump_height']])
        ax2.set_title("Perpendicular Side View")
        #ax2.set_xlabel('Length')
        #ax2.set_ylabel('Width')
        ax2.set_zlabel('elevation')

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Define custom parameters (optional)
    params = {
        'length': 200,
        'width': 200,
        'base_height': 10,
        'slope_x': 0.5,      # Small slope along the X direction
        'slope_y': 0.01,     # Small slope along the Y direction
        'hump_radius': 15,   # Radius of the hump
        'hump_center': (100, 100),  # Center of the hump
        'hump_height': 10    # Height of the hump
    }

    # Create and generate surface
    surface = SurfaceWithSmoothCircularHump(params)
    surface.generate_surface()

    # Save the mesh
    surface.save_mesh()

    # Plot the two views
    surface.plot_two_views()


