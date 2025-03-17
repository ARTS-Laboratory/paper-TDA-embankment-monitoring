# Import required libraries
import numpy as np
import trimesh
import matplotlib.pyplot as plt

# Apply LaTeX Formatting for Matplotlib Plots
plt.rcParams.update({'text.usetex': True})  # Enable LaTeX rendering
plt.rcParams.update({'font.family': 'serif'})  # Use serif fonts
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})  
plt.rcParams.update({'font.size': 10})  # Standard font size
plt.rcParams.update({'mathtext.rm': 'serif'})  # Use serif fonts for math text
plt.rcParams.update({'mathtext.fontset': 'custom'})  # Use custom math fonts

class SurfaceWithFeatures:
    def __init__(self, params=None):
        """Initialize the surface with default parameters and user-defined settings."""
        self.default_params = {
            'length': 200,  # Length of the surface
            'width': 200,   # Width of the surface
            'resolution': 300,  # Grid resolution
            'base_height': 0,  # Base height of the surface
            'slope_x': 2,  # Slope in the X direction
            'slope_y': 2,  # Slope in the Y direction
            'humps': [],  # Humps [{'center': (x, y), 'radius': r, 'height': h}]
            'cavities': []  # Cavities [{'center': (x, y), 'radius': r, 'depth': d}]
        }

        self.params = self.default_params.copy()
        if params:
            self.params.update(params)

    def generate_surface(self):
        """Generate a surface with multiple humps and cavities."""
        x = np.linspace(0, self.params['length'], self.params['resolution'])
        y = np.linspace(0, self.params['width'], self.params['resolution'])
        self.X, self.Y = np.meshgrid(x, y)

        # Initialize base surface with slope
        self.Z = (self.params['base_height'] +
                  self.params['slope_x'] * self.X +
                  self.params['slope_y'] * self.Y)

        # Apply humps
        for hump in self.params['humps']:
            hump_x, hump_y = hump['center']
            hump_radius = hump['radius']
            hump_height = hump['height']
            distance = np.sqrt((self.X - hump_x)**2 + (self.Y - hump_y)**2)
            inside_hump = distance <= hump_radius
            self.Z[inside_hump] += np.clip(hump_height * (1 - (distance[inside_hump] / hump_radius)**2), 0, hump_height)

        # Apply cavities
        for cavity in self.params['cavities']:
            cavity_x, cavity_y = cavity['center']
            cavity_radius = cavity['radius']
            cavity_depth = cavity['depth']
            distance = np.sqrt((self.X - cavity_x)**2 + (self.Y - cavity_y)**2)
            inside_cavity = distance <= cavity_radius
            self.Z[inside_cavity] -= np.clip(cavity_depth * (1 - (distance[inside_cavity] / cavity_radius)**2), 0, cavity_depth)

    def save_mesh(self, filename="surface_with_features.obj"):
        """Save the generated mesh to a file."""
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
        """Plot two views of the surface: tilted side view and perpendicular side view."""
        fig = plt.figure(figsize=(14, 6))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(self.X, self.Y, self.Z, cmap='Greens', edgecolor='red', linewidth=0.5)
        ax1.view_init(elev=20, azim=120)
        ax1.set_xlim(0, self.params['length'])
        ax1.set_ylim(0, self.params['width'])
        ax1.set_zlim(0, 200)  # Adjust elevation from 0 to 200
        ax1.set_box_aspect([1, 1, 1])  # Maintain equal aspect ratio
        ax1.set_title(r'\textbf{Tilted Side View}', fontsize=12, y=1.02)
        ax1.set_xlabel(r'\textbf{Length}')
        ax1.set_ylabel(r'\textbf{Width}')
        ax1.set_zlabel(r'\textbf{Elevation}')

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(self.X, self.Y, self.Z, cmap='Greens', edgecolor='red', linewidth=0.5)
        ax2.view_init(elev=0, azim=90)
        ax2.set_xlim(0, self.params['length'])
        ax2.set_ylim(0, self.params['width'])
        ax2.set_zlim(0, 200)  # Adjust elevation from 0 to 200
        ax2.set_box_aspect([1, 1, 1])  # Maintain equal aspect ratio
        ax2.set_title(r'\textbf{Perpendicular Side View}', fontsize=12, y=1.02)

        ax2.set_xlabel(r'\textbf{Length}')
        ax2.set_ylabel(r'\textbf{Width}')
        ax2.set_zlabel(r'\textbf{Elevation}')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    params = {
        'length': 200,
        'width': 200,
        'base_height': 0,
        'slope_x': 0.5,
        'slope_y': 0,
        'humps': [
            {'center': (50, 50), 'radius': 15, 'height': 10},
            {'center': (150, 150), 'radius': 10, 'height': 10}
        ],
        'cavities': [
            {'center': (100, 100), 'radius': 20, 'depth': 10},
            {'center': (30, 170), 'radius': 12, 'depth': 20}
        ]
    }

    surface = SurfaceWithFeatures(params)
    surface.generate_surface()
    surface.save_mesh()
    surface.plot_two_views()


