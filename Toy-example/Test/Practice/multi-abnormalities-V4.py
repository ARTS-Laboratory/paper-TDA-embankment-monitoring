# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh

# Apply LaTeX Formatting for Matplotlib Plots
plt.rcParams.update({'text.usetex': True})  # Enable LaTeX rendering
plt.rcParams.update({'font.family': 'serif'})  # Use serif fonts
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})  
plt.rcParams.update({'font.size': 12})  # Standard font size
plt.rcParams.update({'mathtext.rm': 'serif'})  # Use serif fonts for math text
plt.rcParams.update({'mathtext.fontset': 'custom'})  # Use custom math fonts

class SurfaceWithFeatures:
    def __init__(self, params=None):
        """Initialize the surface with default parameters and user-defined settings."""
        self.default_params = {
            'length': 200,  
            'width': 200,   
            'resolution': 300,  
            'base_height': 0,  
            'slope_x': 2,  
            'slope_y': 2,  
            'humps': [],  
            'cavities': []  
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

    def plot_large_3D(self):
        """Generate a 3D plot in a large, but resizable window."""

        # Set figure size (Large but not fullscreen)
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface
        ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', edgecolor='k', linewidth=0.5)

        # Set labels and title with LaTeX formatting
        ax.set_xlabel(r'\textbf{Length}', fontsize=14, labelpad=10)
        ax.set_ylabel(r'\textbf{Width}', fontsize=14, labelpad=10)
        ax.set_zlabel(r'\textbf{Elevation}', fontsize=14, labelpad=10)
        ax.set_title(r'\textbf{Slope with Different Abnormalities}', fontsize=16, pad=8)

        # Try to maximize window in a resizable mode
        manager = plt.get_current_fig_manager()
        try:
            manager.window.state('zoomed')  # Windows (Tkinter Backend)
        except AttributeError:
            try:
                manager.resize(*manager.window.maxsize())  # Alternative full-screen approach
            except AttributeError:
                print("Fullscreen mode is not supported in this environment.")

        # Set proper view angle
        ax.view_init(elev=30, azim=135)

        # Show plot in a way that allows easy closing
        plt.show(block=True)  # Ensure plot remains open until manually closed

if __name__ == "__main__":
    params = {
        'length': 200,
        'width': 200,
        'base_height': 0,
        'slope_x': 0.5,
        'slope_y': 0,
        'humps': [
            {'center': (50, 50), 'radius': 15, 'height': 10},
            {'center': (150, 150), 'radius': 10, 'height': 10},
            {'center': (150, 50), 'radius': 30, 'height': 20}
        ],
        'cavities': [
            {'center': (100, 100), 'radius': 20, 'depth': 10},
            {'center': (30, 170), 'radius': 12, 'depth': 20}
        ]
    }

    surface = SurfaceWithFeatures(params)
    surface.generate_surface()
    surface.save_mesh()
    surface.plot_large_3D()
