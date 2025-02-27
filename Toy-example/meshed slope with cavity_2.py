import numpy as np
import trimesh
import matplotlib.pyplot as plt

class SurfaceWithSmoothCircularCavity:
    def __init__(self, params=None):
        # Default parameters
        self.default_params = {
            'length': 200,           
            'width': 200,            
            'resolution': 300,       # mesh resolution
            'base_height': 0,        # surface based hight
            'slope_x': 0.05,         # slope along of x direction
            'slope_y': 0.02,         # slope along of y direction
            'cavity_radius': 20,     # cavity radious
            'cavity_center': (100, 100),  # cavity center 
            'cavity_depth': 10        # Depth of the cavity
        }

        # update above parameters
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)

    def generate_surface(self):
        """Generate a surface with an optional slope and smooth circular cavity."""
        # coordinate grid
        x = np.linspace(0, self.params['length'], self.params['resolution'])
        y = np.linspace(0, self.params['width'], self.params['resolution'])
        self.X, self.Y = np.meshgrid(x, y)

        # surface with slope
        self.Z = (self.params['base_height'] +
                  self.params['slope_x'] * self.X +
                  self.params['slope_y'] * self.Y)

        # hump parameters designation
        cavity_x, cavity_y = self.params['cavity_center']
        cavity_radius = self.params['cavity_radius']
        cavity_depth = self.params['cavity_depth']
        distance_from_center = np.sqrt((self.X - cavity_x)**2 + (self.Y - cavity_y)**2)

        # surface normal components 
        dz_dx = self.params['slope_x']
        dz_dy = self.params['slope_y']
        normal_magnitude = np.sqrt(1 + dz_dx**2 + dz_dy**2)

        #  surface normal vector normalizing
        normal = np.array([-dz_dx, -dz_dy, 1]) / normal_magnitude

        # cavity depth along the surface normal
        inside_cavity = distance_from_center <= cavity_radius
        adjusted_depth = cavity_depth * (1 - (distance_from_center[inside_cavity] / cavity_radius)**2)
        self.Z[inside_cavity] -= adjusted_depth * normal[2]  # Align cavity depth with the normal vector

    def save_mesh(self, filename="surface_with_smooth_circular_cavity.obj"):
        """Save the generated mesh to a file."""
        # mesh generating
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
        
        fig = plt.figure(figsize=(14, 6))

        # main view
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(self.X, self.Y, self.Z, cmap='Greens', edgecolor='red', linewidth=0.5)
        ax1.view_init(elev=20, azim=120)  # First perspective
        ax1.set_xlim(0, self.params['length'])
        ax1.set_ylim(0, self.params['width'])
        ax1.set_zlim(self.params['base_height'] - self.params['cavity_depth'], self.params['base_height'] + 5)
        ax1.set_box_aspect([self.params['length'], self.params['width'], self.params['cavity_depth']])
        ax1.set_title("main View")
        ax1.set_xlabel('length')
        ax1.set_ylabel('width')
        ax1.set_zlabel('elevation')

        # perpendicular view
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(self.X, self.Y, self.Z, cmap='Greens', edgecolor='red', linewidth=0.5)
        ax2.view_init(elev=0, azim=90)  # Perpendicular side view
        ax2.set_xlim(0, self.params['length'])
        ax2.set_ylim(0, self.params['width'])
        ax2.set_zlim(self.params['base_height'] - self.params['cavity_depth'], self.params['base_height'] + 5)
        ax2.set_box_aspect([self.params['length'], self.params['width'], self.params['cavity_depth']])
        ax2.set_title("perpendicular view")
        ax2.set_xlabel('length')
        ax2.set_ylabel('width')
        ax2.set_zlabel('elevation')

        plt.tight_layout()
        plt.show()

# new parametrs importing
if __name__ == "__main__":
    # Define custom parameters (optional)
    params = {
        'length': 200,
        'width': 200,
        'base_height': 0,
        'slope_x': 0.5,      # small slope along the X direction
        'slope_y': 0.01,     # small slope along the Y direction
        'cavity_radius': 10, # radius of the cavity
        'cavity_center': (100, 100),  # center of the cavity
        'cavity_depth': 10   # depth of the cavity
    }

    # surface generating
    surface = SurfaceWithSmoothCircularCavity(params)
    surface.generate_surface()

    # mesh generating
    surface.save_mesh()

    # Plot the two view
    surface.plot_two_views()


