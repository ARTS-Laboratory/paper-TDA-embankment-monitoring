# Import required libraries
import numpy as np
import trimesh
import matplotlib.pyplot as plt

# Matplotlib style
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})


class GreekSurface:
    def __init__(self, params=None):
        self.default_params = {
            # Geometry
            'length': 200,
            'width': 200,
            'resolution': 300,
            'base_height': 0,

            # Slopes
            'slope_left': 0.6,
            'slope_right': 0.6,

            # Flat bottom
            'bottom_width': 50,

            # Free anomalies (THIS is what you use)
            'features': []
        }

        self.params = self.default_params.copy()
        if params:
            self.params.update(params)

    # --------------------------------------------------
    # Geometry generation
    # --------------------------------------------------
    def generate_surface(self):
        x = np.linspace(0, self.params['length'], self.params['resolution'])
        y = np.linspace(0, self.params['width'], self.params['resolution'])
        self.X, self.Y = np.meshgrid(x, y)

        x_center = self.params['length'] / 2
        flat_half = self.params['bottom_width'] / 2

        self.Z = np.zeros_like(self.X) + self.params['base_height']

        # Masks
        left_mask = self.X < (x_center - flat_half)
        flat_mask = (self.X >= (x_center - flat_half)) & (self.X <= (x_center + flat_half))
        right_mask = self.X > (x_center + flat_half)

        # Left slope
        self.Z[left_mask] += (
            self.params['slope_left'] *
            ((x_center - flat_half) - self.X[left_mask])
        )

        # Flat bottom
        self.Z[flat_mask] += 0.0

        # Right slope
        self.Z[right_mask] += (
            self.params['slope_right'] *
            (self.X[right_mask] - (x_center + flat_half))
        )

        # --------------------------------------------------
        # Free anomalies (humps & cavities anywhere)
        # --------------------------------------------------
        for feat in self.params['features']:
            cx, cy = feat['center']
            r = feat['radius']

            dist = np.sqrt((self.X - cx)**2 + (self.Y - cy)**2)
            mask = dist <= r

            if feat['type'] == 'hump':
                h = feat['height']
                self.Z[mask] += np.clip(
                    h * (1 - (dist[mask] / r)**2), 0, h
                )

            elif feat['type'] == 'cavity':
                d = feat['depth']
                self.Z[mask] -= np.clip(
                    d * (1 - (dist[mask] / r)**2), 0, d
                )

            else:
                raise ValueError(f"Unknown feature type: {feat['type']}")

    # --------------------------------------------------
    # Mesh export
    # --------------------------------------------------
    def save_mesh(self, filename="greek_surface.obj"):
        vertices = np.column_stack((
            self.X.flatten(),
            self.Y.flatten(),
            self.Z.flatten()
        ))

        faces = []
        res = self.params['resolution']
        for i in range(res - 1):
            for j in range(res - 1):
                v = i * res + j
                faces.append([v, v + 1, v + res])
                faces.append([v + 1, v + res + 1, v + res])

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(filename)
        print(f"Mesh saved as {filename}")

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    def plot(self):
        fig = plt.figure(figsize=(3.5, 4))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(
            self.X, self.Y, self.Z,
            color='lightgreen',
            edgecolor='red',
            linewidth=0.20
        )

        ax.view_init(elev=20, azim=120)
        ax.set_xlim(0, self.params['length'])
        ax.set_ylim(0, self.params['width'])
        ax.set_zlim(0, 30)
        ax.set_box_aspect([1, 1, 1])

        plt.tight_layout()
        plt.show()


# --------------------------------------------------
# USER DEFINITION AREA
# --------------------------------------------------
if __name__ == "__main__":
    params = {
        'length': 185,
        'width': 200,
        'resolution': 300,

        'bottom_width': 100,
        'slope_left': 0.6,
        'slope_right': 0.6,

        # ADD ANY FEATURES YOU WANT — ANYWHERE
        'features': [
            #{'type': 'cavity', 'center': (75, 75), 'radius': 10, 'depth': 6},
            {'type': 'hump',   'center': (130, 135), 'radius': 30, 'height': 6},
            {'type': 'hump',   'center': (60, 25), 'radius': 25, 'height': 3},
            {'type': 'hump',   'center': (60, 150), 'radius': 15, 'height': 2}
        ]
    }

    surface = GreekSurface(params)
    surface.generate_surface()
    surface.save_mesh("greek_surface.obj")
    surface.plot()
