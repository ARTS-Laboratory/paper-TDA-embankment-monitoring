import ripser
import numpy as np
from gtda.diagrams import PersistenceLandscape
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# Step 1: Generate a random point cloud (replace this with your data)
point_cloud = np.random.random((100, 3))

# Step 2: Compute persistence diagrams using Ripser
diagrams = ripser.ripser(point_cloud)['dgms']
print("Persistence diagrams:", diagrams)

# Step 3: Preprocess diagrams for giotto-tda
# Convert diagrams to a format compatible with giotto-tda
diagrams_gtda = []
for dim, diagram in enumerate(diagrams):
    if len(diagram) > 0:
        # Add a column for the homology dimension
        diagram_with_dim = np.hstack([diagram, np.ones((len(diagram), 1)) * dim])
        diagrams_gtda.append(diagram_with_dim)
# Combine all diagrams into a single array
diagrams_gtda = np.vstack(diagrams_gtda)
print("Preprocessed diagrams:", diagrams_gtda)

# Step 4: Compute persistence landscapes using giotto-tda
landscape = PersistenceLandscape()
landscapes = landscape.fit_transform([diagrams_gtda])
print("Persistence landscapes:", landscapes)

# Step 5: Visualize the persistence landscape in 3D

# Option 1: Using matplotlib
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i, landscape in enumerate(landscapes[0]):
    t = np.arange(len(landscape))  # Time axis
    ax.plot(t, [i] * len(landscape), landscape, label=f'Landscape {i+1}')
ax.set_xlabel('Time')
ax.set_ylabel('Landscape Index')
ax.set_zlabel('Amplitude')
ax.set_title('3D Persistence Landscape (Matplotlib)')
plt.legend()
plt.show()

# Option 2: Using plotly (interactive)
fig = go.Figure()
for i, landscape in enumerate(landscapes[0]):
    t = np.arange(len(landscape))  # Time axis
    fig.add_trace(go.Scatter3d(
        x=t,
        y=[i] * len(landscape),
        z=landscape,
        mode='lines',
        name=f'Landscape {i+1}'
    ))
fig.update_layout(
    scene=dict(
        xaxis_title='Time',
        yaxis_title='Landscape Index',
        zaxis_title='Amplitude'
    ),
    title='3D Persistence Landscape (Plotly)'
)
fig.show()
