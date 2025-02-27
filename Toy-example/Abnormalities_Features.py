import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import math

# Load the Excel file.

df = pd.read_excel("Abnormalities_Features.xlsx", sheet_name="Table", header=2)
print("Initial columns:", df.columns.tolist())
print("Initial shape:", df.shape)

# Clean  the datafram if any column is unnamed.

df = df.loc[:, ~df.columns.str.contains("Unnamed")]


# After cleaning up the dataframe presentes final dataframe
print("Final columns:", df.columns.tolist())
print("Final shape:", df.shape)

# Determine the number of columns features in the dynamic way.
#    This is important because we want our plots to adapt if the number of features changes.
num_cols = df.shape[1]
print("Number of features (columns):", num_cols)

# Compute the grid size using the number of features for plotting.

grid_size = math.ceil(math.sqrt(num_cols))
print("Grid size:", grid_size, "x", grid_size)

# in this loop the fixed Y-axis.
#    For each iteration, one column is set as the Y-axis, and we plot scatter plots of that Y versus each feature (as X).
for i in range(num_cols):
    y_col = df.columns[i]  # This column is fixed as the Y-axis for this figure.
    # Create a new figure with a grid of subplots that is grid_size x grid_size.
    fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(15, 12))
    # Set a main title for the figure indicating which feature is used for the Y-axis.
    fig.suptitle(f"Scatter Plots: Y = '{y_col}' vs. All Features", fontsize=16)
    
    # Flatten the axes array so we can access subplots with a single index.
    axes = axes.flatten()
    
    # 6) Loop over each column as the X-axis.
    for j in range(num_cols):
        x_col = df.columns[j]  #  the x-axis in the subplot.
        ax = axes[j]           # selecting the j-th subplot in the grid.
        
        # Normalize the X-axis data for a continuous colormap.
        # This maps the range of x_col values to [0, 1].
        x_values = df[x_col]
        norm = mcolors.Normalize(vmin=x_values.min(), vmax=x_values.max())
        # Choose the "viridis" colormap.
        cmap = cm.get_cmap("viridis")
        # Map each x value to a color.
        edge_colors = cmap(norm(x_values))
        
        
        ax.scatter(df[x_col], df[y_col],
                   s=80,
                   marker='o',
                   facecolors='none',
                   edgecolors=edge_colors,
                   linewidths=1.2,
                   alpha=0.7)
        
        # Set a small title for the subplot showing which column is used as the X-axis.
        ax.set_title(f"X = {x_col}", fontsize=9)
        # Always display tick labels (we set their font size for clarity).
        ax.tick_params(axis='both', labelsize=7)
    
    # 7) Hide any extra subplots if the grid has more cells than needed.
    for k in range(num_cols, grid_size * grid_size):
        axes[k].axis("off")
    
    # Adjust the layout so titles and labels don't overlap.
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
