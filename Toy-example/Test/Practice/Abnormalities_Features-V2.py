# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Load the Excel file
file_path = "Abnormalities_Features - V2.xlsx"  # Change this to your actual file path
xls = pd.ExcelFile(file_path)

# Load the sheet
df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])  # Load the first sheet dynamically

# Extract the proper header row (row index 2)
df.columns = df.iloc[2]  # Set the actual column names
df = df[3:].reset_index(drop=True)  # Remove the previous header rows

# Rename columns for clarity
df.rename(columns={df.columns[0]: "Index"}, inplace=True)

# Ensure all values are numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Drop any completely empty columns
df.dropna(axis=1, how='all', inplace=True)

# Extract features (excluding the "Index" column)
features = df.columns[1:]
num_features = len(features)

# Create a loop for individual feature plots (Y-axis) over all other features (X-axis)
for i, feature_y in enumerate(features):
    grid_size = int(np.ceil(np.sqrt(num_features)))  # Adjusted for better layout
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))  # Increased figure size
    axes = axes.flatten()

    # Plot feature_y against all other features (X-axis)
    for j, feature_x in enumerate(features):
        ax = axes[j]
        ax.scatter(df[feature_x], df[feature_y], edgecolors=plt.cm.viridis((df[feature_x] - df[feature_x].min()) / (df[feature_x].max() - df[feature_x].min())), 
                   facecolors='none', linewidth=1.2, marker='o', alpha=0.8)
        ax.set_xlabel(feature_x, fontsize=10)  # Adjusted font size
        ax.set_ylabel(feature_y, fontsize=10)  # Adjusted font size
        ax.tick_params(axis='both', which='major', labelsize=8)

    # Hide unused subplots
    for k in range(j + 1, len(axes)):
        fig.delaxes(axes[k])

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.4, wspace=0.4)  # Increased spacing
    plt.suptitle(f"{feature_y} Plotted Over Other Features", fontsize=16, y=0.98)  # Adjusted title position
    plt.show()

# Final plot: All features over the Index
grid_size = int(np.ceil(np.sqrt(num_features)))
fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))  # Adjusted figure size
axes = axes.flatten()

for ax, feature in zip(axes, features):
    ax.scatter(df["Index"], df[feature], edgecolors=plt.cm.viridis((df["Index"] - df["Index"].min()) / (df["Index"].max() - df["Index"].min())), 
               facecolors='none', linewidth=1.2, marker='o', alpha=0.8)
    ax.set_xlabel("Index", fontsize=10)
    ax.set_ylabel(feature, fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)

# Hide unused subplots
for i in range(len(features), len(axes)):
    fig.delaxes(axes[i])

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.4, wspace=0.4)  # Increased spacing
plt.suptitle("Feature vs Index Plots", fontsize=16, y=0.98)  # Adjusted title position
plt.show()


