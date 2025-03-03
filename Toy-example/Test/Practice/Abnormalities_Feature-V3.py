# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

# File loading
file_path = "Abnormalities_Features - V3.xlsx"  # Change this to your actual file path
xls = pd.ExcelFile(file_path)

# First sheet selected automatically
df = pd.read_excel(xls, sheet_name=xls.sheet_names[0], header=None)  

# Headers identified automatically
header_row_index = df[df.apply(lambda row: row.astype(str).str.contains("Index", case=False, na=False).any(), axis=1)].index[0]

# Extract column names
df.columns = df.iloc[header_row_index]
df = df[(header_row_index + 1):].reset_index(drop=True)  # Remove previous non-data rows

# Identify the Index column
df.rename(columns={df.columns[0]: "Index"}, inplace=True)

# Convert all values to numeric to avoid NaN values
df = df.apply(pd.to_numeric, errors='coerce')

# Remove empty columns
df.dropna(axis=1, how='all', inplace=True)

# Drop rows where Index is NaN (in case of missing headers)
df.dropna(subset=["Index"], inplace=True)

# Extract features (all columns except Index)
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
        ax.scatter(df[feature_x], df[feature_y], 
                   edgecolors=plt.cm.viridis((df[feature_x] - df[feature_x].min()) / (df[feature_x].max() - df[feature_x].min())), 
                   facecolors='none', linewidth=1.2, marker='o', alpha=0.8)
        ax.set_xlabel(feature_x, fontsize=10)
        ax.set_ylabel(feature_y, fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)

    # Hide unused subplots
    for k in range(j + 1, len(axes)):
        fig.delaxes(axes[k])

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.4, wspace=0.4)  # Increased spacing
    plt.suptitle(f"{feature_y} Plotted Over Other Features", fontsize=16, y=0.98)  # Adjusted title position
    plt.show()

# **17th Plot: Features vs. Index (Scatter Only)**
grid_size = int(np.ceil(np.sqrt(num_features)))
fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
axes = axes.flatten()

for ax, feature in zip(axes, features):
    ax.scatter(df["Index"], df[feature], 
               edgecolors=plt.cm.viridis((df["Index"] - df["Index"].min()) / (df["Index"].max() - df["Index"].min())), 
               facecolors='none', linewidth=1.2, marker='o', alpha=0.8)
    
    ax.set_xlabel("Index", fontsize=10)
    ax.set_ylabel(feature, fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)

# Hide unused subplots
for i in range(len(features), len(axes)):
    fig.delaxes(axes[i])

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.4, wspace=0.4)  
plt.suptitle("Feature vs Index Plots (Scatter Only)", fontsize=16, y=0.98)  
plt.show()

# **18th Plot: Features vs. Index (Scatter + Line)**
fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
axes = axes.flatten()

for ax, feature in zip(axes, features):
    # Scatter plot (hollow markers)
    ax.scatter(df["Index"], df[feature], 
               edgecolors=plt.cm.viridis((df["Index"] - df["Index"].min()) / (df["Index"].max() - df["Index"].min())), 
               facecolors='none', linewidth=1.2, marker='o', alpha=0.8)
    
    # Line plot connecting the points
    ax.plot(df["Index"], df[feature], color="black", linewidth=0.8, linestyle='-', alpha=0.7)
    
    ax.set_xlabel("Index", fontsize=10)
    ax.set_ylabel(feature, fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)

# Hide unused subplots
for i in range(len(features), len(axes)):
    fig.delaxes(axes[i])

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.4, wspace=0.4)  
plt.suptitle("Feature vs Index Plots (Scatter + Line)", fontsize=16, y=0.98)  
plt.show()
