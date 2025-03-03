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

# Create a loop for individual feature plots against all others
for i, feature_x in enumerate(features):
    fig, axes = plt.subplots(int(np.ceil(np.sqrt(num_features))), 
                             int(np.ceil(np.sqrt(num_features))), figsize=(12, 12))
    axes = axes.flatten()

    # Plot feature_x against all other features
    for j, feature_y in enumerate(features):
        ax = axes[j]
        sc = ax.scatter(df[feature_x], df[feature_y], c=df[feature_x], cmap="viridis", 
                        edgecolors="black", linewidth=0.5, marker='o', alpha=0.7)
        ax.set_xlabel(feature_x, fontsize=8)
        ax.set_ylabel(feature_y, fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)

    # Hide unused subplots
    for k in range(j + 1, len(axes)):
        fig.delaxes(axes[k])

    plt.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.02, pad=0.05, label=feature_x)
    plt.tight_layout()
    plt.suptitle(f"Feature-to-Feature Scatter Plots ({feature_x} vs others)", fontsize=14)
    plt.show()

# Final plot: All features against Index
fig, axes = plt.subplots(int(np.ceil(np.sqrt(num_features))), 
                         int(np.ceil(np.sqrt(num_features))), figsize=(12, 12))
axes = axes.flatten()

for ax, feature in zip(axes, features):
    sc = ax.scatter(df["Index"], df[feature], c=df["Index"], cmap="viridis", 
                    edgecolors="black", linewidth=0.5, marker='o', alpha=0.7)
    ax.set_xlabel("Index", fontsize=8)
    ax.set_ylabel(feature, fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=7)

# Hide unused subplots
for i in range(len(features), len(axes)):
    fig.delaxes(axes[i])

plt.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.02, pad=0.05, label="Index")
plt.tight_layout()
plt.suptitle("Feature vs Index Plots", fontsize=14)
plt.show()
