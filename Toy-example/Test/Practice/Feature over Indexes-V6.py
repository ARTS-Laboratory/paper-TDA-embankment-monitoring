import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# -----------------------------
# Load Excel file
# -----------------------------
file_path = "C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/Abnormalities_Features-V5.xlsx"
df = pd.read_excel(file_path)

# -----------------------------
# Plot formatting
# -----------------------------
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})
plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})

# -----------------------------
# Clean the data
# -----------------------------
df.columns = df.iloc[0]
df = df[1:]
df.columns = [re.sub(r'\s+', ' ', str(col).strip()) for col in df.columns]
df['Index'] = pd.to_numeric(df['Index'], errors='coerce')
df.dropna(subset=['Index'], inplace=True)

# Normalize numeric features
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# -----------------------------
# List available columns
# -----------------------------
available_columns = [col for col in df.columns if col != 'Index']
print("\nAvailable columns to plot:")
for i, col in enumerate(available_columns):
    print(f"{i+1}. {col}")

# -----------------------------
# Prompt user for 4 plots (each with 4 features)
# -----------------------------
all_selections = []
for plot_idx in range(4):
    user_input = input(f"\n Enter 4 column numbers for subplot ({chr(97+plot_idx)}): ").strip()
    try:
        selection = [int(x.strip()) for x in user_input.split(',') if 0 < int(x.strip()) <= len(available_columns)]
        if len(selection) != 4:
            raise ValueError
        selected_cols = [available_columns[i - 1] for i in selection]
        all_selections.append(selected_cols)
    except:
        print(" Invalid input. Please enter exactly 4 valid column numbers separated by commas.")
        exit()

# -----------------------------
# Plotting with captions and legend boxes
# -----------------------------
fig, axs = plt.subplots(2, 2, figsize=(7, 5.4), dpi=300, constrained_layout=True)
axs = axs.flatten()

for i, cols in enumerate(all_selections):
    ax = axs[i]

    for col in cols:
        ax.plot(df['Index'], df[col], label=col, marker='o', linestyle='-', linewidth=0.8, markersize=1)

    # Main plot settings
    ax.set_xlabel(r"damage index", fontsize=8)
    ax.set_ylabel(r"normalized feature value", fontsize=8)
    ax.set_xlim(-6, 6)
    ax.set_xticks(np.arange(-6, 7, 2))
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, color='gray', linestyle='-', linewidth=0.2, alpha=0.3)

    # Place legend above the plot inside its own box
    ax.legend(
        fontsize=6, loc='lower center', bbox_to_anchor=(0.5, 1),   # to move the legend box
        ncol=2, frameon=True, framealpha=1, borderpad=0.5, columnspacing=0.8
    )

    # Move subplot label (a), (b), etc. below the x-axis label
    ax.annotate(f"({chr(97+i)})", xy=(0.5, -0.48), xycoords='axes fraction',
                ha='center', va='center', fontsize=8)

# Adjust spacing
fig.subplots_adjust(
    wspace=0.2,   # horizontal space between plots
    hspace=0.8,  # vertical space between plots
    top=0.92,     # top margin of the entire figure
    bottom=0.08   # bottom margin of the entire figure
)  # wspace controls horizontal space between plots

plt.show()
