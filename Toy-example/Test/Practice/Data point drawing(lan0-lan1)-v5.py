import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Apply LaTeX Formatting for Matplotlib
plt.rcParams.update({'text.usetex': True})  
plt.rcParams.update({'font.family': 'serif'})  
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})  
plt.rcParams.update({'font.size': 10})  
plt.rcParams.update({'mathtext.rm': 'serif'})  
plt.rcParams.update({'mathtext.fontset': 'custom'})  

# ------------------------
# 1) Define your data
# ------------------------
data_points = {
    "2021-06": (4.17823208847893, 3.40024643467844),
    "2021-08": (15.5241577072973, 2.62756461081497),
    "2022-02": (5.37247679114185, 1.19919571170783),
    "2022-10": (22.9935928223699, 1.47650576747287),
    "2023-08": (4.63386894290955, 1.63556146074716),
    "2023-09": (28.4173955407928, 1.31216463232126)
}

# Convert dates to numeric values for colormap normalization
dates = list(data_points.keys())
x, y = zip(*data_points.values())

# Create a numeric mapping for dates as evenly spaced floats between 0 and 1
sorted_dates = sorted(data_points.keys())
date_order = {date: i / (len(sorted_dates) - 1) for i, date in enumerate(sorted_dates)}
date_values = [date_order[date] for date in dates]

# ------------------------
# 2) Plotting the data
# ------------------------
fig, ax = plt.subplots(figsize=(5, 4), dpi=300)  
plt.tight_layout(pad=1.7)

# Continuous color normalization and colormap
norm = Normalize(vmin=0, vmax=1)
colors = cm.viridis(norm(date_values))

# Scatter plot with Viridis colormap
sc = ax.scatter(x, y, c=date_values, cmap='viridis', norm=norm, s=50, zorder=3)

# Adjusted spacing for dates to the left of markers
for i, date in enumerate(dates):
    x_offset = 0.08 * (x[-1] - x[0])  # Increased left offset to avoid overlap
    y_offset = 0.0  # No vertical offset
    ax.text(x[i] + x_offset, y[i], f'{date}', fontsize=10, ha='left', va='center', zorder=5)

# ------------------------
# 3) Add a continuous colorbar with correct date mapping
# ------------------------
cbar = plt.colorbar(sc, ax=ax)
tick_positions = [norm(date_order[date]) for date in sorted_dates]
cbar.set_ticks(tick_positions)
cbar.set_ticklabels(sorted_dates)
cbar.ax.set_ylabel('Date', fontsize=12)

# Grid and formatting
ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7, zorder=1)
ax.tick_params(axis='both', labelsize=10)

# Setting x and y limits with increased margin
x_min, x_max = min(x), max(x)
y_min, y_max = min(y), max(y)
x_margin = (x_max - x_min) * 0.4  # 20% margin
y_margin = (y_max - y_min) * 0.2  # 20% margin
ax.set_xlim(x_min - x_margin, x_max + x_margin)
ax.set_ylim(y_min - y_margin, y_max + y_margin)

# Setting ticks at both ends
x_ticks = np.linspace(x_min - x_margin, x_max + x_margin, num=5)
y_ticks = np.linspace(y_min - y_margin, y_max + y_margin, num=5)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

# x-y labels
ax.set_xlabel(r'H0-landscape', fontsize=12)
ax.set_ylabel(r'H1-landscape', fontsize=12)

# Display the plot
plt.show()
