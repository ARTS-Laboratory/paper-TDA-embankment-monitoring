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
    "2021-06": (9.4841538660995, 6.80312288450234),
    "2021-08": (9.26470523171643, 6.63823350095655),
    "2022-02": (9.45140598614037, 6.9928931490743),
    "2022-10": (9.41268216239237, 7.03893509287341),
    "2023-08": (9.44807061046547, 6.86687641191988),
    "2023-09": (9.37618875152874, 6.9889303875193)
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
fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)  
plt.tight_layout(pad=3)

# Continuous color normalization and colormap
norm = Normalize(vmin=0, vmax=1)
colors = cm.viridis(norm(date_values))

# Scatter plot with Viridis colormap
sc = ax.scatter(x, y, c=date_values, cmap='viridis', norm=norm, s=50, zorder=3)

# Adjusted spacing for dates to the left of markers
for i, date in enumerate(dates):
    x_offset = 0.08 * (x[-1] - x[0])  # Increased left offset to avoid overlap
    y_offset = 0.0  # No vertical offset
    ax.text(x[i] + x_offset, y[i], f'{date}', fontsize=10, ha='right', va='center', zorder=5)

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
ax.set_xlabel(r'H0-entropy', fontsize=12)
ax.set_ylabel(r'H1-entropy', fontsize=12)

# Display the plot
plt.show()
