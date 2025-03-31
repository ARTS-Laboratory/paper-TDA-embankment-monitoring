import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from datetime import datetime
from matplotlib.ticker import FormatStrFormatter

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
        "2021-06": (1346.64323338739, 126.184424383657),
        "2021-08": (1508.54606107209, 119.70973882858),
        "2022-02": (1693.29977332732, 170.474344017366),
        "2022-10": (1792.4018331633, 167.434212672425),
        "2023-08": (1521.36355843286, 150.426334756123),
        "2023-09": (1654.90638742224, 158.660833851576)
}

# Convert date strings to datetime objects
date_format = "%Y-%m"
date_labels = list(data_points.keys())
date_objects = [datetime.strptime(date, date_format) for date in date_labels]

# Convert dates to numeric values (days since earliest)
min_date = min(date_objects)
date_days = np.array([(d - min_date).days for d in date_objects])

# Extract (x, y) points
x, y = zip(*data_points.values())

# ------------------------
# 2) Plotting the data
# ------------------------
fig, ax = plt.subplots(figsize=(5, 4), dpi=300)  
plt.tight_layout(pad=1.7)

# Normalize using actual date range (days)
norm = Normalize(vmin=np.min(date_days), vmax=np.max(date_days))
cmap = cm.viridis

# Scatter plot with real-time mapped colors
sc = ax.scatter(x, y, c=date_days, cmap=cmap, norm=norm, s=50, zorder=3)

# Annotate points
for i, date in enumerate(date_labels):
    x_offset = 0.03 * (max(x) - min(x))
    ax.text(x[i] + x_offset, y[i], f'{date}', fontsize=10, ha='left', va='center', zorder=5)

# ------------------------
# 3) Add fixed colorbar
# ------------------------
cbar = plt.colorbar(sc, ax=ax)
tick_values = date_days
tick_labels = date_labels
cbar.set_ticks(tick_values)
cbar.set_ticklabels(tick_labels)
cbar.ax.set_ylabel('time', fontsize=10)
cbar.ax.tick_params(labelsize=9)               # Font size for tick labels

# Grid and formatting
ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7, zorder=1)
ax.tick_params(axis='both', labelsize=9)

# Axis limits with margin
x_min, x_max = min(x), max(x)
y_min, y_max = min(y), max(y)
x_margin = (x_max - x_min) * 0.3
y_margin = (y_max - y_min) * 0.2
ax.set_xlim(x_min - x_margin, x_max + x_margin)
ax.set_ylim(y_min - y_margin, y_max + y_margin)


# Manually set axis limits
#ax.set_xlim(0, 35)      # X-axis domain
#ax.set_ylim(1.0, 4.0)    # Y-axis domain

# Custom ticks
x_ticks = np.linspace(x_min - x_margin, x_max + x_margin, num=5)
y_ticks = np.linspace(y_min - y_margin, y_max + y_margin, num=5)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Axis labels
ax.set_xlabel(r'H0-Beti', fontsize=10)
ax.set_ylabel(r'H1-Beti', fontsize=10)

# Show plot
plt.show()
