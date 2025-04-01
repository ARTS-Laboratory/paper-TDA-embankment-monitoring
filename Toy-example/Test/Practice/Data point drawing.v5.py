import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from datetime import datetime
from matplotlib.ticker import FormatStrFormatter

# LaTeX Formatting for Matplotlib

plt.rcParams.update({'text.usetex': True})  
plt.rcParams.update({'font.family': 'serif'})  
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})  
plt.rcParams.update({'font.size': 10})  
plt.rcParams.update({'mathtext.rm': 'serif'})  
plt.rcParams.update({'mathtext.fontset': 'custom'})  

# data points

data_points = {
        "2021-06": (1042.119579422192, 127.44319210344881),
        "2021-08": (1145.497867412144, 145.93548588114402),
        "2022-02": (1263.9558479266682, 197.78123768565337),
        "2022-10": (1186.5536021646367, 179.78491398175026),
        "2023-08": (1190.6499139745038, 167.34823571451383),
        "2023-09": (1212.559339099928, 185.29545928554322)
}

# converting the date strings to datetime objects

date_format = "%Y-%m"
date_labels = list(data_points.keys())
date_objects = [datetime.strptime(date, date_format) for date in date_labels]

# converting dates to numeric values earliest days

min_date = min(date_objects)
date_days = np.array([(d - min_date).days for d in date_objects])

# (x, y) points
x, y = zip(*data_points.values())

# plotting the data

fig, ax = plt.subplots(figsize=(5, 4), dpi=300)  
plt.tight_layout(pad=1.7)

# normalize using actual date range (days)

norm = Normalize(vmin=np.min(date_days), vmax=np.max(date_days))
cmap = cm.viridis

# scatter plot

sc = ax.scatter(x, y, c=date_days, cmap=cmap, norm=norm, s=50, zorder=3)

# annotate points
for i, date in enumerate(date_labels):
    x_offset = 0.03 * (max(x) - min(x))
    ax.text(x[i] + x_offset, y[i], f'{date}', fontsize=10, ha='left', va='center', zorder=5)

# add fixed colorbar

cbar = plt.colorbar(sc, ax=ax)
tick_values = date_days
tick_labels = date_labels
cbar.set_ticks(tick_values)
cbar.set_ticklabels(tick_labels)
cbar.ax.set_ylabel('time', fontsize=10)
cbar.ax.tick_params(labelsize=9)              

# grid and formatting
ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7, zorder=1)
ax.tick_params(axis='both', labelsize=9)

# axis limits with margin
x_min, x_max = min(x), max(x)
y_min, y_max = min(y), max(y)
x_margin = (x_max - x_min) * 0.3
y_margin = (y_max - y_min) * 0.2
ax.set_xlim(x_min - x_margin, x_max + x_margin)
ax.set_ylim(y_min - y_margin, y_max + y_margin)


# Manually set axis limits
#ax.set_xlim(0, 35)      # X-axis domain
#ax.set_ylim(1.0, 4.0)    # Y-axis domain

# custom ticks
x_ticks = np.linspace(x_min - x_margin, x_max + x_margin, num=5)
y_ticks = np.linspace(y_min - y_margin, y_max + y_margin, num=5)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax.set_xlabel(r'H0-image', fontsize=10)
ax.set_ylabel(r'H1-image', fontsize=10)

plt.show()
