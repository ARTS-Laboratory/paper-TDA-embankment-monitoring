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


# data point

data_points = {
    "2021-06": (4.17823208847893, 3.40024643467844),
    "2021-08": (15.5241577072973, 2.62756461081497),
    "2022-02": (5.37247679114185, 1.19919571170783),
    "2022-10": (22.9935928223699, 1.47650576747287),
    "2023-08": (4.63386894290955, 1.63556146074716),
    "2023-09": (28.4173955407928, 1.31216463232126)
}

# converting the date strings to datetime objects
date_format = "%Y-%m"
date_labels = list(data_points.keys())
date_objects = [datetime.strptime(date, date_format) for date in date_labels]

# converting dates to numeric values earliest days
min_date = min(date_objects)
date_days = np.array([(d - min_date).days for d in date_objects])

#  (x, y) points
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

ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7, zorder=1)
ax.tick_params(axis='both', labelsize=9)

# axis limits
ax.set_xlim(0, 35)       
ax.set_ylim(1.0, 4.0)    

# custom ticks based on manual limits
ax.set_xticks(np.linspace(0, 35, num=6))
ax.set_yticks(np.linspace(1.0, 4.0, num=5))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax.set_xlabel(r'H0-landscape', fontsize=10)
ax.set_ylabel(r'H1-landscape', fontsize=10)

plt.show()
