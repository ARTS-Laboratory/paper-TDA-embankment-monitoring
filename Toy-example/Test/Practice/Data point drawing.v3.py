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
    "2021-06": (9.4841538660995, 6.80312288450234),
    "2021-08": (9.26470523171643, 6.63823350095655),
    "2022-02": (9.45140598614037, 6.9928931490743),
    "2022-10": (9.41268216239237, 7.03893509287341),
    "2023-08": (9.44807061046547, 6.86687641191988),
    "2023-09": (9.37618875152874, 6.9889303875193)
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

# axis limits

ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7, zorder=1)
ax.tick_params(axis='both', labelsize=9)

# Manually set axis limits
ax.set_xlim(9.25, 9.55)       
ax.set_ylim(6.6, 7.1)    

# custom ticks based on manual limits
ax.set_xticks(np.linspace(9.25, 9.55, num=6))
ax.set_yticks(np.linspace(6.6, 7.1, num=6))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


ax.set_xlabel(r'H0-entropy', fontsize=10)
ax.set_ylabel(r'H1-entropy', fontsize=10)


plt.show()
