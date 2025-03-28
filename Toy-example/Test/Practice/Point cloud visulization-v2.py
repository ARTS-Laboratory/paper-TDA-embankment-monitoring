import matplotlib.pyplot as plt
import numpy as np  # Importing numpy

# ✅ Apply LaTeX Formatting for Matplotlib
plt.rcParams.update({'text.usetex': True})  
plt.rcParams.update({'font.family': 'serif'})  
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})  
plt.rcParams.update({'font.size': 10})  
plt.rcParams.update({'mathtext.rm': 'serif'})  
plt.rcParams.update({'mathtext.fontset': 'custom'})  

# Plot
fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)  
plt.tight_layout(pad=2.3)  

# Data points
x = [9.4841538660995, 9.26470523171643, 9.45140598614037, 9.41268216239237, 9.44807061046547, 9.37618875152874]
y = [6.80312288450234, 6.63823350095655, 6.9928931490743, 7.03893509287341, 6.86687641191988, 6.9889303875193]

# Blue dots only
ax.plot(x, y, 'bo', linestyle='None')

# Solid grid
ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)

# Ticks added for both axes
ax.tick_params(axis='both', labelsize=8)

# Setting x and y limits with a small margin to ensure visibility
x_min, x_max = min(x), max(x)
y_min, y_max = min(y), max(y)

# Extend the limits slightly to ensure points are not at the edge
x_margin = (x_max - x_min) * 0.05  # 5% margin on both sides
y_margin = (y_max - y_min) * 0.05  # 5% margin on both sides

ax.set_xlim(x_min - x_margin, x_max + x_margin)
ax.set_ylim(y_min - y_margin, y_max + y_margin)

# Setting ticks to include the exact min and max values plus intermediate ticks
x_ticks = np.linspace(x_min - x_margin, x_max + x_margin, num=5)
y_ticks = np.linspace(y_min - y_margin, y_max + y_margin, num=5)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

# x-y label
ax.set_xlabel(r'H0-entropy', fontsize=10)
ax.set_ylabel(r'H1-entropy', fontsize=10)

# Display the plot
plt.show()
