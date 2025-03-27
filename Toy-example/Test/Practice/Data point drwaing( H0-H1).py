import matplotlib.pyplot as plt

# Apply LaTeX Formatting for Matplotlib
plt.rcParams.update({'text.usetex': True})  # Enable LaTeX rendering
plt.rcParams.update({'font.family': 'serif'})  # Use serif fonts
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})  
plt.rcParams.update({'font.size': 10})  # Standard font size
plt.rcParams.update({'mathtext.rm': 'serif'})  # Use serif fonts for math text
plt.rcParams.update({'mathtext.fontset': 'custom'})  # Use custom math fonts

# Plot
fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)  # Set figure size and DPI directly in subplots
plt.tight_layout(pad=1.5)  # Apply tight layout with no padding

# Data points
x = [9.4841538660995, 9.26470523171643, 9.45140598614037, 9.41268216239237, 9.44807061046547, 9.37618875152874]
y = [6.80312288450234, 6.63823350095655, 6.9928931490743, 7.03893509287341, 6.86687641191988, 6.9889303875193]

# Blue dots only, no connecting lines
ax.plot(x, y, 'bo', linestyle='None')

# Adding solid grid lines
ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)

# Adding ticks to both axes with specific font size
ax.tick_params(axis='both', labelsize=8)

# X-axis label and Y-axis label with specified font size using LaTeX rendering
ax.set_xlabel(r'H0-entropy', fontsize=10)
ax.set_ylabel(r'H1-entropy', fontsize=10)

# Title with LaTeX rendering
#ax.set_title(r'\textbf{Data Points with LaTeX Labels}', fontsize=12)

# Display the plot
plt.show()
