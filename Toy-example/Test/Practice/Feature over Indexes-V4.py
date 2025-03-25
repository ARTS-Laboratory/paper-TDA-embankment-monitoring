import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# File path
file_path = "C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Toy-example/Data/Abnormalities_Features-V4.xlsx"

# LaTeX Formatting for Plots
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})
plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})

# Load the Excel file
df = pd.read_excel(file_path)

# Clean the dataframe to properly set the header and index
df.columns = df.iloc[0]  # Set the header as the first row
df = df[1:]  # Remove the header row from data

# Remove special characters from column names
df.columns = [re.sub(r'\s+', ' ', str(x).strip()) for x in df.columns]

# All column names to understand the issue
print("\nAll Column Names from the DataFrame:")
print(df.columns.tolist())

# Index column to numeric
df['Index'] = pd.to_numeric(df['Index'], errors='coerce')
df.dropna(subset=['Index'], inplace=True)

# Normalize the Y-axis for each column (scaling between 0 and 1)
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Automatically detect all numerical columns except 'Index'
available_columns = [col for col in df.columns if col != 'Index']
print("\nAvailable columns to plot:", available_columns)

# Specify the desired columns in the code
columns_to_plot = ['persistence entropy (H0)', 'persistence entropy (H1)', 'landscape (H0)', 'landscape (H1)',
                   'persistence image (H0)', 'persistence image (H1)', 
                   'Betti (H0)', 'Betti (H1)', 'heat (H0)', 'heat (H1)']

# Validate selected columns after cleaning
columns_to_plot = [col for col in columns_to_plot if col in available_columns]
if not columns_to_plot:
    print("No valid columns selected. Exiting...")
else:
    print("\nSelected columns to plot:", columns_to_plot)

    # Adjust figure size and DPI
    plt.figure(figsize=(6.5, 4), dpi=300)

    # Plot each feature vs index with connecting lines and markers
    for col in columns_to_plot:
        try:
            plt.plot(df['Index'], df[col], label=col, marker='o', linestyle='-', linewidth=0.8, markersize=4)
        except Exception as e:
            print(f"Skipping column '{col}' due to error: {e}")

    # X-axis label and Y-axis label
    plt.xlabel(r"damage index", fontsize=10)  # X-axis label changed
    plt.ylabel(r"normalized feature value", fontsize=10)  # Y-axis normalized

    # Increase number of X-axis ticks and set xlim
    plt.xlim(-52, 52)  # Set x-axis limit to -52 and 52
    plt.xticks(np.arange(-50, 55, 10), fontsize=8)  # Set ticks at interval of 10

    # Adjust grid and alpha for lighter grid lines
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # Adjusted legend: smaller font size, transparent background, reduced spacing
    plt.legend(fontsize=6.8, loc='lower left', framealpha=0.8, markerscale=0.6, handlelength=0.8)

    # Remove extra whitespace
    plt.tight_layout(pad=0)

    # Show the plot
    plt.show()
