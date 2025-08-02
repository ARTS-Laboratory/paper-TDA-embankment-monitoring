import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# -----------------------------
# Load Excel file
# -----------------------------
file_path = "C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/Abnormalities_Features-V4.xlsx"  # Adjusted to match uploaded path
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

# Convert 'Index' to numeric
df['Index'] = pd.to_numeric(df['Index'], errors='coerce')
df.dropna(subset=['Index'], inplace=True)

# Normalize numeric features
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# -----------------------------
# List and number available columns
# -----------------------------
available_columns = [col for col in df.columns if col != 'Index']
print("\n✅ Available columns to plot:")
for i, col in enumerate(available_columns):
    print(f"{i+1}. {col}")

# -----------------------------
# Let user select by number or name
# -----------------------------
print("\n👉 You can choose:")
print("  - Column numbers (e.g., 1, 3, 5)")
print("  - OR full column names (e.g., persistence entropy (H0), Betti (H1))")
user_input = input("Enter your selection: ")

# Try parsing as numbers first
try:
    selection = [int(x.strip()) for x in user_input.split(',')]
    columns_to_plot = [available_columns[i - 1] for i in selection if 0 < i <= len(available_columns)]
except ValueError:
    columns_to_plot = [x.strip() for x in user_input.split(',') if x.strip() in available_columns]

# -----------------------------
# Plot if valid columns selected
# -----------------------------
if not columns_to_plot:
    print("⚠️ No valid columns selected. Exiting...")
else:
    print("\n📊 Plotting selected columns:", columns_to_plot)

    plt.figure(figsize=(6.5, 4), dpi=300)

    for col in columns_to_plot:
        try:
            plt.plot(df['Index'], df[col], label=col, marker='o', linestyle='-', linewidth=0.8, markersize=4)
        except Exception as e:
            print(f"⚠️ Skipping '{col}' due to error: {e}")

    plt.xlabel(r"damage index", fontsize=10)
    plt.ylabel(r"normalized feature value", fontsize=10)
    plt.xlim(-52, 52)
    plt.xticks(np.arange(-50, 55, 10), fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.legend(fontsize=6.8, loc='lower left', framealpha=1, markerscale=0.6, handlelength=0.8)
    plt.tight_layout(pad=0)
    plt.show()


