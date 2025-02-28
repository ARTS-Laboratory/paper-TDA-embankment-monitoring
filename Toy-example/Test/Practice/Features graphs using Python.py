import pandas as pd
import os

# Ensure the file is in the current working directory
print("Current working directory:", os.getcwd())

# Load the Excel file with header from row 3 (0-indexed header=2)
df = pd.read_excel("Book1.xlsx", sheet_name="Table", header=2)
print("Columns in Excel file:", df.columns.tolist())

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(df["PE (H0)"], df["PE (H1)"], color='blue', label="PE (H1) vs. PE (H0)")
plt.xlabel("Persistence Entropy H0")
plt.ylabel("Persistence Entropy H1")
plt.title("Persistence Entropy: H1 vs. H0")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(df["NoP (H0)"], df["NoP (H1)"], color='green', label="NoP (H1) vs. NoP (H0)")
plt.xlabel("Number-of-Points H0")
plt.ylabel("Number-of-Points H1")
plt.title("Number-of-Points: H1 vs. H0")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(df["Bottleneck amp (H0)"], df["Bottleneck amp (H1)"], color='red', label="Bot (H1) vs. Bot (H0)")
plt.xlabel("Bottleneck amp H0")
plt.ylabel("Bottleneck amp H1")
plt.title("Bottleneck amp: H1 vs. H0")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(df["Wasserstein amp (H0)"], df["Wasserstein amp (H1)"], color='red', label="Was (H1) vs. Was (H0)")
plt.xlabel("Wasserstein amp H0")
plt.ylabel("Wasserstein amp H1")
plt.title("Wasserstein amp: H1 vs. H0")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(df["Landscape amp (H0)"], df["Landscape amp (H1)"], color='red', label="Lan (H1) vs. Lan (H0)")
plt.xlabel("Landscape amp H0")
plt.ylabel("Landscape amp H1")
plt.title("Landscape amp: H1 vs. H0")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(df["Persistence Image amp (H0)"], df["Persistence Image amp (H1)"], color='red', label="Per Ima (H1) vs.Per Lan (H0)")
plt.xlabel("Persistence Image amp H0")
plt.ylabel("Persistence Image amp H1")
plt.title("Persistence Image amp: H1 vs. H0")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(df["Betti amp (H0)"], df["Betti amp (H1)"], color='red', label="Betti (H1) vs.Per Betti (H0)")
plt.xlabel("Betti amp H0")
plt.ylabel("Betti amp H1")
plt.title("Betti amp: H1 vs. H0")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(df["Heat amp (H0)"], df["Heat amp (H1)"], color='red', label="Heat (H1) vs.Per Heat (H0)")
plt.xlabel("Heat amp H0")
plt.ylabel("Heat amp H1")
plt.title("Heat amp: H1 vs. H0")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(df["Heat amp (H0)"], df["Heat amp (H1)"], color='red', label="Heat (H1) vs.Per Heat (H0)")
plt.xlabel("Heat amp H0")
plt.ylabel("Heat amp H1")
plt.title("Heat amp: H1 vs. H0")
plt.legend()
plt.grid(True)
plt.show()



