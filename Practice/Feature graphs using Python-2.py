
import pandas as pd
import os
import matplotlib.pyplot as plt

# Ensure the file is in the current working directory.
print("Current working directory:", os.getcwd())

# Load the Excel file with header from row 3 (0-indexed header=2) and sheet "Table"
df = pd.read_excel("Book1.xlsx", sheet_name="Table", header=2)
print("Columns in Excel file:", df.columns.tolist())

# ---------------------------
# Plot 1: Persistence Entropy H0 vs. Persistence Entropy H0
# ---------------------------
plt.figure(figsize=(8,6))
# Use a continuous colormap based on PE (H0) values.
sc = plt.scatter(df["PE (H0)"], df["PE (H0)"], c=df["PE (H0)"], cmap="viridis", s=50)

plt.xlabel("Persistence Entropy H0")
plt.ylabel("Persistence Entropy H0")
plt.title("Persistence Entropy: H0 vs. H0")
plt.legend(["PE H0 vs. PE H0"])
#cb = plt.colorbar(sc)
plt.grid(True)
plt.show()


# ---------------------------
# Plot 2: Persistence Entropy H0 vs. Persistence Entropy H1
# ---------------------------
plt.figure(figsize=(8,6))
# Use a continuous colormap based on PE (H0) values.
sc = plt.scatter(df["PE (H0)"], df["PE (H1)"], c=df["PE (H0)"], cmap="viridis", s=50)

plt.xlabel("Persistence Entropy H1")
plt.ylabel("Persistence Entropy H0")
plt.title("Persistence Entropy: H0 vs. H1")
plt.legend(["PE H0 vs. PE H1"])
#cb = plt.colorbar(sc)
plt.grid(True)
plt.show()


# ---------------------------
# Plot 3: Persistence Entropy H0 vs. Number-of-Points H0
# ---------------------------
plt.figure(figsize=(8,6))
# Use a continuous colormap based on NoP (H0) values.
sc = plt.scatter(df["PE (H0)"], df["NoP (H0)"], c=df["PE (H0)"], cmap="plasma", s=50)
plt.xlabel("Number-of-Points H0")
plt.ylabel("Persistence Entropy H0")
plt.title("Persistence Entropy: H0 vs. Number-of-Points H0")
plt.legend(["PE H0 vs. NoP H0"])
#cb = plt.colorbar(sc)
plt.grid(True)
plt.show()
#%%
# ---------------------------
# Plot 4: Persistence Entropy H0 vs. Number-of-Points H1
# ---------------------------
plt.figure(figsize=(8,6))
# Use a continuous colormap based on Bottleneck amp (H0) values.
sc = plt.scatter(df["PE (H0)"], df["NoP (H1)"], 
                 c=df["PE (H0)"], cmap="inferno", s=50)
plt.xlabel("Number-of-Points H1")
plt.ylabel("Persistence Entropy H0")
plt.title("Persistence Entropy: H0 vs. Number-of-Points H0")
plt.legend(["PE H0 vs. NoP H1"])
cb = plt.colorbar(sc)
plt.grid(True)
plt.show()

# ---------------------------
# Plot 5: Persistence Entropy H0 vs. Bottleneck amp H0
# ---------------------------
plt.figure(figsize=(8,6))
# Use a continuous colormap based on Wasserstein amp (H0) values.
sc = plt.scatter(df["PE (H0)"], df["Bottleneck amp H0"], 
                 c=df["PE (H0)"], cmap="cividis", s=50)
plt.xlabel("Bottleneck amp H0")
plt.ylabel("PE (H0)")
plt.title("Persistence Entropy H0 vs. Bottleneck amp H0")
plt.legend(["PE H0 vs. Bottleneck amp H0"])
cb = plt.colorbar(sc)
plt.grid(True)
plt.show()

# ---------------------------
# Plot 5: Landscape Amplitude H1 vs. Landscape Amplitude H0
# ---------------------------
plt.figure(figsize=(8,6))
# Use a continuous colormap based on Landscape amp (H0) values.
sc = plt.scatter(df["Landscape amp (H0)"], df["Landscape amp (H1)"], 
                 c=df["Landscape amp (H0)"], cmap="plasma", s=50)
plt.xlabel("Landscape Amplitude H0")
plt.ylabel("Landscape Amplitude H1")
plt.title("Landscape Amplitude: H1 vs. H0")
plt.legend(["Landscape H1 vs. H0"])
cb = plt.colorbar(sc)
plt.grid(True)
plt.show()

# ---------------------------
# Plot 6: Persistence Image Amplitude H1 vs. Persistence Image Amplitude H0
# ---------------------------
plt.figure(figsize=(8,6))
# Use a continuous colormap based on Persistence Image amp (H0) values.
sc = plt.scatter(df["Persistence Image amp (H0)"], df["Persistence Image amp (H1)"], 
                 c=df["Persistence Image amp (H0)"], cmap="viridis", s=50)
plt.xlabel("Persistence Image Amplitude H0")
plt.ylabel("Persistence Image Amplitude H1")
plt.title("Persistence Image Amplitude: H1 vs. H0")
plt.legend(["Persistence Image H1 vs. H0"])
cb = plt.colorbar(sc)
plt.grid(True)
plt.show()

# ---------------------------
# Plot 7: Betti Amplitude H1 vs. Betti Amplitude H0
# ---------------------------
plt.figure(figsize=(8,6))
# Use a continuous colormap based on Betti amp (H0) values.
sc = plt.scatter(df["Betti amp (H0)"], df["Betti amp (H1)"], 
                 c=df["Betti amp (H0)"], cmap="inferno", s=50)
plt.xlabel("Betti Amplitude H0")
plt.ylabel("Betti Amplitude H1")
plt.title("Betti Amplitude: H1 vs. H0")
plt.legend(["Betti H1 vs. H0"])
cb = plt.colorbar(sc)
plt.grid(True)
plt.show()

# ---------------------------
# Plot 8: Heat Amplitude H1 vs. Heat Amplitude H0
# ---------------------------
plt.figure(figsize=(8,6))
# Use a continuous colormap based on Heat amp (H0) values.
sc = plt.scatter(df["Heat amp (H0)"], df["Heat amp (H1)"], 
                 c=df["Heat amp (H0)"], cmap="cividis", s=50)
plt.xlabel("Heat Amplitude H0")
plt.ylabel("Heat Amplitude H1")
plt.title("Heat Amplitude: H1 vs. H0")
plt.legend(["Heat H1 vs. H0"])
cb = plt.colorbar(sc)
plt.grid(True)
plt.show()

# ---------------------------
# Plot 9: Heat Amplitude H1 vs. Heat Amplitude H0
# ---------------------------
plt.figure(figsize=(8,6))
# Use a continuous colormap based on Heat amp (H0) values.
sc = plt.scatter(df["Heat amp (H0)"], df["Heat amp (H1)"], 
                 c=df["Heat amp (H0)"], cmap="cividis", s=50)
plt.xlabel("Heat Amplitude H0")
plt.ylabel("Heat Amplitude H1")
plt.title("Heat Amplitude: H1 vs. H0")
plt.legend(["Heat H1 vs. H0"])
cb = plt.colorbar(sc)
plt.grid(True)
plt.show()

# ---------------------------
# Plot 10: Heat Amplitude H1 vs. Heat Amplitude H0
# ---------------------------
plt.figure(figsize=(8,6))
# Use a continuous colormap based on Heat amp (H0) values.
sc = plt.scatter(df["Heat amp (H0)"], df["Heat amp (H1)"], 
                 c=df["Heat amp (H0)"], cmap="cividis", s=50)
plt.xlabel("Heat Amplitude H0")
plt.ylabel("Heat Amplitude H1")
plt.title("Heat Amplitude: H1 vs. H0")
plt.legend(["Heat H1 vs. H0"])
cb = plt.colorbar(sc)
plt.grid(True)
plt.show()