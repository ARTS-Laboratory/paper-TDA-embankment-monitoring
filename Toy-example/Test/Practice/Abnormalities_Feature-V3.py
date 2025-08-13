
"""
Reads your Abnormalities_Features-style Excel and plots with:
  (1) Per-feature_y grid: feature_y vs all feature_x (viridis circles)
  (2) Grid: each feature vs Index (viridis circles)
NO wire/line plot.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt

# ====== CONFIG ======
file_path = "C:/Users/hp zbook g5/Documents/GitHub/paper-TDA-embankment-monitoring/Pre-processing/Abnormalities_Features-V5.xlsx"  # change to "man.xlsx" on your PC
  # or "/mnt/data/Abnormalities_Features-V5.xlsx"
preferred_sheet = None  # set to "Table" to force; else first sheet
# ====================

# requires: openpyxl
# pip:  python -m pip install openpyxl
# conda: conda install -c conda-forge openpyxl

def detect_header_row(df_raw, max_rows=30):
    # prefer a row that literally contains "Index"
    for i in range(min(max_rows, len(df_raw))):
        row = df_raw.iloc[i].astype(str).str.strip().str.lower()
        if (row == "index").any():
            return i
    # fallback: pick the "textiest" row among first 10 rows
    scores = []
    for i in range(min(10, len(df_raw))):
        r = df_raw.iloc[i]
        non_na = r.notna().sum()
        numeric = pd.to_numeric(r, errors="coerce").notna().sum()
        scores.append((non_na - numeric, i))
    scores.sort(reverse=True)
    return scores[0][1] if scores else 0

def load_features_table(xlsx_path, sheet_name=None):
    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    sh = sheet_name or xls.sheet_names[0]
    df_raw = pd.read_excel(xls, sheet_name=sh, header=None)

    hrow = detect_header_row(df_raw)
    cols = df_raw.iloc[hrow].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df = df_raw.iloc[hrow + 1 :].copy()
    df.columns = cols

    # drop empty stuff
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # standardize Index column
    lower = {str(c).lower(): c for c in df.columns}
    if "index" in lower:
        index_col = lower["index"]
    else:
        index_col = df.columns[0]
        df.rename(columns={index_col: "Index"}, inplace=True)
        index_col = "Index"

    # make numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=[index_col], inplace=True)
    df.sort_values(by=index_col, inplace=True)

    features = [c for c in df.columns if c != index_col]
    if not features:
        raise ValueError("No feature columns found besides the Index.")

    return df, index_col, features

def gridsz(n): 
    return int(ceil(sqrt(n)))

def scatter_viridis(ax, x, y, s=35):  # s controls size
    m = x.notna() & y.notna()
    xv, yv = x[m].to_numpy(), y[m].to_numpy()
    if yv.size == 0:
        return
    vmin, vmax = np.nanmin(yv), np.nanmax(yv)
    if not np.isfinite(vmax - vmin) or vmax == vmin:
        vmin, vmax = vmin - 0.5, vmin + 0.5
    edge_colors = plt.cm.viridis((yv - vmin) / (vmax - vmin))

    ax.scatter(
        xv, yv,
        facecolors='none',          # <-- makes them hollow
        edgecolors=edge_colors,     # <-- viridis edges
        s=s,                        # <-- change this to resize
        linewidths=1.2,
        marker='o',
        alpha=0.9
    )

# ---- load
df, index_name, features = load_features_table(file_path, sheet_name=preferred_sheet)
g = gridsz(len(features))

# ============================================================
# 1) For each feature_y, plot against all feature_x (viridis circles)
# ============================================================
for fy in features:
    fig, axes = plt.subplots(g, g, figsize=(16, 16))
    axes = axes.flatten()

    used = 0
    for j, fx in enumerate(features):
        ax = axes[j]
        scatter_viridis(ax, df[fx], df[fy], s=30)
        ax.set_xlabel(str(fx), fontsize=10)
        ax.set_ylabel(str(fy), fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=8)
        used += 1

    for k in range(used, len(axes)):
        fig.delaxes(axes[k])

    fig.suptitle(f"{fy} vs Other Features", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0.04, 0.06, 0.98, 0.95])
    plt.show()

# ============================================================
# 2) Feature vs Index (viridis circles) — NO wire plot
# ============================================================
fig, axes = plt.subplots(g, g, figsize=(16, 16))
axes = axes.flatten()

used = 0
for i, f in enumerate(features):
    ax = axes[i]
    scatter_viridis(ax, df[index_name], df[f], s=30)
    ax.set_xlabel(str(index_name), fontsize=10)
    ax.set_ylabel(str(f), fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=8)
    used += 1

for k in range(used, len(axes)):
    fig.delaxes(axes[k])

fig.suptitle("Feature vs Index (Scatter, viridis)", fontsize=16, y=0.98)
fig.tight_layout(rect=[0.04, 0.06, 0.98, 0.95])
plt.show()
