# pca_dates_pure_PC1bar_pub_grid.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.decomposition import PCA

# =========================
# 0) SETTINGS — EDIT THESE
# =========================
file_path = r"C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/TDA_PCA_template.xlsx"
excel_header_row = 0
EPS = 1e-12

# ---- Plot layout controls ----
AXIS_MARGIN_FRAC = 0.18   # padding beyond data extent (increase if labels touch axes)
DISPLAY_SHRINK    = 2  # <1.0 pulls points closer (visual only). Try 0.9, 0.85, 0.75.
LABEL_OFFSET      = 7     # default label distance from marker, in *points*

# Per-date fine tuning (offsets in points). Positive x→right, positive y→up.
offset_map = {
     '2021-06': (5, -10),
    # '2021-08': (-8,   8),
    # '2022-02': (10,  -8),
     '2022-10': ( 15,   -4),
    # '2023-06': ( 6, -12),
    # '2023-09': ( 8,   8),
}

# =========================
# 1) PLOTTING STYLE (your request)
# =========================
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})
plt.rcParams.update({'font.size': 9})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})
# If TeX fonts error on your machine, un-comment:
# plt.rcParams.update({'text.usetex': False})

# =========================
# 2) REQUIRED COLUMNS
# =========================
FEATURE_COLS = [
    "persistence entropy (H0)","persistence entropy (H1)",
    "number of points (H0)","number of points(H1)",
    "bottleneck  (H0)","bottleneck  (H1)",
    "wasserstein  (H0)","wasserstein  (H1)",
    "landscape  (H0)","landscape  (H1)",
    "persistence image  (H0)","persistence image  (H1)",
    "Betti  (H0)","Betti  (H1)",
    "heat  (H0)","heat  (H1)"
]
REQUIRED = ["date","humidity_percent"] + FEATURE_COLS

# =========================
# 3) LOAD
# =========================
if file_path.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(file_path, header=excel_header_row)
elif file_path.lower().endswith(".csv"):
    df = pd.read_csv(file_path)
else:
    raise ValueError("Unsupported file type. Use .xlsx/.xls or .csv")

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")
if df.shape[0] < 2:
    raise ValueError("Need at least 2 rows (dates) to run PCA.")

dates = df["date"].astype(str).to_numpy()
humidity = pd.to_numeric(df["humidity_percent"], errors="coerce").to_numpy()
X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy()

# =========================
# 4) CLEAN & STANDARDIZE (SAFE)
# =========================
col_med = np.nanmedian(X, axis=0)
col_med = np.where(np.isnan(col_med), 0.0, col_med)
ri, ci = np.where(np.isnan(X))
X[ri, ci] = col_med[ci]

std = X.std(axis=0, ddof=0)
keep = std >= EPS
dropped = [c for c, k in zip(FEATURE_COLS, keep) if not k]
if dropped:
    print("Dropped constant features:", dropped)

Xk = X[:, keep]
feat_names_kept = [c for c, k in zip(FEATURE_COLS, keep) if k]
if Xk.shape[1] == 0:
    raise ValueError("All features are constant across dates; PCA is undefined.")

mu = Xk.mean(axis=0)
sg = Xk.std(axis=0, ddof=0)
sg[sg < EPS] = 1.0
Xz = (Xk - mu) / sg

# =========================
# 5) PCA → 2D
# =========================
pca = PCA(n_components=2, random_state=0)
Z = pca.fit_transform(Xz)                 # (n_dates, 2)
evr = pca.explained_variance_ratio_
loadings = pca.components_.T

# =========================
# 6) DISPLAY TWEAKS (shrink + overlap handling)
# =========================
# Shrink around the centroid so points look closer (visual only)
Z_mean = Z.mean(axis=0)
Z_disp = Z_mean + DISPLAY_SHRINK * (Z - Z_mean)

# If any dates land exactly together, nudge slightly so both are visible
Z_plot = Z_disp.copy()
Zr = np.round(Z_disp, 6)
seen = {}
for i, xy in enumerate(map(tuple, Zr)):
    seen.setdefault(xy, []).append(i)
for xy, idxs in seen.items():
    if len(idxs) > 1:
        for k, j in enumerate(idxs):
            Z_plot[j] += np.array([0.03*k, -0.03*k])  # tiny deterministic offset

# =========================
# 7) PCA SCATTER (grid behind points; labels offset)
# =========================
def label_offset(x, y, d_pts=LABEL_OFFSET):
    ox = d_pts if x >= Z_mean[0] else -d_pts
    oy = d_pts if y >= Z_mean[1] else -d_pts
    return ox, oy

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.set_axisbelow(True)                                   # grid behind marks
ax.grid(True, which="both", linestyle="--", lw=0.5, alpha=0.35, zorder=0)

sc = ax.scatter(Z_plot[:, 0], Z_plot[:, 1], c=humidity, s=90, zorder=2)

# Labels (in points), with small white box so cross/grid doesn’t cut through text
for (x, y), dlabel in zip(Z_plot, dates):
    ox, oy = offset_map.get(dlabel, label_offset(x, y, LABEL_OFFSET))
    ax.annotate(dlabel, xy=(x, y), xytext=(ox, oy),
                textcoords='offset points',
                ha='left' if ox >= 0 else 'right',
                va='bottom' if oy >= 0 else 'top',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85, lw=0),
                zorder=3)

# Cross hairs (behind points/labels)
ax.axhline(0, lw=0.8, color="#5b7aa3", zorder=1)
ax.axvline(0, lw=0.8, color="#5b7aa3", zorder=1)

# Expand axes symmetrically with margin, so labels don’t hit edges/axes
xhalf = np.max(np.abs(Z_plot[:, 0])) * (1 + AXIS_MARGIN_FRAC)
yhalf = np.max(np.abs(Z_plot[:, 1])) * (1 + AXIS_MARGIN_FRAC)
ax.set_xlim(-xhalf, xhalf)
ax.set_ylim(-yhalf, yhalf)

ax.set_xlabel(rf"PC1 ({evr[0]*100:.1f}\% var)")
ax.set_ylabel(rf"PC2 ({evr[1]*100:.1f}\% var)")
ax.set_title(r"Dates in PCA space (color = humidity)")
cb = plt.colorbar(sc, ax=ax); cb.set_label(r"Humidity (\%)")

plt.tight_layout()
plt.savefig("dates_pca_humidity_pure.png", dpi=600, bbox_inches='tight')
plt.show()

# =========================
# 8) PC1 LOADINGS: ranking + bar plot (H0=blue, H1=orange)
# =========================
def h_color(name: str) -> str:
    n = name.upper()
    return "tab:blue" if "H0" in n else ("tab:orange" if "H1" in n else "gray")

pc1 = loadings[:, 0]
contrib = (pc1**2) / np.sum(pc1**2) if np.sum(pc1**2) > 0 else np.zeros_like(pc1)
order = np.argsort(-np.abs(pc1))
ranked_pc1 = pd.DataFrame({
    "feature": np.array(feat_names_kept)[order],
    "loading_PC1": pc1[order],
    "abs_loading": np.abs(pc1[order]),
    "contribution_%": 100 * contrib[order]
})
ranked_pc1.to_csv("pc1_feature_loadings_ranked.csv", index=False)

fig2, ax2 = plt.subplots(figsize=(9, 6))
y = np.array(ranked_pc1["feature"])
x = np.array(ranked_pc1["loading_PC1"])
colors = [h_color(n) for n in y]
ax2.barh(y, x, color=colors)
ax2.axvline(0, color="k", lw=0.8)
ax2.set_xlabel(r"PC1 loading (sign = direction, |value| = importance)")
ax2.set_ylabel(r"Feature")
ax2.set_title(r"PC1 loadings by feature (H0=blue, H1=orange)")
plt.tight_layout()
plt.savefig("pc1_loadings_bar.png", dpi=600, bbox_inches='tight')
plt.show()

print("\nExplained variance: "
      f"PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}% (total={evr.sum()*100:.1f}%)")
print("Saved: dates_pca_humidity_pure.png, pc1_loadings_bar.png, pc1_feature_loadings_ranked.csv")
