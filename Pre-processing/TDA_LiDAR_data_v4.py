# pca_dates_with_vertical_pc1bar_and_2dlogreg_surface_CENTERED.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =========================
# SETTINGS
# =========================
file_path = r"C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/TDA_PCA_template.xlsx"
excel_header_row = 0
EPS = 1e-12
T_LABEL = 28.0   # wet if humidity >= 28%

# PCA display controls (kept as in your plot)
AXIS_MARGIN_FRAC = 0.18
SHRINK_X, SHRINK_Y = 0.40, 0.40
LABEL_OFFSET = 7
offset_map = {
    '2021-06': (5, -10),
    '2022-10': (15,  -4),
}

# Decision-surface grid expansion (to reveal 0.1/0.9 contours)
GRID_EXPAND = 1.6  # multiply previous span by this factor

# Text sizes for the bar chart
TITLE_FONTSIZE = 11
LABEL_FONTSIZE = 11
TICK_FONTSIZE  = 11

# =========================
# PLOTTING STYLE (your LaTeX/Times settings)
# =========================
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})
plt.rcParams.update({'font.size': 9})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})
# If LaTeX isn't available on your machine, uncomment:
# plt.rcParams.update({'text.usetex': False})

# =========================
# REQUIRED COLS + LOAD
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
# CLEAN + STANDARDIZE
# =========================
col_med = np.nanmedian(X, axis=0)
col_med = np.where(np.isnan(col_med), 0.0, col_med)
ri, ci = np.where(np.isnan(X)); X[ri, ci] = col_med[ci]

std = X.std(axis=0, ddof=0)
keep = std >= EPS
Xk = X[:, keep]
feat_names_kept = [c for c, k in zip(FEATURE_COLS, keep) if k]
if Xk.shape[1] == 0:
    raise ValueError("All features are constant across dates; PCA is undefined.")

mu = Xk.mean(axis=0)
sg = Xk.std(axis=0, ddof=0); sg[sg < EPS] = 1.0
Xz = (Xk - mu) / sg

# =========================
# PCA → 2D
# =========================
pca = PCA(n_components=2, random_state=0)
Z = pca.fit_transform(Xz)                 # (n_dates, 2)
evr = pca.explained_variance_ratio_
loadings = pca.components_.T

# =========================
# Display transform (same as your figure)
# =========================
Z_mean = Z.mean(axis=0)
def shrink_display(z2):
    return np.column_stack([
        Z_mean[0] + SHRINK_X * (z2[:, 0] - Z_mean[0]),
        Z_mean[1] + SHRINK_Y * (z2[:, 1] - Z_mean[1]),
    ])
def unshrink_to_true(d2):
    out = np.empty_like(d2)
    out[:,0] = (d2[:,0] - Z_mean[0]) / (SHRINK_X if SHRINK_X!=0 else 1.0) + Z_mean[0]
    out[:,1] = (d2[:,1] - Z_mean[1]) / (SHRINK_Y if SHRINK_Y!=0 else 1.0) + Z_mean[1]
    return out

Z_disp = shrink_display(Z)
Z_plot = Z_disp.copy()
Zr = np.round(Z_disp, 6)
seen = {}
for i, xy in enumerate(map(tuple, Zr)):
    seen.setdefault(xy, []).append(i)
for xy, idxs in seen.items():
    if len(idxs) > 1:
        for k, j in enumerate(idxs):
            Z_plot[j] += np.array([0.03*k, -0.03*k])

# =========================
# VERTICAL PC1 IMPORTANCE (centered; no +/- labels)
# =========================
def h_color(name: str) -> str:
    n = name.upper()
    return "tab:blue" if "H0" in n else ("tab:orange" if "H1" in n else "gray")

pc1 = loadings[:, 0]
order = np.argsort(-np.abs(pc1))
feat_sorted = np.array(feat_names_kept)[order]
pc1_sorted = pc1[order]
mag = np.abs(pc1_sorted)   # all bars upward
labels2 = [f.replace(" (H0)", "\n(H0)").replace(" (H1)", "\n(H1)") for f in feat_sorted]
colors = [h_color(n) for n in feat_sorted]

fig_bar, ax_bar = plt.subplots(figsize=(11, 5), constrained_layout=True)
xpos = np.arange(len(mag))
ax_bar.bar(xpos, mag, color=colors, width=0.75)
ax_bar.set_ylabel(r"PC1 $|$loading$|$ (importance)", fontsize=LABEL_FONTSIZE)
ax_bar.set_title(r"PC1 loadings by feature (H0=blue, H1=orange)", fontsize=TITLE_FONTSIZE)
ax_bar.set_xticks(xpos)
ax_bar.set_xticklabels(labels2, rotation=90, ha="center", va="top", fontsize=TICK_FONTSIZE)
ax_bar.tick_params(axis='y', labelsize=TICK_FONTSIZE)
ax_bar.margins(x=0.02)  # small symmetric side margins
plt.savefig("pc1_loadings_bar_VERTICAL_abs.png", dpi=600)  # no bbox_inches='tight' → nicer centering
plt.show()

# =========================
# PCA SCATTER + 2D LOGISTIC SURFACE/CONTOURS (all filled circles)
# =========================
y_lab = (humidity >= T_LABEL).astype(int)
both_classes = (np.unique(y_lab).size == 2)

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.set_axisbelow(True)
ax.grid(True, which="both", linestyle="--", lw=0.5, alpha=0.35, zorder=0)

norm = plt.Normalize(vmin=np.nanmin(humidity), vmax=np.nanmax(humidity))
cmap = plt.cm.viridis

# Decision surface behind points
if both_classes:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(penalty="l2", solver="lbfgs",
                                  C=1.0, max_iter=2000,
                                  class_weight="balanced", random_state=0))
    ])
    pipe.fit(Z, y_lab)

    # Grid in DISPLAY coords → map back to true PCA for probabilities
    xhalf = np.max(np.abs(Z_plot[:, 0])) * (1 + AXIS_MARGIN_FRAC) * GRID_EXPAND
    yhalf = np.max(np.abs(Z_plot[:, 1])) * (1 + AXIS_MARGIN_FRAC) * GRID_EXPAND
    xg = np.linspace(-xhalf, xhalf, 500)
    yg = np.linspace(-yhalf, yhalf, 500)
    XXd, YYd = np.meshgrid(xg, yg)
    disp_grid = np.c_[XXd.ravel(), YYd.ravel()]
    true_grid = unshrink_to_true(disp_grid)
    proba = pipe.predict_proba(true_grid)[:, 1].reshape(XXd.shape)

    # softly colored regions
    ax.contourf(XXd, YYd, proba, levels=[0.0, 0.5, 1.0],
                colors=["#cfe8ff", "#f5e6c8"], alpha=0.35, zorder=1)

    # 0.1 / 0.5 / 0.9 contours — 0.5 in RED
    cs = ax.contour(XXd, YYd, proba, levels=[0.1, 0.5, 0.9],
                    colors=["tab:blue", "red", "tab:green"],
                    linestyles=["-", "--", "-"],
                    linewidths=[1.1, 1.8, 1.1], zorder=2)
    ax.clabel(cs, inline=True, fmt={0.1:"0.1", 0.5:"0.5", 0.9:"0.9"}, fontsize=8)
    
    # 0.1 / 0.5 / 0.9 contours — 0.5 in RED
    cs = ax.contour(XXd, YYd, proba, levels=[0.1, 0.5, 0.9],
                colors=["tab:blue", "red", "tab:green"],
                linestyles=["-", "--", "-"],
                linewidths=[1.1, 1.8, 1.1], zorder=2)
    ax.clabel(cs, inline=True, fmt={0.1: "0.1", 0.5: "0.5", 0.9: "0.9"}, fontsize=8)

# >>> ADD THESE TWO MARGIN LINES (±0.25 around 0.5 → P=0.25 and P=0.75) <<<
cs_margin = ax.contour(XXd, YYd, proba, levels=[0.25, 0.75],
                       colors=["k", "k"], linestyles=["--", "--"],
                       linewidths=1.2, zorder=2)
ax.clabel(cs_margin, inline=True, fmt={0.25: "0.25", 0.75: "0.75"},
          fontsize=7, colors="k")


# Scatter: all filled circles (viridis by humidity)
sc = ax.scatter(Z_plot[:,0], Z_plot[:,1],
                c=humidity, cmap=cmap, s=70, edgecolors="k", linewidths=0.3, zorder=3)

# Labels (same offsets)
def label_offset(x, y, d_pts=LABEL_OFFSET):
    ox = d_pts if x >= Z_mean[0] else -d_pts
    oy = d_pts if y >= Z_mean[1] else -d_pts
    return ox, oy

for (x, yv), dlabel in zip(Z_plot, dates):
    ox, oy = offset_map.get(dlabel, label_offset(x, yv, LABEL_OFFSET))
    ax.annotate(dlabel, xy=(x, yv), xytext=(ox, oy),
                textcoords='offset points',
                ha='left' if ox >= 0 else 'right',
                va='bottom' if oy >= 0 else 'top',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85, lw=0),
                zorder=4)

# Cross hairs & limits (same style)
ax.axhline(0, lw=0.8, color="#5b7aa3", zorder=5)
ax.axvline(0, lw=0.8, color="#5b7aa3", zorder=5)

xhalf_disp = np.max(np.abs(Z_plot[:, 0])) * (1 + AXIS_MARGIN_FRAC)
yhalf_disp = np.max(np.abs(Z_plot[:, 1])) * (1 + AXIS_MARGIN_FRAC)
ax.set_xlim(-xhalf_disp, xhalf_disp)
ax.set_ylim(-yhalf_disp, yhalf_disp)

cb = plt.colorbar(sc, ax=ax); cb.set_label(r"Humidity (\%)")
ax.set_xlabel(rf"PC1 ({evr[0]*100:.1f}\% var)")
ax.set_ylabel(rf"PC2 ({evr[1]*100:.1f}\% var)")
ax.set_title(r"Dates in PCA space (color = humidity) — 2D logistic surface")

plt.tight_layout()
plt.savefig("dates_pca_with_2Dlogreg_surface_RED50_filleddots.png", dpi=600, bbox_inches='tight')
plt.show()

print("\nSaved:",
      "pc1_loadings_bar_VERTICAL_abs.png,",
      "dates_pca_with_2Dlogreg_surface_RED50_filleddots.png")
