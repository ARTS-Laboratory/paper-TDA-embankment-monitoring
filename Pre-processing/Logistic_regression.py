# logreg_binary_per_feature_firstsheet_spaced_savefix.py
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score

# =========================
# 0) SETTINGS
# =========================
file_path = r"C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/TDA_PCA_template.xlsx"
T_LABEL = 28.0                   # humidity threshold (%) for wet/dry labels

# Layout/appearance
FIGSIZE = (12, 9)                # larger to prevent collisions
WSPACE, HSPACE = 0.45, 0.80      # inter-subplot gaps
COLORBAR_SIZE = "1.8%"           # slim per-axes colorbars
COLORBAR_PAD  = 0.02
TITLE_FONTSIZE = 9
AXIS_FONTSIZE  = 9

# Make logistic curve “S” more visible (plot over ±k·σ around mean; visual only)
CURVE_STD_SPAN = 3.0

SAVE_INDIVIDUAL_FIGS = False
SHOW_DATES = False
DATE_LABEL_OFFSET = (3, 3)

# =========================
# 1) PLOTTING STYLE (your request)
# =========================
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})
plt.rcParams.update({'font.size': 9})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})
# If LaTeX isn't installed on your machine, uncomment the next line:
# plt.rcParams.update({'text.usetex': False})

# =========================
# 2) Canonical headers + helpers
# =========================
CANON_FEATURES = [
    "persistence entropy (H0)","persistence entropy (H1)",
    "number of points (H0)","number of points (H1)",
    "bottleneck (H0)","bottleneck (H1)",
    "wasserstein (H0)","wasserstein (H1)",
    "landscape (H0)","landscape (H1)",
    "persistence image (H0)","persistence image (H1)",
    "Betti (H0)","Betti (H1)",
    "heat (H0)","heat (H1)"
]
REQUIRED = ["date", "humidity_percent"] + CANON_FEATURES

def normalize_col_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*\(", " (", s)   # exactly one space before '('
    return s

def keyize(s: str) -> str:
    return normalize_col_name(s).lower()

# =========================
# 3) LOAD FIRST SHEET + normalize headers
# =========================
lower = file_path.lower()
if lower.endswith((".xlsx", ".xls")):
    df = pd.read_excel(file_path, sheet_name=0)     # ALWAYS the first worksheet
elif lower.endswith(".csv"):
    df = pd.read_csv(file_path)
else:
    raise ValueError("Use .xlsx/.xls or .csv")

# Map normalized key -> actual column name
col_key_to_actual = {keyize(c): c for c in df.columns}
rename_map, missing = {}, []
for canon in REQUIRED:
    k = keyize(canon)
    actual = col_key_to_actual.get(k)
    if actual is None:
        missing.append(canon)
    else:
        rename_map[actual] = normalize_col_name(canon)

if missing:
    print("Columns present:", list(df.columns))
    raise ValueError(f"Required columns not found after normalization: {missing}")

df = df.rename(columns=rename_map)

# =========================
# 4) Coerce numerics, labels, color scale
# =========================
df["humidity_percent"] = pd.to_numeric(df["humidity_percent"], errors="coerce")
for c in CANON_FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=CANON_FEATURES, how="all").copy()
if df.shape[0] < 2:
    raise ValueError("Need at least 2 rows with some feature values.")

dates = df["date"].astype(str).values
humidity = df["humidity_percent"].values
if np.isnan(humidity).all():
    raise ValueError("humidity_percent is all NaN.")

# Binary labels from threshold
y_all = (humidity >= T_LABEL).astype(int)
print(f"Label threshold = {T_LABEL:.1f}% → dry: {(y_all==0).sum()}, wet: {(y_all==1).sum()}")

# Color scale (actual humidity)
norm = plt.Normalize(vmin=np.nanmin(humidity), vmax=np.nanmax(humidity))
cmap = plt.cm.viridis

# =========================
# 5) Fit & plot (16 panels)
# =========================
loo = LeaveOneOut()
rows = []

fig, axes = plt.subplots(4, 4, figsize=FIGSIZE)
axes = axes.ravel()

constant_debug = []  # to print min/max when we flag a feature constant

for i, (ax, feat) in enumerate(zip(axes, CANON_FEATURES)):
    x_raw = df[feat].values.astype(float)
    mask = ~np.isnan(x_raw) & ~np.isnan(humidity)
    x = x_raw[mask]
    y = y_all[mask]
    hum_use = humidity[mask]
    dates_use = df.loc[mask, "date"].astype(str).values
    n_pts = len(x)

    ax.set_axisbelow(True)
    ax.grid(True, ls="--", lw=0.5, alpha=0.35)

    # Skip constant/NaN features or single-class after masking
    if n_pts < 2 or np.nanstd(x) == 0:
        ax.text(0.5, 0.5, "constant/NaN\nfeature", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{feat}  (n={n_pts})", fontsize=TITLE_FONTSIZE)
        ax.set_ylim(-0.1, 1.1); ax.set_yticks([0, 0.5, 1.0])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=COLORBAR_SIZE, pad=COLORBAR_PAD)
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax).set_label(r"Soil moisture (\%)", fontsize=AXIS_FONTSIZE)
        if n_pts >= 1:
            constant_debug.append((feat, float(np.nanmin(x)), float(np.nanmax(x))))
        rows.append({"feature": feat, "loo_acc": np.nan, "coef_stdX": np.nan,
                     "intercept": np.nan, "x_star_p05": np.nan})
        continue

    if len(np.unique(y)) < 2:
        ax.text(0.5, 0.5, "only one class\nunder this threshold", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{feat}  (n={n_pts})", fontsize=TITLE_FONTSIZE)
        ax.set_ylim(-0.1, 1.1); ax.set_yticks([0, 0.5, 1.0])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=COLORBAR_SIZE, pad=COLORBAR_PAD)
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax).set_label(r"Soil moisture (\%)", fontsize=AXIS_FONTSIZE)
        rows.append({"feature": feat, "loo_acc": np.nan, "coef_stdX": np.nan,
                     "intercept": np.nan, "x_star_p05": np.nan})
        continue

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l2", solver="lbfgs", C=1.0, max_iter=1000,
            class_weight="balanced", random_state=0
        ))
    ])

    # LOO accuracy
    try:
        loo_acc = cross_val_score(pipe, x.reshape(-1,1), y, cv=loo, scoring="accuracy").mean()
    except Exception:
        loo_acc = np.nan

    # Fit for plotting
    pipe.fit(x.reshape(-1,1), y)

    # Probability curve over ±k·σ (to reveal the S curve better)
    mu = x.mean()
    sigma = x.std() if x.std() > 0 else 1.0
    xmin = mu - CURVE_STD_SPAN * sigma
    xmax = mu + CURVE_STD_SPAN * sigma
    grid = np.linspace(xmin, xmax, 300).reshape(-1, 1)
    proba = pipe.predict_proba(grid)[:, 1]

    # Decision boundary (p=0.5) in original units
    beta1 = pipe.named_steps["lr"].coef_[0, 0]
    beta0 = pipe.named_steps["lr"].intercept_[0]
    x_star = np.nan
    if beta1 != 0:
        z_star = -beta0 / beta1
        x_star = mu + sigma * z_star

    # Observations: shape by class, color by humidity
    jitter = (np.random.rand(n_pts) - 0.5) * 0.06
    y_obs = y + jitter
    colors = cmap(norm(hum_use))
    dmask = (y == 0); wmask = ~dmask
    ax.scatter(x[dmask], y_obs[dmask], marker="x", s=35, c=colors[dmask], linewidths=1.0, label="dry")
    ax.scatter(x[wmask], y_obs[wmask], marker="o", s=40, facecolors="none",
               edgecolors=colors[wmask], linewidths=1.0, label="wet")

    if SHOW_DATES:
        for xi, yi, dt in zip(x, y_obs, dates_use):
            ax.annotate(str(dt), xy=(xi, yi), xytext=DATE_LABEL_OFFSET,
                        textcoords="offset points", fontsize=8)

    ax.plot(grid.ravel(), proba, lw=1.5, color="k")
    if np.isfinite(x_star):
        ax.axvline(x_star, color="k", ls=":", lw=1.0)

    ax.set_title(f"{feat}  (n={n_pts}, LOO acc={loo_acc:.2f})", fontsize=TITLE_FONTSIZE)
    ax.set_ylim(-0.1, 1.1); ax.set_yticks([0, 0.5, 1.0])
    if i % 4 == 0:
        ax.set_ylabel(r"P(wet)", fontsize=AXIS_FONTSIZE)
    if i // 4 == 3:
        ax.set_xlabel("feature value", fontsize=AXIS_FONTSIZE)

    if i == 0:
        ax.legend(loc="lower right", frameon=False, fontsize=8)

    # per-axes colorbar (LaTeX-safe label)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=COLORBAR_SIZE, pad=COLORBAR_PAD)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.ax.tick_params(labelsize=7)
    cb.set_label(r"Soil moisture (\%)", fontsize=8)

# Improve spacing and add LaTeX-safe suptitle (avoid Unicode ≥ and %)
plt.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.08, wspace=WSPACE, hspace=HSPACE)
plt.suptitle(rf"1D Logistic Regression per feature  (wet if humidity $\geq$ {T_LABEL:.1f}\%)", y=0.98)

# --- Save (LaTeX-safe) ---
plt.savefig("logreg_per_feature_threshold28_grid.png", dpi=500, bbox_inches="tight")
plt.savefig("logreg_per_feature_threshold28_grid.pdf", bbox_inches="tight")
plt.show()

# =========================
# 6) Metrics table (also helpful)
# =========================
summary_rows = []
for feat in CANON_FEATURES:
    x_raw = df[feat].values.astype(float)
    mask = ~np.isnan(x_raw) & ~np.isnan(humidity)
    x = x_raw[mask]; y = y_all[mask]
    if len(x) < 2 or np.std(x) == 0 or len(np.unique(y)) < 2:
        summary_rows.append({"feature": feat, "loo_acc": np.nan, "coef_stdX": np.nan,
                             "intercept": np.nan, "x_star_p05": np.nan})
        continue
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("lr", LogisticRegression(penalty="l2", solver="lbfgs",
                                               C=1.0, max_iter=1000,
                                               class_weight="balanced", random_state=0))])
    try:
        loo_acc = cross_val_score(pipe, x.reshape(-1,1), y, cv=LeaveOneOut(),
                                  scoring="accuracy").mean()
    except Exception:
        loo_acc = np.nan
    pipe.fit(x.reshape(-1,1), y)
    coef = pipe.named_steps["lr"].coef_[0,0]
    inter = pipe.named_steps["lr"].intercept_[0]
    mu = x.mean(); sigma = x.std() if x.std() > 0 else 1.0
    x_star = mu + sigma * (-inter/coef) if coef != 0 else np.nan
    summary_rows.append({"feature": feat, "loo_acc": loo_acc, "coef_stdX": coef,
                         "intercept": inter, "x_star_p05": x_star})

summary = pd.DataFrame(summary_rows).sort_values("loo_acc", ascending=False)
with pd.option_context("display.max_rows", None, "display.max_colwidth", 70):
    print("\nLeave-one-out accuracy per feature (wet if humidity ≥ 28%):\n")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:0.3f}"))
summary.to_csv("logreg_per_feature_threshold28_metrics.csv", index=False)
print("\nSaved: logreg_per_feature_threshold28_grid.[png|pdf], "
      "logreg_per_feature_threshold28_metrics.csv")

# Print debug for any "constant feature" panels
if 'constant_debug' in locals() and constant_debug:
    print("\nFeatures flagged as constant/NaN — min/max values:")
    for feat, vmin, vmax in constant_debug:
        print(f"  {feat:28s}  min={vmin:.6g}, max={vmax:.6g}")
