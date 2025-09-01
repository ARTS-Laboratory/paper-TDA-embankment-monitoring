r"C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/Abnormalities_Features-V5-2.xlsx"


# =========================
# PCA + Logistic Regression (PC1–PC2 + PC1-only) + Scree + PC1 Loadings
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -------- CONFIG --------
XLS_PATH = r"C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/Abnormalities_Features-V5-2.xlsx"
USE_LAST_N = 16
COLORBAR_RANGE = (-5.0, 5.0)
VMIN, VMAX = COLORBAR_RANGE
RANDOM_SEED = 42

# ✅ ADD: your exact tick list (and only use this)
COLORBAR_TICKS = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

# -------- Load Excel --------
def load_with_index_col(path):
    for header in (1, 0):
        df = pd.read_excel(path, header=header)
        df.columns = [str(c).strip() for c in df.columns]
        idx_candidates = [c for c in df.columns if "index" in c.lower()]
        if idx_candidates:
            return df, idx_candidates[0]
    raise ValueError("Could not find a column with name containing 'index'.")

df_raw, index_col = load_with_index_col(XLS_PATH)

numeric_cols_all = [c for c in df_raw.columns
                    if c != index_col and pd.api.types.is_numeric_dtype(df_raw[c])]
df_all = df_raw[[index_col] + numeric_cols_all].replace([np.inf, -np.inf], np.nan).dropna()

if USE_LAST_N is not None:
    df = df_all.tail(USE_LAST_N).copy()
    subtitle_rows = f"{USE_LAST_N} most recent samples"
else:
    df = df_all.copy()
    subtitle_rows = "All samples"

idx_vals = df[index_col].astype(float).to_numpy()
y = (idx_vals > 0).astype(int)
X = df[numeric_cols_all].to_numpy()

# -------- Standardize + PCA --------
scaler = StandardScaler().fit(X)
X_std = scaler.transform(X)

ncomp = min(X_std.shape[0], X_std.shape[1])
pca = PCA(n_components=ncomp, random_state=RANDOM_SEED).fit(X_std)

Z_all = pca.transform(X_std)
Z = Z_all[:, :2]
pc1_var = 100 * pca.explained_variance_ratio_[0]
pc2_var = 100 * pca.explained_variance_ratio_[1]

def label_contour_upright(contour, txt, color):
    try:
        path = contour.collections[0].get_paths()[0].vertices
        i = path.shape[0] // 2
        x0, y0 = path[i]; x1, y1 = path[min(i+1, path.shape[0]-1)]
        ang = np.degrees(np.arctan2(y1 - y0, x1 - x0))
        if ang > 90:  ang -= 180
        if ang < -90: ang += 180
        plt.text(x0, y0, txt, color=color, fontsize=10, rotation=ang,
                 ha="center", va="center", rotation_mode="anchor",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))
    except Exception:
        pass

# =========================
# (1) PC1–PC2 LR surface
# =========================
plt.figure(figsize=(8, 6))
if len(np.unique(y)) >= 2:
    lr2d = LogisticRegression(
        penalty="l2", solver="liblinear", C=1.0,
        class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
    ).fit(Z, y)

    pad = 0.6
    x_min, x_max = Z[:,0].min() - pad, Z[:,0].max() + pad
    y_min, y_max = Z[:,1].min() - pad, Z[:,1].max() + pad

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    proba = lr2d.predict_proba(grid)[:, 1].reshape(xx.shape)

    plt.contourf(xx, yy, proba, levels=[0.0, 0.5, 1.0],
                 colors=["#f3e7d8", "#dbe8fb"], alpha=0.35)
    c025 = plt.contour(xx, yy, proba, levels=[0.25], linestyles="--", linewidths=1.6, colors="black")
    c075 = plt.contour(xx, yy, proba, levels=[0.75], linestyles="--", linewidths=1.6, colors="black")
    c050 = plt.contour(xx, yy, proba, levels=[0.50], linestyles="--", linewidths=2.0, colors="red")
    label_contour_upright(c025, "0.25", "black")
    label_contour_upright(c050, "0.5",  "red")
    label_contour_upright(c075, "0.75", "black")

sc = plt.scatter(Z[:,0], Z[:,1], c=idx_vals, cmap="viridis",
                 vmin=VMIN, vmax=VMAX, s=90, marker="o",
                 edgecolor="black", linewidth=0.7)
cb = plt.colorbar(sc, pad=0.02)
cb.set_label("Index size (− cavities  •  + humps)")
# ✅ USE your list of ticks here
cb.set_ticks(COLORBAR_TICKS)
cb.ax.set_yticklabels([f"{t:g}" for t in COLORBAR_TICKS])
cb.update_ticks()

plt.axhline(0.0, color="gray", linewidth=1.0, alpha=0.7)
plt.axvline(0.0, color="gray", linewidth=1.0, alpha=0.7)
plt.xlabel(f"PC1 ({pc1_var:.1f}% var)")
plt.ylabel(f"PC2 ({pc2_var:.1f}% var)")
plt.title(f"PCA(2) — LR probability surface ({subtitle_rows})")
plt.tight_layout()
plt.show()

# =========================
# (2) PC1-only Logistic Regression
# =========================
pc1 = Z_all[:, [0]]
lr1d = LogisticRegression(
    penalty="l2", solver="liblinear", C=1.0,
    class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
).fit(pc1, y)

def x_at_p(p):
    w = lr1d.coef_[0,0]; b = lr1d.intercept_[0]
    logit = np.log(p/(1.0-p))
    return (logit - b) / w

x25, x50, x75 = x_at_p(0.25), x_at_p(0.50), x_at_p(0.75)
xs = np.linspace(pc1.min()-0.3*(pc1.max()-pc1.min()+1e-9),
                 pc1.max()+0.3*(pc1.max()-pc1.min()+1e-9), 400).reshape(-1,1)
ps = lr1d.predict_proba(xs)[:,1]
p_hat = lr1d.predict_proba(pc1)[:,1]

plt.figure(figsize=(9, 5.5))
plt.plot(xs.ravel(), ps, '-', lw=2, label="Logistic probability (class=1)")
sc2 = plt.scatter(pc1.ravel(), p_hat, c=idx_vals, cmap="viridis",
                  vmin=VMIN, vmax=VMAX, s=90, marker="o",
                  edgecolor="black", linewidth=0.7, label="Samples")
cb2 = plt.colorbar(sc2, pad=0.02)
cb2.set_label("Index size (− cavities  •  + humps)")
# ✅ USE your list of ticks here
cb2.set_ticks(COLORBAR_TICKS)
cb2.ax.set_yticklabels([f"{t:g}" for t in COLORBAR_TICKS])
cb2.update_ticks()

for xv, txt, col, lw, yy in [
    (x25, "0.25", "black", 1.6, 0.25),
    (x50, "0.5",  "red",   2.0, 0.50),
    (x75, "0.75", "black", 1.6, 0.75),
]:
    plt.axvline(xv, linestyle="--", color=col, linewidth=lw)
    plt.text(xv, yy+0.04, txt, color=col, rotation=0, ha="center",
             va="bottom", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))

plt.ylim(-0.02, 1.02)
plt.xlabel(f"PC1 score ({pc1_var:.1f}% var)")
plt.ylabel("P(class = humps)")
plt.title(f"Logistic regression on PC1 only ({subtitle_rows})")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# =========================
# (3) Scree + (4) PC1 loadings — your original code continues unchanged
# =========================

eigvals = pca.explained_variance_
dims = np.arange(1, len(eigvals) + 1)

plt.figure(figsize=(8, 5))
plt.bar(dims, eigvals, alpha=0.85, color="tab:blue")
plt.plot(dims, eigvals, "-ok")
plt.xticks(dims)
plt.xlabel("Dimension")
plt.ylabel("Eigenvalue")
plt.title("Scree plot")
for x, yv in zip(dims, eigvals):
    if yv > 0:
        plt.text(x, yv + 0.05, f"{yv:.1f}", ha="center", va="bottom", fontsize=9)
plt.axhline(1.0, linestyle="--", color="red")
plt.text(dims[-1] + 0.2, 1.02, "Kaiser = 1", color="red", fontsize=10, ha="left", va="bottom")
plt.tight_layout()
plt.show()

pc1_vec = pca.components_[0]
feat_info = []
for feat, coef in zip(numeric_cols_all, pc1_vec):
    if feat.lower().startswith("unnamed"):
        continue
    if "(H0)" in feat:
        grp = "H0"
    elif "(H1)" in feat:
        grp = "H1"
    else:
        grp = "H1" if "H1" in feat else "H0"
    base = feat.replace("(H0)", "").replace("(H1)", "").strip()
    feat_info.append((base, abs(coef), grp))

feat_info.sort(key=lambda t: t[1], reverse=True)

labels = [f"{base}\n({grp})" for base, _, grp in feat_info]
vals   = [v for _, v, _ in feat_info]
colors = ["tab:blue" if grp == "H0" else "tab:orange" for _, _, grp in feat_info]

plt.figure(figsize=(12, 5.5))
plt.bar(range(len(vals)), vals, color=colors)
plt.xticks(range(len(labels)), labels, rotation=90, ha="center", va="top")
plt.ylabel("PC1 |loading| (importance)")
plt.title("PC1 loadings by feature (H0=blue, H1=orange)")
plt.tight_layout()
plt.show()

print(f"Index column: {index_col}")
print(f"Rows used: {len(df)} ({subtitle_rows})")
print(f"PCA variance: PC1={pc1_var:.1f}%, PC2={pc2_var:.1f}%")
print("Top 5 features by |PC1 loading|:")
for base, v, grp in feat_info[:5]:
    print(f"  {base:40s}  {v:.3f}  [{grp}]")
