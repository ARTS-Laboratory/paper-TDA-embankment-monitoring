
# pca_dates_humidity.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression

# =========================
# 0) SETTINGS — EDIT THESE
# =========================
file_path = r"C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/TDA_PCA_template.xlsx"   # or .csv
excel_header_row = 0   # set to 1 if your Excel has headers on the 2nd row
EPS = 1e-12

# Exact column names expected (match the template)
feature_cols = [
    "persistence entropy (H0)","persistence entropy (H1)",
    "number of points (H0)","number of points(H1)",
    "bottleneck  (H0)","bottleneck  (H1)",
    "wasserstein  (H0)","wasserstein  (H1)",
    "landscape  (H0)","landscape  (H1)",
    "persistence image  (H0)","persistence image  (H1)",
    "Betti  (H0)","Betti  (H1)",
    "heat  (H0)","heat  (H1)"
]

# =========================
# 1) LOAD DATA
# =========================
if file_path.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(file_path, header=excel_header_row)
elif file_path.lower().endswith(".csv"):
    df = pd.read_csv(file_path)
else:
    raise ValueError("Unsupported file type. Use .xlsx, .xls, or .csv")

required = ["date", "humidity_percent"] + feature_cols
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Extract arrays
dates = df["date"].astype(str).to_numpy()
humidity = pd.to_numeric(df["humidity_percent"], errors="coerce").to_numpy()
X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy()  # (n_dates, 16)

# =========================
# 2) CLEAN / STANDARDIZE SAFELY
# =========================
# Fill NaNs by column medians
col_med = np.nanmedian(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_med, inds[1])

# Drop constant features (std ~ 0)
std = X.std(axis=0, ddof=0)
keep_mask = std >= EPS
dropped = [c for c, k in zip(feature_cols, keep_mask) if not k]
if dropped:
    print("Dropped constant features:", dropped)

Xk = X[:, keep_mask]
if Xk.shape[1] == 0:
    raise ValueError("All features are constant across dates; PCA is undefined.")

# Manual z-score (safe)
mu = Xk.mean(axis=0)
sg = Xk.std(axis=0, ddof=0)
sg[sg < EPS] = 1.0
Xz = (Xk - mu) / sg

# Degenerate check: if all rows identical after scaling
if np.linalg.norm(Xz) < EPS:
    print("All dates identical after scaling — plotting a degenerate map.")
    Z = np.zeros((Xz.shape[0], 2))
    evr = np.array([0.0, 0.0])
else:
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xz)                 # (n_dates, 2)
    S2 = getattr(pca, "explained_variance_", None)
    evr = pca.explained_variance_ratio_ if (S2 is not None and S2.sum() >= EPS) else np.array([0.0, 0.0])

# =========================
# 3) OPTIONAL: CLUSTERING IN PC SPACE
# =========================
labels = None
centers = np.empty((0, 2))
sil = np.nan
# Only try clustering if there's >1 distinct location
if len({tuple(xy) for xy in np.round(Z, 6)}) > 1:
    try:
        km = KMeans(n_clusters=2, n_init=20, random_state=0)
        labels = km.fit_predict(Z)
        centers = km.cluster_centers_
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(Z, labels)
    except Exception as e:
        print("KMeans note:", e)

# =========================
# 4) HUMIDITY ASSOCIATION (SAFE)
# =========================
def safe_corr(x, y, eps=EPS):
    if np.std(x) < eps or np.std(y) < eps:
        return np.nan
    return np.corrcoef(x, y)[0, 1]

pc1_corr = safe_corr(Z[:, 0], humidity)
pc2_corr = safe_corr(Z[:, 1], humidity)

# humidity ~ PC1 + PC2 regression (for arrow)
r2 = np.nan
vec = np.zeros(2)
try:
    if np.std(humidity) >= EPS and np.linalg.norm(Z.std(axis=0)) >= EPS:
        reg = LinearRegression().fit(Z, humidity)
        r2 = reg.score(Z, humidity)
        beta = reg.coef_
        bnorm = np.linalg.norm(beta)
        if bnorm >= EPS:
            muZ = Z.mean(axis=0)
            vec = (beta / bnorm) * (0.8 * np.linalg.norm(Z.std(axis=0)))
except Exception as e:
    print("Humidity regression note:", e)

# =========================
# 5) MAKE OVERLAP VISIBLE (TINY OFFSETS)
# =========================
Z_plot = Z.copy()
Zr = np.round(Z, 6)
seen = {}
for i, xy in enumerate(map(tuple, Zr)):
    seen.setdefault(xy, []).append(i)
for xy, idxs in seen.items():
    if len(idxs) > 1:
        for k, j in enumerate(idxs):
            Z_plot[j] += np.array([0.03 * k, -0.03 * k])  # small deterministic offset

# =========================
# 6) PLOT PC1–PC2 (COLOR = HUMIDITY)
# =========================
fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(Z_plot[:, 0], Z_plot[:, 1], c=humidity, s=90)  # default cmap; color encodes humidity
for i, d in enumerate(dates):
    ax.text(Z_plot[i, 0], Z_plot[i, 1], f" {d}", fontsize=8, va="center")

if centers.size:
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=120)

# draw humidity arrow only if meaningful
if np.linalg.norm(vec) > EPS:
    muZ = Z.mean(axis=0)
    ax.arrow(muZ[0], muZ[1], vec[0], vec[1], width=0.02, length_includes_head=True)

ax.axhline(0, lw=0.8); ax.axvline(0, lw=0.8)
ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
ax.set_title("Dates in PCA space (color = humidity)")
cb = plt.colorbar(sc, ax=ax); cb.set_label("Humidity (%)")
plt.tight_layout()
plt.savefig("dates_pca_humidity.png", dpi=300)
plt.show()

# =========================
# 7) SAVE SCORES & PRINT DIAGNOSTICS
# =========================
scores_df = pd.DataFrame({"date": dates, "PC1": Z[:, 0], "PC2": Z[:, 1], "humidity_percent": humidity})
scores_df.to_csv("dates_pca_scores.csv", index=False)

print(f"Explained variance: PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}%  (total={evr.sum()*100:.1f}%)")
if labels is not None:
    print(f"KMeans k=2 silhouette: {sil:.3f}")
print("Dropped constant features:", dropped if dropped else "None")
print(f"corr(PC1, humidity) = {pc1_corr:.3f}, corr(PC2, humidity) = {pc2_corr:.3f},  R^2(humidity ~ PC1+PC2) = {r2:.3f}")
print("Wrote: dates_pca_humidity.png, dates_pca_scores.csv")






