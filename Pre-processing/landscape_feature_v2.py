import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.ticker import FuncFormatter  # <-- added

def tent_value(b, d, t):
    return np.maximum(0.0, np.minimum(t - b, d - t))

def load_pairs_or_demo(csv_path=None, seed=0):
    if csv_path is not None:
        try:
            df = pd.read_csv(csv_path)
            bd = df[['birth','death']].to_numpy(dtype=float)
            m = np.isfinite(bd[:,0]) & np.isfinite(bd[:,1]) & (bd[:,1] > bd[:,0])
            out = bd[m]
            if out.size > 0:
                return out, True
        except Exception:
            pass
    rng = np.random.default_rng(seed)
    births = np.round(np.sort(rng.uniform(0.3, 0.38, size=10)), 3)  # clustered to look like your H1
    deaths  = births + np.round(rng.uniform(0.01, 0.04, size=10), 3)
    return np.stack([births, deaths], axis=1), False

def plot_filled_tents(bd_pairs, title, y_gap=0.4, n_t=150, ny=15, max_features=20):
    bd_pairs = bd_pairs[:max_features]

    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111, projection='3d')

    for i, (b, d) in enumerate(bd_pairs):
        T = np.linspace(b, d, n_t)
        Z = tent_value(b, d, T)

        # Build a thicker ribbon (ny rows along Y) so it renders as a filled surface
        y0 = i * (1.0 + y_gap)
        y1 = y0 + 1.0
        Y_rows = np.linspace(y0, y1, ny)

        X = np.tile(T, (ny, 1))
        Y = np.tile(Y_rows[:, None], (1, n_t))
        Zsurf = np.tile(Z, (ny, 1))

        ax.plot_surface(X, Y, Zsurf, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False)  # solid fill, no wireframe lines

    ax.set_xlabel('t')
    ax.set_ylabel('feature index (separate lanes)')
    ax.set_zlabel('tent height')
    ax.set_title(title)
    ax.view_init(elev=15, azim=-60)

    # --------- ROUND TICK LABELS (only change) ----------
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.2f}"))  # X (t) -> 3 decimals
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(round(v))}"))  # Y -> integers
    ax.zaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.2f}"))  # Z (height) -> 3 decimals
    # Change .3f to .2f, .4f, etc. if you want different precision
    # ----------------------------------------------------

    plt.tight_layout()
    plt.show()

# Try to use your files if they exist; otherwise demo data
h0_path = "C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/3D_hump_3_H0.csv"
h1_path = "C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Pre-processing/3D_hump_3_H1.csv"
H0_pairs, have_h0 = load_pairs_or_demo(h0_path, seed=1)
H1_pairs, have_h1 = load_pairs_or_demo(h1_path, seed=2)

# Plot filled tents
plot_filled_tents(H1_pairs, title=f"H1 tents — filled{' (demo)' if not have_h1 else ''}", y_gap=0.4, n_t=150, ny=21, max_features=20)
plot_filled_tents(H0_pairs, title=f"H0 tents — filled{' (demo)' if not have_h0 else ''}", y_gap=0.4, n_t=150, ny=21, max_features=20)
