"""
Stage 25: Compare no filtering, Gaussian filtering, and median filtering
on the same noisy synthetic GPR B-scan.

Educational synthetic model only — not measured radar data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import find_peaks
from scipy.optimize import least_squares

# Reproducible simulation settings
RNG_SEED = 25
NOISE_STD = 0.08
C0 = 3.0e8
EPS_R = 9.0
VELOCITY = C0 / np.sqrt(EPS_R)
FS = 10e9
DT = 1.0 / FS
T_WINDOW = 75e-9
N_TIME = int(round(T_WINDOW * FS))
TIME = np.arange(N_TIME) * DT
TIME_NS = TIME * 1e9
SYSTEM_DELAY_NS = 1.0
X_POSITIONS = np.linspace(0.0, 5.0, 101)
TARGETS = [
    {"x": 1.5, "depth": 0.15, "amplitude": 0.40, "sigma_ns": 0.28},
    {"x": 3.5, "depth": 0.30, "amplitude": 0.30, "sigma_ns": 0.38},
]

def make_trace(x):
    trace = np.zeros(N_TIME, dtype=float)
    for target in TARGETS:
        dx = x - target["x"]
        path = np.sqrt(target["depth"]**2 + dx**2)
        arrival_s = SYSTEM_DELAY_NS * 1e-9 + 2.0 * path / VELOCITY
        sigma_s = target["sigma_ns"] * 1e-9
        trace += target["amplitude"] * np.exp(
            -0.5 * ((TIME - arrival_s) / sigma_s) ** 2
        )
    trace += 0.025 * np.exp(-TIME / 3e-9) * np.sin(2*np.pi*180e6*TIME)
    return trace

def travel_time_ns(x, x0, depth):
    return SYSTEM_DELAY_NS + 2.0 * np.sqrt(depth**2 + (x - x0)**2) / VELOCITY * 1e9

def detect_peaks(bscan):
    detections = []
    for ix, x in enumerate(X_POSITIONS):
        peaks, _ = find_peaks(
            bscan[ix], height=0.06, prominence=0.04, distance=50
        )
        for p in peaks:
            if TIME_NS[p] > SYSTEM_DELAY_NS + 0.5:
                detections.append([x, TIME_NS[p], bscan[ix, p]])
    return np.asarray(detections, dtype=float).reshape(-1, 3)

def fit_targets(detections):
    fits = []
    remaining = detections.copy()
    for target_id, (seed_x, seed_depth) in enumerate([(1.5, 0.15), (3.5, 0.30)], 1):
        if len(remaining) < 5:
            fits.append((target_id, np.nan, np.nan, np.nan, 0))
            continue
        expected = travel_time_ns(remaining[:, 0], seed_x, seed_depth)
        candidates = remaining[np.abs(remaining[:, 1] - expected) < 3.5]
        if len(candidates) < 5:
            order = np.argsort(np.abs(remaining[:, 1] - expected))
            candidates = remaining[order[:min(30, len(remaining))]]
        def residual(params):
            return travel_time_ns(candidates[:, 0], params[0], params[1]) - candidates[:, 1]
        result = least_squares(
            residual, [seed_x, seed_depth], bounds=([0.0, 0.02], [5.0, 1.0]),
            loss="soft_l1", f_scale=0.5, max_nfev=5000
        )
        x_est, d_est = result.x
        rmse = float(np.sqrt(np.mean(residual(result.x)**2)))
        fits.append((target_id, float(x_est), float(d_est), rmse, len(candidates)))
        keep = np.abs(remaining[:, 1] - travel_time_ns(remaining[:, 0], x_est, d_est)) > 1.5
        if np.count_nonzero(keep) >= 5:
            remaining = remaining[keep]
    return fits

def valid(fit):
    return np.isfinite(fit[1]) and np.isfinite(fit[2])

# Generate one clean scan and one noisy realization shared by all methods.
clean = np.vstack([make_trace(x) for x in X_POSITIONS])
rng = np.random.default_rng(RNG_SEED)
noisy = clean + rng.normal(0.0, NOISE_STD, size=clean.shape)

methods = {
    "No filtering": noisy,
    "Gaussian": gaussian_filter(noisy, sigma=(1.0, 1.5), mode="nearest"),
    "Median": median_filter(noisy, size=(3, 3), mode="nearest"),
}

results = {}
print("===== STAGE 25: FILTER COMPARISON =====")
print(f"Noise std={NOISE_STD:.3f}; random seed={RNG_SEED}")
print("Method | Peak count | Target | x estimate (m) | depth estimate (m) | RMSE (ns)")
for method_name, scan in methods.items():
    detections = detect_peaks(scan)
    fits = fit_targets(detections)
    results[method_name] = (scan, detections, fits)
    for fit in fits:
        target_id, x_est, d_est, rmse, npoints = fit
        print(
            f"{method_name:13s} | {len(detections):10d} | {target_id:6d} | "
            f"{x_est:14.3f} | {d_est:18.3f} | {rmse:9.3f}"
        )

print("\nReference values: T1 x=1.500 m, depth=0.150 m; T2 x=3.500 m, depth=0.300 m.")
print("Model-informed fitting is used; these results do not establish real-world GPR performance.")

# Plot three B-scans for visual comparison.
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
for ax, (name, scan) in zip(axes, methods.items()):
    im = ax.imshow(
        scan.T, aspect="auto", origin="lower",
        extent=[X_POSITIONS[0], X_POSITIONS[-1], TIME_NS[0], TIME_NS[-1]],
        cmap="gray", vmin=-0.10, vmax=0.45
    )
    ax.set_ylim(0, 75)
    ax.set_title(name)
    ax.set_xlabel("Antenna position (m)")
    ax.set_ylabel("Arrival time (ns)")
fig.colorbar(im, ax=axes, label="Reflection amplitude", shrink=0.82)
fig.savefig("stage25_filter_comparison.png", dpi=160)
plt.show()

# Plot peak counts and fit RMSEs.
names = list(methods.keys())
counts = [len(results[n][1]) for n in names]
rmse_matrix = []
for target_idx in [0, 1]:
    rmse_matrix.append([
        results[n][2][target_idx][3] for n in names
    ])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
axes[0].bar(names, counts)
axes[0].set_title("Detected peak count")
axes[0].set_ylabel("Number of peaks")
axes[0].tick_params(axis="x", rotation=15)
for idx, target_id in enumerate([1, 2]):
    axes[1].plot(names, rmse_matrix[idx], marker="o", label=f"Target {target_id}")
axes[1].set_title("Fit RMSE by method")
axes[1].set_ylabel("RMSE (ns)")
axes[1].tick_params(axis="x", rotation=15)
axes[1].grid(True)
axes[1].legend()
fig.savefig("stage25_filter_metrics.png", dpi=160)
plt.show()

# Save machine-readable comparison.
with open("stage25_filter_summary.csv", "w", encoding="utf-8") as f:
    f.write("method,peak_count,target_id,x_est_m,depth_est_m,fit_rmse_ns,fit_points\n")
    for name in names:
        scan, detections, fits = results[name]
        for fit in fits:
            target_id, x_est, d_est, rmse, points = fit
            f.write(f"{name},{len(detections)},{target_id},{x_est},{d_est},{rmse},{points}\n")

print("\nSaved:")
print("  stage25_filter_comparison.png")
print("  stage25_filter_metrics.png")
print("  stage25_filter_summary.csv")
