"""
Stage 27: Iterative curve fitting with outlier rejection
Synthetic educational GPR model only — not measured radar data.

Uses a noisy synthetic B-scan, Gaussian smoothing, then iterative fitting:
1. collect local peaks near an approximate target travel-time curve
2. fit the hyperbola robustly
3. reject points with large time residuals
4. refit until stable

Because approximate target locations/depths initialize the search, this is not blind localization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.optimize import least_squares

SEED = 27
NOISE_STD = 0.08
C0 = 3e8
EPS_R = 9.0
V = C0 / np.sqrt(EPS_R)
FS = 10e9
DT = 1 / FS
TMAX = 75e-9
N = int(round(TMAX * FS))
TIME = np.arange(N) * DT
TIME_NS = TIME * 1e9
T0_NS = 1.0
X = np.linspace(0, 5, 101)
TARGETS = [
    {"x": 1.5, "depth": 0.15, "amp": 0.40, "sigma_ns": 0.28},
    {"x": 3.5, "depth": 0.30, "amp": 0.30, "sigma_ns": 0.38},
]

def make_trace(x):
    y = np.zeros(N)
    for t in TARGETS:
        path = np.sqrt(t["depth"]**2 + (x-t["x"])**2)
        arrival = T0_NS*1e-9 + 2*path/V
        sigma = t["sigma_ns"]*1e-9
        y += t["amp"]*np.exp(-0.5*((TIME-arrival)/sigma)**2)
    y += 0.025*np.exp(-TIME/3e-9)*np.sin(2*np.pi*180e6*TIME)
    return y

def tt_ns(x, x0, depth):
    return T0_NS + 2*np.sqrt(depth**2 + (x-x0)**2)/V*1e9

def extract_peaks(scan):
    pts = []
    for i, x in enumerate(X):
        p, _ = find_peaks(scan[i], height=0.06, prominence=0.04, distance=50)
        for k in p:
            if TIME_NS[k] > T0_NS + 0.5:
                pts.append([x, TIME_NS[k], scan[i, k]])
    return np.asarray(pts, dtype=float).reshape(-1, 3)

def iterative_fit(points, seed_x, seed_d, initial_gate_ns=2.0, reject_ns=0.45):
    if len(points) < 5:
        return None
    initial = tt_ns(points[:, 0], seed_x, seed_d)
    selected = points[np.abs(points[:, 1]-initial) <= initial_gate_ns]
    if len(selected) < 8:
        return None

    params = np.array([seed_x, seed_d], dtype=float)
    for _ in range(8):
        def residual(p):
            return tt_ns(selected[:, 0], p[0], p[1]) - selected[:, 1]
        fit = least_squares(
            residual, params, bounds=([0.0, 0.02], [5.0, 1.0]),
            loss="soft_l1", f_scale=0.20, max_nfev=5000
        )
        params = fit.x
        errors = np.abs(residual(params))
        keep = errors <= reject_ns
        if np.count_nonzero(keep) < 8:
            # Preserve the eight closest points if the threshold is too strict.
            keep[np.argsort(errors)[:8]] = True
        if np.all(keep):
            break
        selected = selected[keep]

    final_residual = tt_ns(selected[:, 0], params[0], params[1]) - selected[:, 1]
    rmse = float(np.sqrt(np.mean(final_residual**2)))
    return {
        "x": float(params[0]), "depth": float(params[1]),
        "rmse": rmse, "points": selected,
        "n": len(selected)
    }

clean = np.vstack([make_trace(x) for x in X])
rng = np.random.default_rng(SEED)
noisy = clean + rng.normal(0, NOISE_STD, size=clean.shape)
filtered = gaussian_filter(noisy, sigma=(1.0, 1.5), mode="nearest")
peaks = extract_peaks(filtered)

fits = [
    iterative_fit(peaks, 1.5, 0.15),
    iterative_fit(peaks, 3.5, 0.30),
]

print("===== STAGE 27: ITERATIVE OUTLIER REJECTION =====")
print(f"Noise standard deviation: {NOISE_STD:.3f}")
print(f"Peak points before curve fitting: {len(peaks)}")
for i, fit in enumerate(fits, 1):
    target = TARGETS[i-1]
    if fit is None:
        print(f"Target {i}: fit failed; inspect thresholds or widen initial gate.")
        continue
    print(
        f"Target {i}: x={fit['x']:.3f} m (reference {target['x']:.3f}), "
        f"depth={fit['depth']:.3f} m (reference {target['depth']:.3f}), "
        f"RMSE={fit['rmse']:.3f} ns, retained points={fit['n']}"
    )
print("Note: approximate target geometry initializes each fit; this is not blind localization.")

# Visualize detections and selected inliers.
plt.figure(figsize=(10.5, 6))
plt.imshow(filtered.T, aspect="auto", origin="lower",
           extent=[X[0], X[-1], TIME_NS[0], TIME_NS[-1]],
           cmap="gray", vmin=-0.10, vmax=0.45)
if len(peaks):
    plt.scatter(peaks[:, 0], peaks[:, 1], s=7, c="orange", alpha=0.20,
                label="All filtered peaks")
colors = ["cyan", "yellow"]
x_curve = np.linspace(0, 5, 501)
for i, fit in enumerate(fits):
    if fit is None:
        continue
    pts = fit["points"]
    plt.scatter(pts[:, 0], pts[:, 1], s=10, c=colors[i],
                label=f"Target {i+1} inliers ({fit['n']})")
    plt.plot(x_curve, tt_ns(x_curve, fit["x"], fit["depth"]),
             color=colors[i], linewidth=1.7,
             label=f"Fit T{i+1}: x={fit['x']:.2f} m, d={fit['depth']:.2f} m")
plt.ylim(0, 75)
plt.xlabel("Antenna position (m)")
plt.ylabel("Arrival time (ns)")
plt.title("Stage 27 — Iterative Curve Fit and Outlier Rejection")
plt.colorbar(label="Reflection amplitude")
plt.legend(loc="upper center", fontsize=8)
plt.tight_layout()
plt.savefig("stage27_iterative_fit.png", dpi=160)
plt.show()

with open("stage27_target_estimates.csv", "w", encoding="utf-8") as f:
    f.write("target,x_est_m,depth_est_m,fit_rmse_ns,inlier_points\n")
    for i, fit in enumerate(fits, 1):
        if fit is None:
            f.write(f"{i},nan,nan,nan,0\n")
        else:
            f.write(f"{i},{fit['x']},{fit['depth']},{fit['rmse']},{fit['n']}\n")

print("\nSaved:")
print("  stage27_iterative_fit.png")
print("  stage27_target_estimates.csv")
