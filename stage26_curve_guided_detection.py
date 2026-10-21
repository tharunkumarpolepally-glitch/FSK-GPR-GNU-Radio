"""
Stage 26: Gaussian filtering + target-curve-guided peak selection
Synthetic educational GPR model only — not measured radar data.

This stage compares:
1) all local peaks after Gaussian filtering
2) peak points that lie near either fitted travel-time curve
3) target curves refitted using only gated points

IMPORTANT: the target gates are initialized from approximate target geometry.
This demonstrates clutter rejection, not blind unknown-target discovery.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.optimize import least_squares

RNG_SEED = 26
NOISE_STD = 0.08
C0 = 3.0e8
EPS_R = 9.0
VELOCITY = C0 / np.sqrt(EPS_R)
FS = 10e9
DT = 1 / FS
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
    trace = np.zeros(N_TIME)
    for target in TARGETS:
        dx = x - target["x"]
        path = np.sqrt(target["depth"]**2 + dx**2)
        arrival = SYSTEM_DELAY_NS * 1e-9 + 2 * path / VELOCITY
        sigma = target["sigma_ns"] * 1e-9
        trace += target["amplitude"] * np.exp(-0.5 * ((TIME-arrival)/sigma)**2)
    trace += 0.025 * np.exp(-TIME/3e-9) * np.sin(2*np.pi*180e6*TIME)
    return trace

def travel_time_ns(x, x0, depth):
    return SYSTEM_DELAY_NS + 2*np.sqrt(depth**2 + (x-x0)**2)/VELOCITY*1e9

def get_peaks(scan, height=0.06, prominence=0.04):
    points = []
    for ix, x in enumerate(X_POSITIONS):
        peaks, _ = find_peaks(scan[ix], height=height, prominence=prominence, distance=50)
        for p in peaks:
            if TIME_NS[p] > SYSTEM_DELAY_NS + 0.5:
                points.append([x, TIME_NS[p], scan[ix, p]])
    return np.asarray(points, dtype=float).reshape(-1, 3)

def fit_curve(points, seed_x, seed_depth, gate_ns=3.5):
    if len(points) < 5:
        return None
    pred = travel_time_ns(points[:, 0], seed_x, seed_depth)
    candidates = points[np.abs(points[:, 1] - pred) < gate_ns]
    if len(candidates) < 5:
        return None

    def residual(params):
        return travel_time_ns(candidates[:, 0], params[0], params[1]) - candidates[:, 1]

    result = least_squares(
        residual, [seed_x, seed_depth],
        bounds=([0.0, 0.02], [5.0, 1.0]),
        loss="soft_l1", f_scale=0.5, max_nfev=5000
    )
    x_est, depth_est = result.x
    rmse = float(np.sqrt(np.mean(residual(result.x)**2)))
    return {"x": float(x_est), "depth": float(depth_est), "rmse": rmse,
            "points": candidates, "n": len(candidates)}

# Create clean/noisy/filtered scans.
clean = np.vstack([make_trace(x) for x in X_POSITIONS])
rng = np.random.default_rng(RNG_SEED)
noisy = clean + rng.normal(0, NOISE_STD, size=clean.shape)
filtered = gaussian_filter(noisy, sigma=(1.0, 1.5), mode="nearest")

all_peaks = get_peaks(filtered)
seeds = [(1.5, 0.15), (3.5, 0.30)]
fits = []
gated_points = []
for seed_x, seed_depth in seeds:
    fit = fit_curve(all_peaks, seed_x, seed_depth)
    fits.append(fit)
    if fit is not None:
        gated_points.append(fit["points"])

if gated_points:
    gated = np.vstack(gated_points)
    # Deduplicate points if any detection was assigned to both tracks.
    gated = np.unique(gated, axis=0)
else:
    gated = np.empty((0, 3))

print("===== STAGE 26: CURVE-GUIDED PEAK SELECTION =====")
print(f"Noise standard deviation: {NOISE_STD:.3f}")
print(f"Peaks after Gaussian filtering: {len(all_peaks)}")
print(f"Peak points retained by target-curve gates: {len(gated)}")
if len(all_peaks):
    print(f"Reduction in peak points: {(1-len(gated)/len(all_peaks))*100:.1f}%")
print("\nFitted target estimates:")
for i, fit in enumerate(fits, 1):
    if fit is None:
        print(f"Target {i}: fit failed; widen the gate or adjust peak threshold.")
    else:
        t = TARGETS[i-1]
        print(f"Target {i}: x_est={fit['x']:.3f} m (reference {t['x']:.3f} m), "
              f"depth_est={fit['depth']:.3f} m (reference {t['depth']:.3f} m), "
              f"RMSE={fit['rmse']:.3f} ns, points={fit['n']}")
print("\nCaution: target-like initial guesses are used, so this is not blind localization.")
print("Synthetic model only — not measured radar data.")

# B-scan with selected points overlaid.
plt.figure(figsize=(10.5, 6))
plt.imshow(filtered.T, aspect="auto", origin="lower",
           extent=[X_POSITIONS[0], X_POSITIONS[-1], TIME_NS[0], TIME_NS[-1]],
           cmap="gray", vmin=-0.10, vmax=0.45)
if len(all_peaks):
    plt.scatter(all_peaks[:, 0], all_peaks[:, 1], s=7, c="orange",
                alpha=0.30, label="All filtered peaks")
if len(gated):
    plt.scatter(gated[:, 0], gated[:, 1], s=9, c="lime",
                label="Curve-gated points")
x_curve = np.linspace(0, 5, 501)
line_colors = ["cyan", "yellow"]
for i, fit in enumerate(fits):
    if fit is not None:
        plt.plot(x_curve, travel_time_ns(x_curve, fit["x"], fit["depth"]),
                 color=line_colors[i], linewidth=1.7,
                 label=f"Target {i+1}: x={fit['x']:.2f} m, d={fit['depth']:.2f} m")
plt.ylim(0, 75)
plt.xlabel("Antenna position (m)")
plt.ylabel("Arrival time (ns)")
plt.title("Stage 26 — Curve-Guided Peak Selection")
plt.colorbar(label="Reflection amplitude")
plt.legend(loc="upper center", fontsize=8)
plt.tight_layout()
plt.savefig("stage26_curve_guided_bscan.png", dpi=160)
plt.show()

# Count comparison plot.
plt.figure(figsize=(7.5, 4.5))
plt.bar(["All filtered peaks", "Curve-gated points"], [len(all_peaks), len(gated)])
plt.ylabel("Peak points")
plt.title("Stage 26 — Peak-Point Reduction")
plt.tight_layout()
plt.savefig("stage26_peak_reduction.png", dpi=160)
plt.show()

# CSV output
np.savetxt("stage26_all_filtered_peaks.csv", all_peaks, delimiter=",",
           header="antenna_x_m,peak_time_ns,peak_amplitude", comments="")
np.savetxt("stage26_curve_gated_peaks.csv", gated, delimiter=",",
           header="antenna_x_m,peak_time_ns,peak_amplitude", comments="")
with open("stage26_target_estimates.csv", "w", encoding="utf-8") as f:
    f.write("target,x_est_m,depth_est_m,fit_rmse_ns,points_used\n")
    for i, fit in enumerate(fits, 1):
        if fit is None:
            f.write(f"{i},nan,nan,nan,0\n")
        else:
            f.write(f"{i},{fit['x']},{fit['depth']},{fit['rmse']},{fit['n']}\n")

print("\nSaved:")
print("  stage26_curve_guided_bscan.png")
print("  stage26_peak_reduction.png")
print("  stage26_all_filtered_peaks.csv")
print("  stage26_curve_gated_peaks.csv")
print("  stage26_target_estimates.csv")
