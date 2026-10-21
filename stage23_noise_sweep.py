"""
Stage 23: Multi-noise robustness sweep for synthetic GPR localization
Educational synthetic data only — not real GPR measurements.

Tests multiple additive Gaussian-noise levels and reports:
- fitted x/depth estimates and absolute errors
- fit RMSE
- peak count
- simple false-detection proxy: peak count minus expected target-curve points

This is a controlled robustness experiment. Because noise can create extra
peaks and the fitting algorithm uses target-like initial guesses/time gates,
results are not a real-world detection probability.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import least_squares

# ---------------- Model configuration ----------------
RNG_SEED = 23
NOISE_LEVELS = [0.015, 0.03, 0.05, 0.08]
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

def make_clean_trace(x):
    trace = np.zeros(N_TIME, dtype=float)
    for target in TARGETS:
        dx = x - target["x"]
        path = np.sqrt(target["depth"]**2 + dx**2)
        arrival = SYSTEM_DELAY_NS * 1e-9 + 2 * path / VELOCITY
        sigma = target["sigma_ns"] * 1e-9
        trace += target["amplitude"] * np.exp(-0.5 * ((TIME - arrival) / sigma)**2)
    trace += 0.025 * np.exp(-TIME / 3e-9) * np.sin(2*np.pi*180e6*TIME)
    return trace

def travel_time_ns(x, x0, depth):
    return SYSTEM_DELAY_NS + (2.0 / VELOCITY) * np.sqrt(depth**2 + (x - x0)**2) * 1e9

def detect(bscan):
    detections = []
    for ix, x in enumerate(X_POSITIONS):
        peaks, _ = find_peaks(
            bscan[ix], height=0.06, prominence=0.04, distance=50
        )
        for p in peaks:
            if TIME_NS[p] > SYSTEM_DELAY_NS + 0.5:
                detections.append([x, TIME_NS[p], bscan[ix, p]])
    return np.asarray(detections, dtype=float).reshape(-1, 3)

def fit_two_targets(detections):
    fits = []
    unused = detections.copy()
    seeds = [(1.5, 0.15), (3.5, 0.30)]
    for number, (seed_x, seed_d) in enumerate(seeds, 1):
        if len(unused) < 5:
            fits.append({"target": number, "x": np.nan, "depth": np.nan,
                         "rmse": np.nan, "points": len(unused)})
            continue
        predicted = travel_time_ns(unused[:, 0], seed_x, seed_d)
        gate = np.abs(unused[:, 1] - predicted) < 3.5
        candidates = unused[gate]
        if len(candidates) < 5:
            order = np.argsort(np.abs(unused[:, 1] - predicted))
            candidates = unused[order[:min(30, len(unused))]]

        def residual(params):
            return travel_time_ns(candidates[:, 0], params[0], params[1]) - candidates[:, 1]

        result = least_squares(
            residual, x0=[seed_x, seed_d],
            bounds=([0.0, 0.02], [5.0, 1.0]),
            loss="soft_l1", f_scale=0.5, max_nfev=5000
        )
        x_est, depth_est = result.x
        rmse = float(np.sqrt(np.mean(residual(result.x)**2)))
        fits.append({"target": number, "x": float(x_est), "depth": float(depth_est),
                     "rmse": rmse, "points": len(candidates)})
        keep = np.abs(unused[:, 1] - travel_time_ns(unused[:, 0], x_est, depth_est)) > 1.5
        if np.count_nonzero(keep) >= 5:
            unused = unused[keep]
    return fits

def fit_is_valid(fit):
    return np.isfinite(fit["x"]) and np.isfinite(fit["depth"])

# Generate a clean reference B-scan.
clean = np.vstack([make_clean_trace(x) for x in X_POSITIONS])
rng = np.random.default_rng(RNG_SEED)

rows = []
plot_curves = []
print("===== STAGE 23: MULTI-NOISE ROBUSTNESS SWEEP =====")
print(f"Seed={RNG_SEED}; sample rate={FS/1e9:.1f} GHz; positions={len(X_POSITIONS)}")
print("Noise std | peaks | target | x estimate | x error | depth estimate | depth error | RMSE(ns)")

for noise_std in NOISE_LEVELS:
    noisy = clean + rng.normal(0.0, noise_std, size=clean.shape)
    detections = detect(noisy)
    fits = fit_two_targets(detections)

    # Detection-rate proxy: target fit counts as detected if fit is valid and near
    # the reference location/depth. This is not a statistical probability.
    for target, fit in zip(TARGETS, fits):
        valid = fit_is_valid(fit)
        if valid:
            x_error = fit["x"] - target["x"]
            depth_error = fit["depth"] - target["depth"]
            detected = abs(x_error) <= 0.10 and abs(depth_error) <= 0.05
        else:
            x_error = np.nan
            depth_error = np.nan
            detected = False
        rows.append([
            noise_std, len(detections), target["x"], fit["x"], x_error,
            target["depth"], fit["depth"], depth_error, fit["rmse"],
            int(detected), fit["points"]
        ])
        print(
            f"{noise_std:8.3f} | {len(detections):5d} | {fit['target']:6d} | "
            f"{fit['x']:10.3f} | {x_error:+7.3f} | {fit['depth']:14.3f} | "
            f"{depth_error:+11.3f} | {fit['rmse']:8.3f}"
        )
    plot_curves.append((noise_std, noisy, fits, len(detections)))

# Save CSV.
header = (
    "noise_std,peak_count,target_id,x_est_m,x_error_m,depth_true_m,"
    "depth_est_m,depth_error_m,fit_rmse_ns,within_tolerance,fit_points"
)
np.savetxt("stage23_noise_sweep.csv", np.asarray(rows, dtype=float),
           delimiter=",", header=header, comments="")

# Plot fitted errors vs noise.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
for target_id in [1, 2]:
    target_rows = [r for r in rows if int(r[2] == TARGETS[target_id-1]["x"]) == 1]
    # rows columns: noise, peak count, target_x, x_est, x_error, depth_true, depth_est, depth_error...
    axes[0].plot([r[0] for r in target_rows], [abs(r[4]) for r in target_rows],
                 marker="o", label=f"Target {target_id}")
    axes[1].plot([r[0] for r in target_rows], [abs(r[7]) for r in target_rows],
                 marker="o", label=f"Target {target_id}")
axes[0].set_title("Absolute horizontal-position error")
axes[0].set_xlabel("Noise standard deviation")
axes[0].set_ylabel("Absolute error (m)")
axes[1].set_title("Absolute depth error")
axes[1].set_xlabel("Noise standard deviation")
axes[1].set_ylabel("Absolute error (m)")
for ax in axes:
    ax.grid(True)
    ax.legend()
fig.savefig("stage23_error_vs_noise.png", dpi=160)
plt.show()

# Display the highest noise B-scan and fitted curves.
noise_std, noisy, fits, peak_count = plot_curves[-1]
plt.figure(figsize=(10.5, 5.8))
plt.imshow(noisy.T, aspect="auto", origin="lower",
           extent=[X_POSITIONS[0], X_POSITIONS[-1], TIME_NS[0], TIME_NS[-1]],
           cmap="gray", vmin=-0.10, vmax=0.45)
x_curve = np.linspace(0, 5, 501)
colors = ["cyan", "yellow"]
for i, fit in enumerate(fits):
    if fit_is_valid(fit):
        plt.plot(x_curve, travel_time_ns(x_curve, fit["x"], fit["depth"]),
                 color=colors[i], linewidth=1.5,
                 label=f"Target {fit['target']}: x={fit['x']:.2f} m, d={fit['depth']:.2f} m")
plt.ylim(0, 75)
plt.title(f"Stage 23 — Highest Noise Level (std={noise_std:.3f})")
plt.xlabel("Antenna horizontal position (m)")
plt.ylabel("Arrival time (ns)")
plt.colorbar(label="Reflection amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("stage23_high_noise_fit.png", dpi=160)
plt.show()

print("\nFiles saved:")
print("  stage23_noise_sweep.csv")
print("  stage23_error_vs_noise.png")
print("  stage23_high_noise_fit.png")
print("\nInterpretation: assess errors and failed fits at each noise level.")
print("These are single-run results for a fixed random seed, not generalized probabilities.")
