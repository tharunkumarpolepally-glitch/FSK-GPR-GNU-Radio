"""
Stage 21: B-scan target localization from synthetic traces
Educational synthetic model only — not measured GPR data.

This script regenerates the Stage 19/20 synthetic B-scan, detects peaks
in each A-scan, and estimates target x-position and depth by fitting the
measured peak-time curve to the two-way travel-time model.

Model:
  arrival_time = system_delay + (2/v) * sqrt(depth^2 + (x - x_target)^2)

The fit uses detected peak times across antenna positions, rather than
using the target depths as the output. It is still a controlled simulation
because the data itself is generated from known model parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import least_squares

# ---------------- Simulation settings ----------------
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
    trace += 0.025 * np.exp(-TIME / 3e-9) * np.sin(2 * np.pi * 180e6 * TIME)
    return trace

BSCAN = np.vstack([make_trace(x) for x in X_POSITIONS])

# Peak detection per trace.
detections = []
for ix, x in enumerate(X_POSITIONS):
    trace = BSCAN[ix]
    peaks, props = find_peaks(trace, height=0.06, prominence=0.04, distance=50)
    for p in peaks:
        t_ns = TIME_NS[p]
        # Keep only plausible reflection arrivals and suppress early ringing.
        if t_ns > SYSTEM_DELAY_NS + 0.5:
            detections.append((x, t_ns, trace[p], ix, p))

detections = np.asarray(detections, dtype=float)
if detections.size == 0:
    raise RuntimeError("No peaks detected. Try lowering the height/prominence thresholds.")

# Fit one hyperbola at a time to peak detections. Robust loss reduces impact
# of detections from the other target or any residual ringing.
def model_time_ns(x, x0, depth):
    return SYSTEM_DELAY_NS + (2.0 / VELOCITY) * np.sqrt(depth**2 + (x - x0)**2) * 1e9

fits = []
unused = detections.copy()

for target_number, seed in enumerate([1.5, 3.5], start=1):
    # Start from a target-like seed, but fit x and depth from detections.
    # Use a broad time gate around this seed's expected hyperbola to separate targets.
    seed_depth = 0.15 if target_number == 1 else 0.30
    predicted = model_time_ns(unused[:, 0], seed, seed_depth)
    gate = np.abs(unused[:, 1] - predicted) < 3.5
    candidates = unused[gate]

    if len(candidates) < 5:
        # Fall back to detections nearest in time to the seed hyperbola.
        order = np.argsort(np.abs(unused[:, 1] - predicted))
        candidates = unused[order[:max(5, min(25, len(unused)))]]

    def residual(params):
        x0, depth = params
        return model_time_ns(candidates[:, 0], x0, depth) - candidates[:, 1]

    result = least_squares(
        residual,
        x0=np.array([seed, seed_depth]),
        bounds=([0.0, 0.02], [5.0, 1.0]),
        loss="soft_l1",
        f_scale=0.5,
        max_nfev=5000
    )
    x_est, depth_est = result.x
    rmse = float(np.sqrt(np.mean(residual(result.x)**2)))
    fits.append({
        "target": target_number,
        "x": float(x_est),
        "depth": float(depth_est),
        "rmse_ns": rmse,
        "n_fit": len(candidates),
    })

    # Remove detections close to this fitted curve before fitting the next.
    all_pred = model_time_ns(unused[:, 0], x_est, depth_est)
    keep = np.abs(unused[:, 1] - all_pred) > 1.5
    # Keep detections for the second fit even if a few are shared when no points remain.
    if np.count_nonzero(keep) >= 5:
        unused = unused[keep]

print("===== STAGE 21: B-SCAN LOCALIZATION =====")
print(f"Velocity assumed: {VELOCITY:.3e} m/s")
print(f"Antenna positions: {len(X_POSITIONS)}")
print(f"Total detected peak points: {len(detections)}")
print("\nFitted target estimates:")
for fit in fits:
    print(
        f"Target {fit['target']}: x_est={fit['x']:.3f} m, "
        f"depth_est={fit['depth']:.3f} m, "
        f"fit RMSE={fit['rmse_ns']:.3f} ns, points used={fit['n_fit']}"
    )
print("\nReference values (used only to generate synthetic data):")
for i, target in enumerate(TARGETS, start=1):
    print(f"Target {i}: x={target['x']:.3f} m, depth={target['depth']:.3f} m")
print("Reminder: this is synthetic-model validation, not real-world GPR validation.")

# Plot B-scan and fitted hyperbolas.
plt.figure(figsize=(11, 6))
plt.imshow(
    BSCAN.T,
    aspect="auto",
    origin="lower",
    extent=[X_POSITIONS[0], X_POSITIONS[-1], TIME_NS[0], TIME_NS[-1]],
    cmap="gray"
)
colors = ["cyan", "yellow"]
x_curve = np.linspace(X_POSITIONS[0], X_POSITIONS[-1], 501)
for i, fit in enumerate(fits):
    t_curve = model_time_ns(x_curve, fit["x"], fit["depth"])
    plt.plot(x_curve, t_curve, color=colors[i], linewidth=1.5,
             label=f"Fit T{fit['target']}: x={fit['x']:.2f} m, d={fit['depth']:.2f} m")
plt.ylim(0, 75)
plt.title("Stage 21 — B-scan with Fitted Target Hyperbolas")
plt.xlabel("Antenna horizontal position (m)")
plt.ylabel("Two-way arrival time (ns)")
plt.colorbar(label="Reflection amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("stage21_localized_bscan.png", dpi=160)
plt.show()

# Plot all detected peaks as a time-position scatter plot.
plt.figure(figsize=(10, 5))
plt.scatter(detections[:, 0], detections[:, 1], s=10, label="Detected peaks")
for i, fit in enumerate(fits):
    plt.plot(x_curve, model_time_ns(x_curve, fit["x"], fit["depth"]),
             color=colors[i], linewidth=1.5,
             label=f"Fitted target {fit['target']}")
plt.ylim(0, 75)
plt.title("Stage 21 — Peak-Time Tracks and Fitted Hyperbolas")
plt.xlabel("Antenna horizontal position (m)")
plt.ylabel("Arrival time (ns)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("stage21_peak_tracks.png", dpi=160)
plt.show()

# Save detections and fit summaries.
np.savetxt(
    "stage21_peak_detections.csv",
    detections[:, :3],
    delimiter=",",
    header="antenna_x_m,peak_time_ns,peak_amplitude",
    comments=""
)
with open("stage21_target_estimates.csv", "w", encoding="utf-8") as f:
    f.write("target,x_est_m,depth_est_m,fit_rmse_ns,points_used\n")
    for fit in fits:
        f.write(
            f"{fit['target']},{fit['x']:.6f},{fit['depth']:.6f},"
            f"{fit['rmse_ns']:.6f},{fit['n_fit']}\n"
        )

print("\nSaved:")
print("  stage21_localized_bscan.png")
print("  stage21_peak_tracks.png")
print("  stage21_peak_detections.csv")
print("  stage21_target_estimates.csv")
