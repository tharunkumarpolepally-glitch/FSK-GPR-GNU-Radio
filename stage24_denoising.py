"""
Stage 24: B-scan denoising and coherent-peak detection
Synthetic educational GPR simulation only — not measured radar data.

Applies 2D Gaussian smoothing to a noisy B-scan, compares peak counts before
and after filtering, then fits target travel-time curves to filtered detections.
The fitting seeds/time gates remain model-informed; this is not blind detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.optimize import least_squares

# Model parameters consistent with Stages 21–23
RNG_SEED = 24
NOISE_STD = 0.08
C0 = 3e8
EPS_R = 9.0
VELOCITY = C0 / np.sqrt(EPS_R)
FS = 10e9
DT = 1 / FS
T_WINDOW = 75e-9
N_TIME = int(round(T_WINDOW * FS))
TIME = np.arange(N_TIME) * DT
TIME_NS = TIME * 1e9
SYSTEM_DELAY_NS = 1.0
X_POSITIONS = np.linspace(0, 5, 101)
TARGETS = [
    {"x": 1.5, "depth": 0.15, "amplitude": 0.40, "sigma_ns": 0.28},
    {"x": 3.5, "depth": 0.30, "amplitude": 0.30, "sigma_ns": 0.38},
]

def make_trace(x):
    trace = np.zeros(N_TIME)
    for target in TARGETS:
        dx = x - target["x"]
        path = np.sqrt(target["depth"]**2 + dx**2)
        arrival_s = SYSTEM_DELAY_NS * 1e-9 + 2 * path / VELOCITY
        sigma_s = target["sigma_ns"] * 1e-9
        trace += target["amplitude"] * np.exp(
            -0.5 * ((TIME - arrival_s) / sigma_s)**2
        )
    trace += 0.025 * np.exp(-TIME / 3e-9) * np.sin(2*np.pi*180e6*TIME)
    return trace

def travel_time_ns(x, x0, depth):
    return SYSTEM_DELAY_NS + 2 * np.sqrt(depth**2 + (x-x0)**2) / VELOCITY * 1e9

def detect_peaks(bscan, height=0.06, prominence=0.04):
    points = []
    for ix, x in enumerate(X_POSITIONS):
        peaks, _ = find_peaks(
            bscan[ix], height=height, prominence=prominence, distance=50
        )
        for p in peaks:
            if TIME_NS[p] > SYSTEM_DELAY_NS + 0.5:
                points.append([x, TIME_NS[p], bscan[ix, p]])
    return np.asarray(points, dtype=float).reshape(-1, 3)

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
            idx = np.argsort(np.abs(remaining[:, 1] - expected))[:min(30, len(remaining))]
            candidates = remaining[idx]
        def residual(p):
            return travel_time_ns(candidates[:, 0], p[0], p[1]) - candidates[:, 1]
        result = least_squares(
            residual, [seed_x, seed_depth], bounds=([0, 0.02], [5, 1.0]),
            loss="soft_l1", f_scale=0.5, max_nfev=5000
        )
        x_est, d_est = result.x
        rmse = float(np.sqrt(np.mean(residual(result.x)**2)))
        fits.append((target_id, float(x_est), float(d_est), rmse, len(candidates)))
        keep = np.abs(remaining[:, 1] - travel_time_ns(remaining[:, 0], x_est, d_est)) > 1.5
        if np.count_nonzero(keep) >= 5:
            remaining = remaining[keep]
    return fits

# Generate noisy data with reproducible random noise.
clean = np.vstack([make_trace(x) for x in X_POSITIONS])
rng = np.random.default_rng(RNG_SEED)
noisy = clean + rng.normal(0, NOISE_STD, size=clean.shape)

# Smooth across adjacent antenna positions and adjacent time samples.
# sigma=(rows, columns): modest spatial smoothing, stronger time smoothing.
denoised = gaussian_filter(noisy, sigma=(1.0, 1.5), mode="nearest")

raw_detections = detect_peaks(noisy)
filtered_detections = detect_peaks(denoised)
fits = fit_targets(filtered_detections)

print("===== STAGE 24: DENOISING AND PEAK DETECTION =====")
print(f"Noise standard deviation: {NOISE_STD:.3f}")
print(f"Gaussian filter sigma: antenna={1.0:.1f} traces, time={1.5:.1f} samples")
print(f"Raw detections: {len(raw_detections)}")
print(f"Filtered detections: {len(filtered_detections)}")
print("\nFits from filtered detections:")
for target_id, x_est, d_est, rmse, npoints in fits:
    if np.isfinite(x_est):
        target = TARGETS[target_id-1]
        print(
            f"Target {target_id}: x={x_est:.3f} m "
            f"(reference {target['x']:.3f}), depth={d_est:.3f} m "
            f"(reference {target['depth']:.3f}), RMSE={rmse:.3f} ns, "
            f"points={npoints}"
        )
    else:
        print(f"Target {target_id}: fit failed")
print("\nNote: smoothing can blur nearby reflections; compare the plots and errors.")
print("This is synthetic model validation, not real measured GPR data.")

# Display clean, noisy, and filtered B-scans.
fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), constrained_layout=True)
images = []
for ax, data, title in [
    (axes[0], clean, "Clean B-scan"),
    (axes[1], noisy, f"Noisy B-scan (std={NOISE_STD})"),
    (axes[2], denoised, "Denoised B-scan"),
]:
    im = ax.imshow(
        data.T, aspect="auto", origin="lower",
        extent=[X_POSITIONS[0], X_POSITIONS[-1], TIME_NS[0], TIME_NS[-1]],
        cmap="gray", vmin=-0.10, vmax=0.45
    )
    images.append(im)
    ax.set_ylim(0, 75)
    ax.set_xlabel("Antenna position (m)")
    ax.set_title(title)
axes[0].set_ylabel("Arrival time (ns)")
fig.colorbar(images[-1], ax=axes, label="Reflection amplitude", shrink=0.82)
fig.savefig("stage24_denoising_comparison.png", dpi=160)
plt.show()

# Plot fitted curves on the denoised scan.
plt.figure(figsize=(10.5, 5.8))
plt.imshow(
    denoised.T, aspect="auto", origin="lower",
    extent=[X_POSITIONS[0], X_POSITIONS[-1], TIME_NS[0], TIME_NS[-1]],
    cmap="gray", vmin=-0.10, vmax=0.45
)
x_curve = np.linspace(0, 5, 501)
line_colors = ["cyan", "yellow"]
for i, fit in enumerate(fits):
    target_id, x_est, d_est, rmse, _ = fit
    if np.isfinite(x_est):
        plt.plot(x_curve, travel_time_ns(x_curve, x_est, d_est),
                 color=line_colors[i], linewidth=1.5,
                 label=f"T{target_id}: x={x_est:.2f} m, depth={d_est:.2f} m")
plt.ylim(0, 75)
plt.xlabel("Antenna position (m)")
plt.ylabel("Arrival time (ns)")
plt.title("Stage 24 — Fitted Curves on Denoised B-scan")
plt.colorbar(label="Reflection amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("stage24_denoised_fit.png", dpi=160)
plt.show()

np.savetxt(
    "stage24_filtered_detections.csv", filtered_detections, delimiter=",",
    header="antenna_x_m,peak_time_ns,peak_amplitude", comments=""
)
with open("stage24_target_estimates.csv", "w", encoding="utf-8") as f:
    f.write("target,x_est_m,depth_est_m,fit_rmse_ns,points_used\n")
    for fit in fits:
        f.write(",".join(map(str, fit)) + "\n")

print("\nSaved:")
print("  stage24_denoising_comparison.png")
print("  stage24_denoised_fit.png")
print("  stage24_filtered_detections.csv")
print("  stage24_target_estimates.csv")
