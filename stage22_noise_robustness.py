"""
Stage 22: Robustness test for synthetic GPR localization
Adds controlled Gaussian noise to the synthetic B-scan and compares estimated
target locations/depths against the clean model.

Educational synthetic model only — not measured radar data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import least_squares

# ---------------- Settings ----------------
RNG_SEED = 22
NOISE_STD = 0.015          # additive Gaussian noise standard deviation
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
    return SYSTEM_DELAY_NS + (2.0 / VELOCITY) * np.sqrt(depth**2 + (x-x0)**2) * 1e9

def detect_peaks(bscan, height, prominence):
    detections = []
    for ix, x in enumerate(X_POSITIONS):
        trace = bscan[ix]
        peaks, _ = find_peaks(
            trace, height=height, prominence=prominence, distance=50
        )
        for p in peaks:
            if TIME_NS[p] > SYSTEM_DELAY_NS + 0.5:
                detections.append([x, TIME_NS[p], trace[p]])
    return np.asarray(detections, dtype=float)

def fit_targets(detections):
    fits = []
    unused = detections.copy()
    seeds = [(1.5, 0.15), (3.5, 0.30)]
    for number, (seed_x, seed_d) in enumerate(seeds, start=1):
        if len(unused) == 0:
            fits.append((number, np.nan, np.nan, np.nan, 0))
            continue
        pred = travel_time_ns(unused[:, 0], seed_x, seed_d)
        gate = np.abs(unused[:, 1] - pred) < 3.5
        candidates = unused[gate]
        if len(candidates) < 5:
            order = np.argsort(np.abs(unused[:, 1] - pred))
            candidates = unused[order[:min(30, len(unused))]]
        def residual(params):
            return travel_time_ns(candidates[:, 0], params[0], params[1]) - candidates[:, 1]
        result = least_squares(
            residual, x0=[seed_x, seed_d],
            bounds=([0.0, 0.02], [5.0, 1.0]),
            loss="soft_l1", f_scale=0.5, max_nfev=5000
        )
        x_est, d_est = result.x
        rmse = float(np.sqrt(np.mean(residual(result.x)**2)))
        fits.append((number, float(x_est), float(d_est), rmse, len(candidates)))
        pred_all = travel_time_ns(unused[:, 0], x_est, d_est)
        keep = np.abs(unused[:, 1] - pred_all) > 1.5
        if np.count_nonzero(keep) >= 5:
            unused = unused[keep]
    return fits

def summarize(label, detections, fits):
    print(f"\n--- {label} ---")
    print(f"Peak detections: {len(detections)}")
    for number, x_est, d_est, rmse, nfit in fits:
        if np.isfinite(x_est):
            target = TARGETS[number-1]
            print(
                f"Target {number}: x={x_est:.3f} m "
                f"(error {x_est-target['x']:+.3f} m), "
                f"depth={d_est:.3f} m (error {d_est-target['depth']:+.3f} m), "
                f"fit RMSE={rmse:.3f} ns, points={nfit}"
            )
        else:
            print(f"Target {number}: fit failed (not enough detections)")

# Build clean B-scan and noisy version.
clean = np.vstack([make_clean_trace(x) for x in X_POSITIONS])
rng = np.random.default_rng(RNG_SEED)
noise = rng.normal(0.0, NOISE_STD, size=clean.shape)
noisy = clean + noise

# Thresholds are chosen for this synthetic model; tune for other noise levels.
clean_detections = detect_peaks(clean, height=0.06, prominence=0.04)
noisy_detections = detect_peaks(noisy, height=0.06, prominence=0.04)

clean_fits = fit_targets(clean_detections)
noisy_fits = fit_targets(noisy_detections)

print("===== STAGE 22: NOISE ROBUSTNESS TEST =====")
print(f"Noise standard deviation: {NOISE_STD:.3f}")
print(f"Assumed propagation velocity: {VELOCITY:.3e} m/s")
print(f"Random seed: {RNG_SEED}")
summarize("CLEAN DATA", clean_detections, clean_fits)
summarize("NOISY DATA", noisy_detections, noisy_fits)
print("\nReference targets:")
for i, t in enumerate(TARGETS, 1):
    print(f"Target {i}: x={t['x']:.3f} m, depth={t['depth']:.3f} m")
print("NOTE: This tests robustness only against the selected synthetic noise model.")

# Plot side-by-side B-scans.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
for ax, data, title in [
    (axes[0], clean, "Clean synthetic B-scan"),
    (axes[1], noisy, f"Noisy synthetic B-scan (std={NOISE_STD})"),
]:
    image = ax.imshow(
        data.T, aspect="auto", origin="lower",
        extent=[X_POSITIONS[0], X_POSITIONS[-1], TIME_NS[0], TIME_NS[-1]],
        cmap="gray", vmin=-0.02, vmax=0.45
    )
    ax.set_ylim(0, 75)
    ax.set_title(title)
    ax.set_xlabel("Antenna horizontal position (m)")
    ax.set_ylabel("Arrival time (ns)")
fig.colorbar(image, ax=axes, label="Reflection amplitude", shrink=0.85)
fig.savefig("stage22_clean_vs_noisy.png", dpi=160)
plt.show()

# Plot fitted curves on the noisy B-scan.
plt.figure(figsize=(10.5, 6))
plt.imshow(
    noisy.T, aspect="auto", origin="lower",
    extent=[X_POSITIONS[0], X_POSITIONS[-1], TIME_NS[0], TIME_NS[-1]],
    cmap="gray", vmin=-0.02, vmax=0.45
)
x_curve = np.linspace(0, 5, 501)
colors = ["cyan", "yellow"]
for i, fit in enumerate(noisy_fits):
    number, x_est, d_est, rmse, _ = fit
    if np.isfinite(x_est):
        plt.plot(
            x_curve, travel_time_ns(x_curve, x_est, d_est),
            color=colors[i], linewidth=1.5,
            label=f"Target {number}: x={x_est:.2f} m, d={d_est:.2f} m"
        )
plt.ylim(0, 75)
plt.title("Stage 22 — Noisy B-scan with Fitted Targets")
plt.xlabel("Antenna horizontal position (m)")
plt.ylabel("Arrival time (ns)")
plt.colorbar(label="Reflection amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("stage22_noisy_fit.png", dpi=160)
plt.show()

# Save compact summary CSV.
with open("stage22_summary.csv", "w", encoding="utf-8") as f:
    f.write("dataset,target,x_est_m,depth_est_m,fit_rmse_ns,points_used\n")
    for label, fits in [("clean", clean_fits), ("noisy", noisy_fits)]:
        for number, x_est, d_est, rmse, nfit in fits:
            f.write(f"{label},{number},{x_est},{d_est},{rmse},{nfit}\n")

np.savetxt(
    "stage22_noisy_peak_detections.csv",
    noisy_detections, delimiter=",",
    header="antenna_x_m,peak_time_ns,peak_amplitude", comments=""
)
print("\nSaved:")
print("  stage22_clean_vs_noisy.png")
print("  stage22_noisy_fit.png")
print("  stage22_summary.csv")
print("  stage22_noisy_peak_detections.csv")
