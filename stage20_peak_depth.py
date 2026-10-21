"""
Stage 20 — Complete corrected synthetic GPR peak detection
This is a synthetic educational model, not measured radar data.

At x=2.5 m, both targets should contribute reflections:
  Target 1: x=1.5 m, depth=0.15 m, amplitude=0.40
  Target 2: x=3.5 m, depth=0.30 m, amplitude=0.30

The script explicitly reports the modeled arrival times and detects peaks
from the generated A-scan. It does not use the known depths to create the
depth estimates; the known horizontal target positions are used for the
lateral-offset correction, so this remains a controlled simulation test.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ---------------- Simulation settings ----------------
C0 = 3.0e8
EPS_R = 9.0
VELOCITY = C0 / np.sqrt(EPS_R)  # 1.0e8 m/s
FS = 10e9                       # 10 GHz
DT = 1.0 / FS                   # 0.1 ns/sample
T_WINDOW = 60e-9                # 60 ns
N_TIME = int(round(T_WINDOW * FS))
TIME = np.arange(N_TIME) * DT
TIME_NS = TIME * 1e9
SYSTEM_DELAY_NS = 1.0

X_POSITIONS = np.linspace(0.0, 5.0, 101)
TARGETS = [
    {"x": 1.5, "depth": 0.15, "amplitude": 0.40, "sigma_ns": 0.28},
    {"x": 3.5, "depth": 0.30, "amplitude": 0.30, "sigma_ns": 0.38},
]
SELECT_X = 1.5

def make_trace(x):
    """Generate one synthetic A-scan using two-way travel-time responses."""
    trace = np.zeros(N_TIME, dtype=float)

    for target in TARGETS:
        dx = x - target["x"]
        one_way_path = np.sqrt(target["depth"]**2 + dx**2)
        travel_time_s = 2.0 * one_way_path / VELOCITY
        arrival_s = SYSTEM_DELAY_NS * 1e-9 + travel_time_s
        sigma_s = target["sigma_ns"] * 1e-9

        trace += target["amplitude"] * np.exp(
            -0.5 * ((TIME - arrival_s) / sigma_s) ** 2
        )

    # Small damped background/ringing component.
    trace += 0.025 * np.exp(-TIME / 3e-9) * np.sin(2 * np.pi * 180e6 * TIME)
    return trace

# Build scan and choose nearest antenna position.
BSCAN = np.vstack([make_trace(x) for x in X_POSITIONS])
scan_index = int(np.argmin(np.abs(X_POSITIONS - SELECT_X)))
x_selected = float(X_POSITIONS[scan_index])
AScan = BSCAN[scan_index]

# Print modeled arrivals independently of peak detection.
print("===== STAGE 20: COMPLETE PEAK DETECTION =====")
print(f"Selected antenna position: x={x_selected:.2f} m")
print(f"Propagation velocity: {VELOCITY:.3e} m/s")
print(f"Sampling rate: {FS/1e9:.1f} GHz; time resolution: {DT*1e9:.2f} ns")
print(f"Signal maximum: {np.max(AScan):.4f}")
print(f"Signal minimum: {np.min(AScan):.4f}")
print("Expected target arrivals from model geometry:")
for i, target in enumerate(TARGETS, 1):
    dx = x_selected - target["x"]
    path = np.sqrt(target["depth"]**2 + dx**2)
    arrival_ns = SYSTEM_DELAY_NS + 2.0 * path / VELOCITY * 1e9
    print(f"  Target {i}: x={target['x']:.2f} m, depth={target['depth']:.2f} m, arrival={arrival_ns:.2f} ns")

# Detect peaks. A distance of 5 ns separates the expected two target arrivals.
peaks, properties = find_peaks(
    AScan,
    height=0.03,
    prominence=0.02,
    distance=int(5e-9 * FS)
)

print(f"\nDetection settings: height=0.03, prominence=0.02, minimum spacing=5 ns")
print(f"Detected peaks: {len(peaks)}")

results = []
for i, p in enumerate(peaks, 1):
    peak_time_ns = float(TIME_NS[p])
    amplitude = float(AScan[p])
    corrected_round_trip_s = (peak_time_ns - SYSTEM_DELAY_NS) * 1e-9

    # Candidate depth calculations use the known horizontal target positions
    # to correct for lateral offset. This is not blind x localization.
    candidates = []
    if corrected_round_trip_s > 0:
        half_path = VELOCITY * corrected_round_trip_s / 2.0
        for target in TARGETS:
            dx = abs(x_selected - target["x"])
            depth_sq = half_path**2 - dx**2
            if depth_sq >= 0:
                candidates.append(
                    (np.sqrt(depth_sq), target["x"], dx, target["depth"])
                )

    if candidates:
        # Select the feasible target with the smallest mismatch to model depth
        # only for diagnostic comparison; do not present as blind detection.
        depth_est, candidate_x, dx, model_depth = min(
            candidates, key=lambda c: abs(c[0] - c[3])
        )
        print(
            f"  Peak {i}: time={peak_time_ns:.2f} ns, amplitude={amplitude:.3f}, "
            f"candidate x={candidate_x:.2f} m, depth estimate={depth_est:.3f} m "
            f"(model depth={model_depth:.3f} m)"
        )
        results.append([peak_time_ns, amplitude, candidate_x, depth_est, model_depth])
    else:
        print(
            f"  Peak {i}: time={peak_time_ns:.2f} ns, amplitude={amplitude:.3f}; "
            "no feasible target-depth candidate for the assumed geometry"
        )
        results.append([peak_time_ns, amplitude, np.nan, np.nan, np.nan])

if len(peaks) < len(TARGETS):
    print("\nWARNING: Fewer peaks were detected than modeled targets.")
    print("Inspect the printed expected arrival times and plot before changing thresholds.")

# Plot A-scan over the complete 60 ns interval.
plt.figure(figsize=(11, 5.5))
plt.plot(TIME_NS, AScan, linewidth=1.4, label="Synthetic A-scan")
if len(peaks):
    plt.plot(TIME_NS[peaks], AScan[peaks], "x", markersize=9, label="Detected peaks")
for i, target in enumerate(TARGETS, 1):
    dx = x_selected - target["x"]
    path = np.sqrt(target["depth"]**2 + dx**2)
    arrival_ns = SYSTEM_DELAY_NS + 2.0 * path / VELOCITY * 1e9
    plt.axvline(
        arrival_ns,
        linestyle="--",
        linewidth=1,
        label=f"Target {i} modeled arrival: {arrival_ns:.2f} ns"
    )
plt.axvline(SYSTEM_DELAY_NS, color="gray", linestyle=":", label="System delay: 1 ns")
plt.xlim(0, 60)
plt.title(f"Stage 20 — Automatic Peak Detection at x={x_selected:.1f} m")
plt.xlabel("Time (ns)")
plt.ylabel("Reflection amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("stage20_peak_detection.png", dpi=160)
plt.show()

# Save detected peaks and model-derived estimates.
if results:
    np.savetxt(
        "stage20_depth_estimates.csv",
        np.asarray(results, dtype=float),
        delimiter=",",
        header="peak_time_ns,amplitude,candidate_target_x_m,estimated_depth_m,model_depth_m",
        comments=""
    )
    print("\nSaved: stage20_depth_estimates.csv")
else:
    print("\nNo peaks were detected; inspect signal and settings.")

# Save the selected A-scan and full B-scan matrix for later stages.
np.savetxt(
    "stage20_ascan.csv",
    np.column_stack((TIME_NS, AScan)),
    delimiter=",",
    header="time_ns,reflection_amplitude",
    comments=""
)
np.savetxt("stage20_bscan.csv", BSCAN, delimiter=",")
print("Saved: stage20_peak_detection.png, stage20_ascan.csv, stage20_bscan.csv")
print("Reminder: this is a controlled synthetic model, not measured GPR data.")
