"""
Stage 18: Synthetic FSK-GPR A-scan and B-scan demonstration
This is a synthetic educational model, not real GPR measurements.
Targets:
  Target 1: x=1.5 m, depth=0.15 m, reflection amplitude=0.4
  Target 2: x=3.5 m, depth=0.30 m, reflection amplitude=0.3
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------- USER-ADJUSTABLE SETTINGS ----------------
C0 = 3.0e8                  # speed of light in vacuum (m/s)
RELATIVE_PERMITTIVITY = 9.0 # illustrative soil dielectric constant
VELOCITY = C0 / np.sqrt(RELATIVE_PERMITTIVITY)  # approx. 1e8 m/s
FS = 10.0e9                 # 10 GHz simulation sample rate
T_WINDOW = 12e-9             # 12 ns time window
N_TIME = int(FS * T_WINDOW)  # 12 samples at 1 GHz; increased below for useful waveform
N_TIME = 1200
DT = 1.0 / FS
TIME = np.arange(N_TIME) * DT

# A short pulse stands in for a transmitted/processed radar pulse.
# This is a synthetic pulse model; it does not simulate a complete RF antenna or soil.
PULSE_CENTER = 1.2e-9
PULSE_SIGMA = 0.22e-9
def pulse(t):
    return np.exp(-0.5 * ((t - PULSE_CENTER) / PULSE_SIGMA) ** 2)

# Scan positions along the ground
X_POSITIONS = np.linspace(0.0, 5.0, 101)
TARGETS = [
    {"x": 1.5, "depth": 0.15, "amplitude": 0.40, "width": 0.28},
    {"x": 3.5, "depth": 0.30, "amplitude": 0.30, "width": 0.38},
]

def make_ascan(x):
    """Create one synthetic trace with hyperbolic travel-time responses."""
    trace = np.zeros(N_TIME, dtype=float)
    for target in TARGETS:
        lateral_offset = x - target["x"]
        # Two-way travel time to a point reflector.
        travel_time = 2.0 * np.sqrt(target["depth"]**2 + lateral_offset**2) / VELOCITY
        # A gaussian pulse centered at the simulated travel time.
        # Include a small fixed system delay so the trace is easy to view.
        center = 1.0e-9 + travel_time
        sigma = target["width"] * 1e-9
        trace += target["amplitude"] * np.exp(
            -0.5 * ((TIME - center) / sigma) ** 2
        )
    # Add a small, deterministic background wavelet/ringing term.
    trace += 0.025 * np.exp(-TIME / 3e-9) * np.sin(2 * np.pi * 180e6 * TIME)
    return trace

# Build B-scan: one A-scan for each antenna position
BSCAN = np.vstack([make_ascan(x) for x in X_POSITIONS])
# A-scan nearest to target 1 (x=1.5 m)
AScan = BSCAN[np.argmin(np.abs(X_POSITIONS - 1.5))]
time_ns = TIME * 1e9

# Find local maxima in the chosen trace (simple illustrative threshold)
peak_idx = np.where(
    (AScan[1:-1] > AScan[:-2]) &
    (AScan[1:-1] >= AScan[2:]) &
    (AScan[1:-1] > 0.08)
)[0] + 1

print("===== STAGE 18: SYNTHETIC GPR MODEL =====")
print(f"Propagation velocity: {VELOCITY:.3e} m/s")
print(f"Sampling rate: {FS/1e9:.2f} GHz")
print(f"Time resolution: {DT*1e9:.2f} ns")
print("Targets (simulation inputs):")
for i, target in enumerate(TARGETS, 1):
    print(f"  Target {i}: x={target['x']:.2f} m, depth={target['depth']:.2f} m, amplitude={target['amplitude']:.2f}")
print(f"A-scan selected at x={X_POSITIONS[np.argmin(np.abs(X_POSITIONS - 1.5))]:.2f} m")
print(f"Detected peaks above threshold: {len(peak_idx)}")
for p in peak_idx:
    print(f"  Peak time={time_ns[p]:.2f} ns, amplitude={AScan[p]:.3f}")
print("NOTE: This is a synthetic educational model, not measured radar data.")

# A-scan plot
plt.figure(figsize=(10, 5))
plt.plot(time_ns, AScan, linewidth=1.5)
for target in TARGETS:
    t_ns = (1.0e-9 + 2.0 * target["depth"] / VELOCITY) * 1e9
    plt.axvline(t_ns, linestyle="--", alpha=0.7,
                label=f"Target depth {target['depth']:.2f} m (nominal)")
plt.title("Stage 18 — Synthetic GPR A-scan at x = 1.5 m")
plt.xlabel("Time (ns)")
plt.ylabel("Normalized reflection amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("stage18_ascan.png", dpi=160)
plt.show()

# B-scan plot
plt.figure(figsize=(11, 6))
plt.imshow(
    BSCAN.T,
    aspect="auto",
    origin="lower",
    extent=[X_POSITIONS[0], X_POSITIONS[-1], time_ns[0], time_ns[-1]],
    cmap="gray"
)
plt.title("Stage 18 — Synthetic GPR B-scan")
plt.xlabel("Antenna horizontal position (m)")
plt.ylabel("Time (ns)")
plt.colorbar(label="Reflection amplitude")
plt.tight_layout()
plt.savefig("stage18_bscan.png", dpi=160)
plt.show()

np.savetxt("stage18_bscan.csv", BSCAN, delimiter=",")
print("Saved: stage18_ascan.png, stage18_bscan.png, stage18_bscan.csv")
