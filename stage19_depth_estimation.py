"""
Stage 19: Corrected synthetic GPR depth estimation
Educational synthetic model only; not real GPR measurements.
Uses known target geometry and calculates expected two-way travel times.
"""

import numpy as np
import matplotlib.pyplot as plt

C0 = 3.0e8
EPS_R = 9.0
VELOCITY = C0 / np.sqrt(EPS_R)   # 1e8 m/s
FS = 10e9
DT = 1.0 / FS
T_WINDOW = 120e-9
N_TIME = int(T_WINDOW * FS)
TIME = np.arange(N_TIME) * DT
TIME_NS = TIME * 1e9
SYSTEM_DELAY = 1.0e-9

X_POSITIONS = np.linspace(0.0, 5.0, 101)
TARGETS = [
    {"x": 1.5, "depth": 0.15, "amplitude": 0.40, "sigma_ns": 0.28},
    {"x": 3.5, "depth": 0.30, "amplitude": 0.30, "sigma_ns": 0.38},
]

def trace_at_x(x):
    trace = np.zeros_like(TIME)
    for target in TARGETS:
        offset = x - target["x"]
        distance_one_way = np.sqrt(target["depth"]**2 + offset**2)
        travel_time = 2.0 * distance_one_way / VELOCITY
        center = SYSTEM_DELAY + travel_time
        sigma = target["sigma_ns"] * 1e-9
        trace += target["amplitude"] * np.exp(
            -0.5 * ((TIME - center) / sigma) ** 2
        )
    trace += 0.025 * np.exp(-TIME / 3e-9) * np.sin(2 * np.pi * 180e6 * TIME)
    return trace

BSCAN = np.vstack([trace_at_x(x) for x in X_POSITIONS])
ix = int(np.argmin(np.abs(X_POSITIONS - 1.5)))
x_selected = X_POSITIONS[ix]
ASCAN = BSCAN[ix]

print("===== STAGE 19: DEPTH ESTIMATION =====")
print(f"Assumed propagation velocity: {VELOCITY:.3e} m/s")
print(f"System delay: {SYSTEM_DELAY*1e9:.2f} ns")
print(f"Sampling rate: {FS/1e9:.1f} GHz ({DT*1e9:.2f} ns/sample)")
print(f"Selected antenna position: {x_selected:.2f} m")
print("\nTarget travel-time and depth estimates at the selected antenna position:")
for i, target in enumerate(TARGETS, 1):
    offset = x_selected - target["x"]
    path_one_way = np.sqrt(target["depth"]**2 + offset**2)
    round_trip = 2 * path_one_way / VELOCITY
    peak_time = SYSTEM_DELAY + round_trip
    # Estimate the vertical depth only if the target is directly beneath antenna.
    if abs(offset) < 1e-9:
        estimated_depth = VELOCITY * (peak_time - SYSTEM_DELAY) / 2
        depth_note = f"estimated vertical depth={estimated_depth:.3f} m"
    else:
        estimated_depth = np.sqrt(max(0.0, (VELOCITY*(peak_time-SYSTEM_DELAY)/2)**2 - offset**2))
        depth_note = f"depth after lateral-offset correction={estimated_depth:.3f} m"
    print(
        f"Target {i}: x={target['x']:.2f} m, true model depth={target['depth']:.3f} m, "
        f"expected peak={peak_time*1e9:.2f} ns, {depth_note}"
    )

# Plot A-scan and label the expected target arrivals.
plt.figure(figsize=(11, 5.5))
plt.plot(TIME_NS, ASCAN, linewidth=1.3, label="Synthetic A-scan")
for i, target in enumerate(TARGETS, 1):
    offset = x_selected - target["x"]
    path = np.sqrt(target["depth"]**2 + offset**2)
    arrival_ns = (SYSTEM_DELAY + 2*path/VELOCITY) * 1e9
    plt.axvline(arrival_ns, linestyle="--",
                label=f"Target {i} expected arrival: {arrival_ns:.2f} ns")
plt.xlim(0, 50)
plt.title(f"Stage 19 — A-scan with Corrected Arrival Times (x={x_selected:.1f} m)")
plt.xlabel("Time (ns)")
plt.ylabel("Reflection amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("stage19_ascan_depth.png", dpi=160)
plt.show()

# B-scan
plt.figure(figsize=(11, 6))
plt.imshow(
    BSCAN.T, aspect="auto", origin="lower",
    extent=[X_POSITIONS[0], X_POSITIONS[-1], TIME_NS[0], TIME_NS[-1]],
    cmap="gray"
)
plt.ylim(0, 75)
plt.title("Stage 19 — Synthetic B-scan with Two Targets")
plt.xlabel("Antenna horizontal position (m)")
plt.ylabel("Time (ns)")
plt.colorbar(label="Reflection amplitude")
plt.tight_layout()
plt.savefig("stage19_bscan.png", dpi=160)
plt.show()

# Save synthetic B-scan data with the time axis as first column.
output = np.column_stack([TIME_NS, BSCAN.T])
header = "time_ns," + ",".join(f"x_{x:.3f}_m" for x in X_POSITIONS)
np.savetxt("stage19_bscan.csv", output, delimiter=",", header=header, comments="")
print("\nSaved: stage19_ascan_depth.png, stage19_bscan.png, stage19_bscan.csv")
print("Reminder: outputs are synthetic; actual depth estimation needs calibration and real measured data.")
