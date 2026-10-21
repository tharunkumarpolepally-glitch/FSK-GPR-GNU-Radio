
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STAGE 59: SYNTHETIC GPR B-SCAN
# ============================================================

np.random.seed(59)

# Scan dimensions
n_positions = 201
n_time = 500
scan_length_m = 4.0

# Synthetic time axis
time_max_ns = 60.0
time_ns = np.linspace(0, time_max_ns, n_time)
x_m = np.linspace(0, scan_length_m, n_positions)

# Background and noise
bscan = 0.025 * np.random.randn(n_time, n_positions)

# Synthetic target model:
# Each target is described by horizontal position and two-way time.
targets = [
    {"x": 1.2, "time_ns": 15.0, "amplitude": 1.0},
    {"x": 2.8, "time_ns": 30.0, "amplitude": 0.8},
]

# Simulate localized reflection responses across scan positions
for target in targets:
    for j, x_pos in enumerate(x_m):
        lateral = np.exp(
            -0.5 * ((x_pos - target["x"]) / 0.16) ** 2
        )

        pulse = np.exp(
            -0.5 * ((time_ns - target["time_ns"]) / 0.7) ** 2
        )

        # Small hyperbolic moveout for an illustrative target
        moveout_ns = 2.0 * (
            np.sqrt(
                1.0 + ((x_pos - target["x"]) / 0.35) ** 2
            ) - 1.0
        )

        pulse = np.exp(
            -0.5
            * (
                (time_ns - target["time_ns"] - moveout_ns)
                / 0.7
            ) ** 2
        )

        bscan[:, j] += (
            target["amplitude"] * lateral * pulse
        )

# Normalize for display
bscan /= np.max(np.abs(bscan))

# Display B-scan
plt.figure(figsize=(12, 6))
plt.imshow(
    bscan,
    extent=[x_m[0], x_m[-1], time_ns[-1], time_ns[0]],
    aspect="auto",
    cmap="gray",
    interpolation="nearest",
    vmin=-1,
    vmax=1,
)

plt.xlabel("Scan position (m)")
plt.ylabel("Two-way time (ns)")
plt.title("Stage 59: Synthetic GPR B-Scan")
plt.colorbar(label="Normalized amplitude")
plt.tight_layout()
plt.savefig("stage59_synthetic_bscan.png", dpi=150)
plt.show()

print("=" * 55)
print("STAGE 59: SYNTHETIC B-SCAN")
print("=" * 55)
print(f"Scan positions: {n_positions}")
print(f"Time samples per position: {n_time}")
print(f"Scan length: {scan_length_m:.2f} m")
print(f"Time window: {time_max_ns:.1f} ns")
print(f"Synthetic targets: {len(targets)}")
print("Saved: stage59_synthetic_bscan.png")
print(
    "Note: This is a synthetic visualization, "
    "not measured subsurface data."
)
