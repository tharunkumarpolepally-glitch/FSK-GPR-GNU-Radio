
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Project files
folder = Path(__file__).resolve().parent
reference_file = folder / "reference_stage53.dat"

if not reference_file.exists():
    raise FileNotFoundError(
        f"Cannot find {reference_file}"
    )

# Read the captured complex FSK reference
reference = np.fromfile(reference_file, dtype=np.complex64)

if reference.size < 4096:
    raise ValueError("Reference capture has too few samples.")

# Simulation settings
sample_rate = 500_000
trace_length = 4096
positions = np.linspace(0, 4, 201)

# Two simulated targets: (position in metres, delay in samples, gain)
targets = [
    (1.2, 80, 0.40),
    (2.8, 160, 0.30),
]

rng = np.random.default_rng(64)
base = reference[:trace_length].copy()

# Each column is a separate simulated received trace.
bscan = np.zeros(
    (trace_length, len(positions)),
    dtype=np.complex64
)

for j, x in enumerate(positions):
    received = base.copy()

    for target_x, delay, strength in targets:
        # Echo strength is greatest near the target position.
        spatial_gain = np.exp(
            -0.5 * ((x - target_x) / 0.16) ** 2
        )

        # Delayed copy of the captured FSK reference.
        echo = np.zeros_like(base)
        echo[delay:] = base[:-delay]

        received += (
            strength * spatial_gain * echo
        )

    # Add small simulated noise.
    noise = (
        rng.normal(0, 0.01, trace_length)
        + 1j * rng.normal(0, 0.01, trace_length)
    )
    bscan[:, j] = received + noise

# Save the generated dataset.
np.save(folder / "stage64_echo_dataset.npy", bscan)
np.save(folder / "stage64_scan_positions_m.npy", positions)

# Plot the real component as a radargram.
time_ms = np.arange(trace_length) / sample_rate * 1000
radargram = bscan.real

# Remove the mean of each trace to reduce direct-current background.
radargram = radargram - radargram.mean(axis=0, keepdims=True)

scale = np.max(np.abs(radargram))
if scale > 0:
    radargram = radargram / scale

fig, ax = plt.subplots(figsize=(12, 6))
image = ax.imshow(
    radargram,
    extent=[
        positions[0], positions[-1],
        time_ms[-1], time_ms[0]
    ],
    aspect="auto",
    cmap="gray",
    vmin=-1,
    vmax=1
)

ax.set_title("FSK-GPR B-Scan — Multi-Position Echo Simulation")
ax.set_xlabel("Scan position (m)")
ax.set_ylabel("Time (ms)")
fig.colorbar(image, ax=ax, label="Normalized amplitude")
fig.tight_layout()

output = folder / "stage64_mult_position_bscan.png"
fig.savefig(output, dpi=200)

print("Dataset saved:", folder / "stage64_echo_dataset.npy")
print("Positions saved:", folder / "stage64_scan_positions_m.npy")
print("B-scan saved:", output)
print("Number of scan positions:", len(positions))
print("Samples per trace:", trace_length)
print("Time per sample:", 1 / sample_rate * 1000, "ms")

plt.show()
