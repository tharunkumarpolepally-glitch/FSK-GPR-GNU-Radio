
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path(__file__).resolve().parent
received_file = folder / "received_stage53.dat"

if not received_file.exists():
    raise FileNotFoundError(f"Missing file: {received_file}")

# Read the complex received-signal capture.
received = np.fromfile(received_file, dtype=np.complex64)

if received.size == 0:
    raise ValueError("Received capture is empty.")

# A short reference trace from the real capture.
trace_length = min(4096, received.size)
base = received[:trace_length].real
base = base - np.mean(base)

# Simulated scan geometry.
positions = np.linspace(0, 4, 201)
time_ns = np.linspace(0, 60, 500)
dt_ns = time_ns[1] - time_ns[0]

# Synthetic radargram; targets are modeled explicitly.
rng = np.random.default_rng(7)
bscan = 0.025 * rng.standard_normal((len(time_ns), len(positions)))

targets = [
    (1.2, 15.0, 1.0),
    (2.8, 30.0, 0.8),
]

for x0, t0, amplitude in targets:
    for j, x in enumerate(positions):
        # Localized response across neighboring scan positions.
        spatial = np.exp(-0.5 * ((x - x0) / 0.16) ** 2)
        # Short synthetic pulse around the selected two-way time.
        pulse = np.exp(-0.5 * ((time_ns - t0) / 0.8) ** 2)
        bscan[:, j] += amplitude * spatial * pulse

# Normalize for display.
bscan /= np.max(np.abs(bscan))

fig, ax = plt.subplots(figsize=(12, 6))
image = ax.imshow(
    bscan,
    extent=[positions[0], positions[-1], time_ns[-1], time_ns[0]],
    aspect="auto",
    cmap="gray",
    vmin=-1,
    vmax=1,
)
ax.set_title("FSK-GPR B-Scan — Synthetic Echo Model")
ax.set_xlabel("Scan position (m)")
ax.set_ylabel("Two-way time (ns)")
fig.colorbar(image, ax=ax, label="Normalized amplitude")
fig.tight_layout()

output = folder / "captured_gpr_bscan.png"
fig.savefig(output, dpi=200)
print(f"Saved: {output}")
print("Note: This B-scan uses a synthetic target model.")
plt.show()
