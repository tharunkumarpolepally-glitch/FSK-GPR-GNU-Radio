
import os
import numpy as np
import matplotlib.pyplot as plt

FILE = "demod_stage49.dat"
FS = 500_000

if not os.path.exists(FILE):
    raise FileNotFoundError(f"Missing file: {FILE}")

data = np.fromfile(FILE, dtype=np.float32)

if len(data) == 0:
    raise ValueError("The demodulator file is empty.")

print("=" * 55)
print("STAGE 50: ACTUAL GNU RADIO DEMODULATOR ANALYSIS")
print("=" * 55)
print(f"Samples: {len(data):,}")
print(f"File size: {os.path.getsize(FILE):,} bytes")
print(f"Sample rate: {FS:,} samples/s")
print(f"Duration: {len(data)/FS:.6f} seconds")
print(f"Minimum: {np.min(data):.6f}")
print(f"Maximum: {np.max(data):.6f}")
print(f"Mean: {np.mean(data):.6f}")
print(f"Standard deviation: {np.std(data):.6f}")

# Ignore the first sample when calculating the state distribution.
valid = data[1:]
negative = valid[valid < 0]
positive = valid[valid >= 0]

print("\nDemodulated state statistics:")
print(f"Negative-state samples: {len(negative):,}")
print(f"Positive-state samples: {len(positive):,}")

if len(negative):
    print(f"Negative-state mean: {np.mean(negative):.6f}")
if len(positive):
    print(f"Positive-state mean: {np.mean(positive):.6f}")

# Plot a short section so individual symbol states are visible.
n = min(100, len(data))
time_ms = np.arange(n) / FS * 1000

plt.figure(figsize=(12, 5))
plt.plot(time_ms, data[:n], linewidth=0.8)
plt.axhline(0, linestyle="--")
plt.xlabel("Time (ms)")
plt.ylabel("Quadrature Demod output")
plt.title("Stage 50: Actual GNU Radio FSK Demodulator Output")
plt.grid(True)
plt.tight_layout()
plt.savefig("stage50_actual_demod_output.png", dpi=150)
plt.show()

# Save numerical summary.
with open("stage50_actual_demod_summary.txt", "w") as f:
    f.write(f"Samples: {len(data)}\n")
    f.write(f"Minimum: {np.min(data):.8f}\n")
    f.write(f"Maximum: {np.max(data):.8f}\n")
    f.write(f"Mean: {np.mean(data):.8f}\n")
    f.write(f"Standard deviation: {np.std(data):.8f}\n")
    f.write(f"Negative-state samples: {len(negative)}\n")
    f.write(f"Positive-state samples: {len(positive)}\n")

print("\nSaved: stage50_actual_demod_output.png")
print("Saved: stage50_actual_demod_summary.txt")
