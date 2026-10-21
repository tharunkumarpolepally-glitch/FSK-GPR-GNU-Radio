
import os
import numpy as np
import matplotlib.pyplot as plt

FILE = "demod_echo_stage51.dat"
FS = 500_000

if not os.path.exists(FILE):
    raise FileNotFoundError(f"File not found: {FILE}")

data = np.fromfile(FILE, dtype=np.float32)

if len(data) == 0:
    raise ValueError("The capture file is empty.")

print("=" * 55)
print("STAGE 52: RECEIVED-SIGNAL DEMODULATION ANALYSIS")
print("=" * 55)
print(f"Samples: {len(data):,}")
print(f"File size: {os.path.getsize(FILE):,} bytes")
print(f"Duration: {len(data)/FS:.6f} seconds")
print(f"Minimum: {np.min(data):.6f}")
print(f"Maximum: {np.max(data):.6f}")
print(f"Mean: {np.mean(data):.6f}")
print(f"Standard deviation: {np.std(data):.6f}")

negative = data[data < 0]
positive = data[data >= 0]

print("\nOutput states:")
print(f"Negative samples: {len(negative):,}")
print(f"Positive samples: {len(positive):,}")

if len(negative):
    print(f"Negative mean: {np.mean(negative):.6f}")
if len(positive):
    print(f"Positive mean: {np.mean(positive):.6f}")

# Plot first 100 samples
n = min(100, len(data))
time_ms = np.arange(n) / FS * 1000

plt.figure(figsize=(12, 5))
plt.plot(time_ms, data[:n], marker=".", linewidth=0.8)
plt.axhline(0, linestyle="--")
plt.xlabel("Time (ms)")
plt.ylabel("Demodulator output")
plt.title("Stage 52: Demodulated Simulated Received Signal")
plt.grid(True)
plt.tight_layout()
plt.savefig("stage52_echo_demod_output.png", dpi=150)
plt.show()

# Plot output distribution
plt.figure(figsize=(9, 5))
plt.hist(data, bins=100)
plt.xlabel("Demodulator output")
plt.ylabel("Number of samples")
plt.title("Stage 52: Demodulator Output Distribution")
plt.grid(True)
plt.tight_layout()
plt.savefig("stage52_echo_demod_histogram.png", dpi=150)
plt.show()

print("\nSaved: stage52_echo_demod_output.png")
print("Saved: stage52_echo_demod_histogram.png")
