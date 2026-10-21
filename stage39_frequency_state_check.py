
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ==================================================
# STAGE 39: FSK FREQUENCY-STATE ANALYSIS
# ==================================================

FOLDER = r"C:\Users\tharunkumar\Desktop\FSK_GPR"
FS = 500_000
COUNT = 100_000

FILE = os.path.join(FOLDER, "reference_stage35.dat")
PNG = os.path.join(FOLDER, "stage39_frequency_distribution.png")
CSV = os.path.join(FOLDER, "stage39_frequency_metrics.csv")

# 1. Load and validate data
if not os.path.isfile(FILE):
    raise FileNotFoundError(f"File not found: {FILE}")

x = np.fromfile(FILE, dtype=np.complex64, count=COUNT)

if len(x) < 1000:
    raise ValueError("Not enough samples in the reference file.")

if not np.all(np.isfinite(x)):
    raise ValueError("Signal contains NaN or infinite values.")

# 2. Estimate instantaneous frequency
phase_diff = np.angle(x[1:] * np.conj(x[:-1]))
freq = phase_diff * FS / (2 * np.pi)

# 3. Smooth the estimate
freq_smooth = uniform_filter1d(freq, size=20)

# Exclude edges affected by smoothing
valid = freq_smooth[100:-100]

negative = valid[valid < 0]
positive = valid[valid >= 0]

# 4. Calculate statistics
print("=" * 55)
print("STAGE 39: FSK FREQUENCY-STATE ANALYSIS")
print("=" * 55)
print(f"Input file: {FILE}")
print(f"Samples loaded: {len(x):,}")
print(f"Sample rate: {FS:,} samples/s")
print(f"Capture duration: {len(x) / FS * 1000:.3f} ms")
print(f"Frequency samples analyzed: {len(valid):,}")

print("\nFrequency statistics:")
print(f"Mean frequency: {np.mean(valid):.2f} Hz")
print(f"Standard deviation: {np.std(valid):.2f} Hz")
print(f"Minimum frequency: {np.min(valid):.2f} Hz")
print(f"Maximum frequency: {np.max(valid):.2f} Hz")

print("\nPositive/negative frequency statistics:")
print(f"Negative samples: {len(negative):,}")
print(f"Positive samples: {len(positive):,}")

if len(negative):
    print(f"Negative mean: {np.mean(negative):.2f} Hz")
if len(positive):
    print(f"Positive mean: {np.mean(positive):.2f} Hz")

# 5. Plot frequency histogram
plt.figure(figsize=(11, 5))
plt.hist(valid, bins=100)
plt.xlabel("Estimated instantaneous frequency (Hz)")
plt.ylabel("Number of samples")
plt.title("Stage 39 — FSK Frequency Distribution")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PNG, dpi=150)
plt.show()

# 6. Save metrics
with open(CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["sample_rate_hz", FS])
    writer.writerow(["samples_loaded", len(x)])
    writer.writerow(["duration_ms", len(x) / FS * 1000])
    writer.writerow(["mean_frequency_hz", np.mean(valid)])
    writer.writerow(["frequency_std_hz", np.std(valid)])
    writer.writerow(["minimum_frequency_hz", np.min(valid)])
    writer.writerow(["maximum_frequency_hz", np.max(valid)])
    writer.writerow(["negative_sample_count", len(negative)])
    writer.writerow(["positive_sample_count", len(positive)])
    if len(negative):
        writer.writerow(["negative_mean_hz", np.mean(negative)])
    if len(positive):
        writer.writerow(["positive_mean_hz", np.mean(positive)])

print("\nSaved files:")
print(PNG)
print(CSV)
print("\nAnalysis complete.")
print("A histogram alone does not prove correct binary FSK modulation.")
