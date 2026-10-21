
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv

FOLDER = Path(r"C:\Users\tharunkumar\Desktop\FSK_GPR")
FS = 500_000
EXPECTED_DEVIATION = 25_000

source_file = FOLDER / "reference_stage45.dat"

if not source_file.exists():
    raise FileNotFoundError(f"Missing file: {source_file}")

signal = np.fromfile(source_file, dtype=np.complex64)

if len(signal) < 10:
    raise ValueError("Not enough samples for analysis.")

# Estimate instantaneous frequency from phase increments.
phase_difference = np.angle(signal[1:] * np.conj(signal[:-1]))
frequency_hz = phase_difference * FS / (2 * np.pi)

# Ignore non-finite values.
frequency_hz = frequency_hz[np.isfinite(frequency_hz)]

# Divide samples by frequency sign.
negative = frequency_hz[frequency_hz < 0]
positive = frequency_hz[frequency_hz >= 0]

print("=" * 60)
print("STAGE 47: FSK STATE VERIFICATION")
print("=" * 60)
print(f"Samples analyzed: {len(signal):,}")
print(f"Sample rate: {FS:,} samples/s")
print(f"Expected deviation: +/-{EXPECTED_DEVIATION:,} Hz")

print("\nInstantaneous-frequency statistics:")
print(f"Minimum: {np.min(frequency_hz):.2f} Hz")
print(f"Maximum: {np.max(frequency_hz):.2f} Hz")
print(f"Mean: {np.mean(frequency_hz):.2f} Hz")
print(f"Standard deviation: {np.std(frequency_hz):.2f} Hz")

print("\nFrequency samples split by sign:")
if len(negative):
    print(f"Negative-state mean: {np.mean(negative):.2f} Hz")
    print(f"Negative-state standard deviation: {np.std(negative):.2f} Hz")
else:
    print("No negative-frequency samples.")

if len(positive):
    print(f"Positive-state mean: {np.mean(positive):.2f} Hz")
    print(f"Positive-state standard deviation: {np.std(positive):.2f} Hz")
else:
    print("No positive-frequency samples.")

# Save CSV with all frequency estimates.
csv_path = FOLDER / "stage47_fsk_frequency_samples.csv"
with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Sample_index", "Instantaneous_frequency_Hz"])
    writer.writerows(
        (index, float(value))
        for index, value in enumerate(frequency_hz)
    )

# Plot the frequency estimates and their distribution.
fig, axes = plt.subplots(2, 1, figsize=(11, 7))

plot_count = min(2000, len(frequency_hz))
time_ms = np.arange(plot_count) / FS * 1000

axes[0].plot(time_ms, frequency_hz[:plot_count] / 1000)
axes[0].axhline(EXPECTED_DEVIATION / 1000, linestyle="--")
axes[0].axhline(-EXPECTED_DEVIATION / 1000, linestyle="--")
axes[0].set_title("Stage 47: Reference Instantaneous Frequency")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Frequency (kHz)")
axes[0].grid(True)

axes[1].hist(frequency_hz / 1000, bins=100)
axes[1].set_title("Frequency Distribution")
axes[1].set_xlabel("Frequency (kHz)")
axes[1].set_ylabel("Sample count")
axes[1].grid(True)

plt.tight_layout()

plot_path = FOLDER / "stage47_fsk_state_verification.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print(f"\nSaved frequency data: {csv_path}")
print(f"Saved verification plot: {plot_path}")
print("\nNote: This checks the digital reference signal, not physical GPR.")
