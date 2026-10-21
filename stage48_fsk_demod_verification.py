
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv

FOLDER = Path(r"C:\Users\tharunkumar\Desktop\FSK_GPR")
FS = 500_000
GAIN = FS / (2 * np.pi * 25_000)

# Read the transmitted reference signal.
path = FOLDER / "reference_stage45.dat"
if not path.exists():
    raise FileNotFoundError(path)

signal = np.fromfile(path, dtype=np.complex64)
if len(signal) < 2:
    raise ValueError("Not enough samples.")

# Equivalent of the phase-difference operation used by
# a quadrature demodulator, scaled to frequency in Hz.
phase_diff = np.angle(signal[1:] * np.conj(signal[:-1]))
frequency_hz = phase_diff * FS / (2 * np.pi)

# GNU Radio Quadrature Demod output with the configured gain.
demod_output = phase_diff * GAIN

# Report the two sign groups.
negative = frequency_hz[frequency_hz < 0]
positive = frequency_hz[frequency_hz >= 0]

print("=" * 60)
print("STAGE 48: FSK DEMODULATION VERIFICATION")
print("=" * 60)
print(f"Input samples: {len(signal):,}")
print(f"Sample rate: {FS:,} samples/s")
print(f"Quadrature Demod gain: {GAIN:.6f}")
print()
print("Frequency estimate from reference signal:")
print(f"Minimum: {frequency_hz.min():.2f} Hz")
print(f"Maximum: {frequency_hz.max():.2f} Hz")
print(f"Mean: {frequency_hz.mean():.2f} Hz")
print()
print("Sign-group means:")
print(f"Negative group mean: {negative.mean():.2f} Hz")
print(f"Positive group mean: {positive.mean():.2f} Hz")
print()
print("Demodulator output statistics:")
print(f"Minimum: {demod_output.min():.6f}")
print(f"Maximum: {demod_output.max():.6f}")
print(f"Mean: {demod_output.mean():.6f}")
print()
print("Note: This calculates a demodulator equivalent from the")
print("reference capture; it does not read a captured GNU Radio")
print("Quadrature Demod output file.")

# Save numerical results.
csv_path = FOLDER / "stage48_fsk_demod_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Input samples", len(signal)])
    writer.writerow(["Sample rate Hz", FS])
    writer.writerow(["Quadrature Demod gain", GAIN])
    writer.writerow(["Minimum frequency Hz", frequency_hz.min()])
    writer.writerow(["Maximum frequency Hz", frequency_hz.max()])
    writer.writerow(["Mean frequency Hz", frequency_hz.mean()])
    writer.writerow(["Negative group mean Hz", negative.mean()])
    writer.writerow(["Positive group mean Hz", positive.mean()])
    writer.writerow(["Demod output minimum", demod_output.min()])
    writer.writerow(["Demod output maximum", demod_output.max()])
    writer.writerow(["Demod output mean", demod_output.mean()])

# Plot a short segment.
n = min(2000, len(demod_output))
time_ms = np.arange(n) / FS * 1000

fig, axes = plt.subplots(2, 1, figsize=(11, 7))
axes[0].plot(time_ms, frequency_hz[:n] / 1000)
axes[0].set_title("Stage 48: Estimated Instantaneous Frequency")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Frequency (kHz)")
axes[0].grid(True)

axes[1].plot(time_ms, demod_output[:n])
axes[1].set_title("Quadrature Demod Equivalent Output")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Output (scaled)")
axes[1].grid(True)

plt.tight_layout()
plot_path = FOLDER / "stage48_fsk_demod_verification.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print(f"\nSaved CSV: {csv_path}")
print(f"Saved graph: {plot_path}")
