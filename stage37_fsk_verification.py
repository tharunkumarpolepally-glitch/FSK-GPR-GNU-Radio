
import os
import numpy as np
import matplotlib.pyplot as plt

FOLDER = r"C:\Users\tharunkumar\Desktop\FSK_GPR"
FS = 500_000
COUNT = 100_000

FILE = os.path.join(FOLDER, "reference_stage35.dat")
PNG = os.path.join(FOLDER, "stage37_fsk_analysis.png")
CSV = os.path.join(FOLDER, "stage37_fsk_metrics.csv")

if not os.path.isfile(FILE):
    raise FileNotFoundError(
        f"Cannot find {FILE}. Check the filename and folder."
    )

x = np.fromfile(FILE, dtype=np.complex64, count=COUNT)

if len(x) < 1000:
    raise ValueError("The reference file contains too few samples.")

if not np.all(np.isfinite(x)):
    raise ValueError("Signal contains invalid values.")

# Instantaneous phase difference estimates frequency relative
# to the complex signal's sampling rate.
phase_difference = np.angle(x[1:] * np.conj(x[:-1]))
instantaneous_frequency = phase_difference * FS / (2 * np.pi)

# Spectrum
window = np.hanning(len(x))
spectrum = np.fft.fftshift(np.fft.fft(x * window))
frequency = np.fft.fftshift(np.fft.fftfreq(len(x), d=1 / FS))
magnitude_db = 20 * np.log10(
    np.maximum(np.abs(spectrum) / np.max(np.abs(spectrum)), 1e-12)
)

# Basic statistics
magnitude = np.abs(x)
freq_mean = float(np.mean(instantaneous_frequency))
freq_std = float(np.std(instantaneous_frequency))
freq_min = float(np.min(instantaneous_frequency))
freq_max = float(np.max(instantaneous_frequency))

print("===== STAGE 37: FSK SIGNAL VERIFICATION =====")
print(f"File: {FILE}")
print(f"Samples analyzed: {len(x):,}")
print(f"Sample rate: {FS:,} samples/s")
print(f"Duration: {len(x) / FS * 1000:.3f} ms")
print(f"Mean magnitude: {np.mean(magnitude):.6f}")
print(f"Minimum magnitude: {np.min(magnitude):.6f}")
print(f"Maximum magnitude: {np.max(magnitude):.6f}")
print(f"Mean instantaneous frequency: {freq_mean:.2f} Hz")
print(f"Frequency standard deviation: {freq_std:.2f} Hz")
print(f"Instantaneous frequency range: {freq_min:.2f} to {freq_max:.2f} Hz")

# Plot first 4 ms of I/Q and instantaneous frequency
show = min(len(x), int(0.004 * FS))
time_ms = np.arange(show) / FS * 1000

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(time_ms, x.real[:show], label="I")
axes[0].plot(time_ms, x.imag[:show], label="Q")
axes[0].set_title("Stage 37 — FSK Reference I/Q")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Amplitude")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(
    np.arange(len(instantaneous_frequency[:show - 1])) / FS * 1000,
    instantaneous_frequency[:show - 1]
)
axes[1].set_title("Instantaneous Frequency Estimate")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Frequency (Hz)")
axes[1].grid(True)

axes[2].plot(frequency / 1000, magnitude_db)
axes[2].set_title("Normalized Signal Spectrum")
axes[2].set_xlabel("Frequency (kHz)")
axes[2].set_ylabel("Magnitude (dB)")
axes[2].set_xlim(-FS / 2000, FS / 2000)
axes[2].grid(True)

plt.tight_layout()
plt.savefig(PNG, dpi=150)

# Save reproducible summary metrics
metrics = np.array([
    ["sample_rate_hz", str(FS)],
    ["samples_analyzed", str(len(x))],
    ["duration_ms", f"{len(x) / FS * 1000:.6f}"],
    ["mean_magnitude", f"{np.mean(magnitude):.9f}"],
    ["minimum_magnitude", f"{np.min(magnitude):.9f}"],
    ["maximum_magnitude", f"{np.max(magnitude):.9f}"],
    ["mean_instantaneous_frequency_hz", f"{freq_mean:.6f}"],
    ["frequency_std_hz", f"{freq_std:.6f}"],
    ["minimum_instantaneous_frequency_hz", f"{freq_min:.6f}"],
    ["maximum_instantaneous_frequency_hz", f"{freq_max:.6f}"]
], dtype=str)

np.savetxt(
    CSV, metrics, delimiter=",", fmt="%s",
    header="Metric,Value", comments=""
)

print(f"\nSaved plot: {PNG}")
print(f"Saved metrics: {CSV}")
plt.show()
