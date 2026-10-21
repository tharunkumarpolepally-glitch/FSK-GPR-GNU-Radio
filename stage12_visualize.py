
import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# STAGE 12: GNU RADIO SIGNAL VISUALIZATION
# ==========================================

filename = "gr_samples.dat"
fs = 500_000  # GNU Radio sample rate: 500 kS/s
num_samples = 200_000

if not os.path.exists(filename):
    raise FileNotFoundError(f"File not found: {filename}")

# Read only a small section of the large file
with open(filename, "rb") as f:
    samples = np.fromfile(
        f, dtype=np.complex64, count=num_samples
    )

if len(samples) == 0:
    raise ValueError("No samples were read.")

print("========== STAGE 12 RESULTS ==========")
print(f"Samples plotted: {len(samples):,}")
print(f"Sample rate: {fs / 1000:.1f} kS/s")
print(f"Plot duration: {len(samples) / fs * 1000:.2f} ms")
print(f"Mean magnitude: {np.mean(np.abs(samples)):.6f}")

# ==========================================
# 1. TIME-DOMAIN PLOT
# ==========================================

# Plot only the first 5,000 samples for readability
n_plot = min(5000, len(samples))
time_ms = np.arange(n_plot) / fs * 1000

plt.figure(figsize=(11, 4))
plt.plot(time_ms, samples.real[:n_plot], label="I / Real")
plt.plot(time_ms, samples.imag[:n_plot], label="Q / Imaginary")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.title("GNU Radio FSK-GPR Output: Time Domain")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("stage12_time.png", dpi=150)

# ==========================================
# 2. FREQUENCY-DOMAIN PLOT
# ==========================================

# Use a short section and remove its mean
n_fft = min(65536, len(samples))
x = samples[:n_fft]
x = x - np.mean(x)

window = np.hanning(n_fft)
spectrum = np.fft.fftshift(
    np.fft.fft(x * window)
)

freq_khz = np.fft.fftshift(
    np.fft.fftfreq(n_fft, d=1/fs)
) / 1000

magnitude = np.abs(spectrum)
spectrum_db = 20 * np.log10(
    magnitude / (np.max(magnitude) + 1e-12) + 1e-12
)

plt.figure(figsize=(11, 4))
plt.plot(freq_khz, spectrum_db)
plt.xlabel("Frequency (kHz)")
plt.ylabel("Relative magnitude (dB)")
plt.title("GNU Radio FSK-GPR Output: Spectrum")
plt.grid(True)
plt.tight_layout()
plt.savefig("stage12_spectrum.png", dpi=150)

print("Saved: stage12_time.png")
print("Saved: stage12_spectrum.png")

plt.show()
