
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

FOLDER = r"C:\Users\tharunkumar\Desktop\FSK_GPR"
FS = 500_000
COUNT = 100_000
EXPECTED_DELAY = 250

ref_path = os.path.join(FOLDER, "reference_stage35.dat")
rx_path = os.path.join(FOLDER, "received_stage35.dat")

for path in (ref_path, rx_path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

reference = np.fromfile(
    ref_path, dtype=np.complex64, count=COUNT
)
received = np.fromfile(
    rx_path, dtype=np.complex64, count=COUNT
)

n = min(len(reference), len(received))
reference = reference[:n]
received = received[:n]

if n <= EXPECTED_DELAY + 2000:
    raise ValueError("Insufficient samples for analysis.")

# Estimate direct-path scale from samples before the delayed component.
direct_ref = reference[:EXPECTED_DELAY]
direct_rx = received[:EXPECTED_DELAY]

energy = np.vdot(direct_ref, direct_ref)
if abs(energy) < 1e-12:
    raise ValueError("Reference direct-path energy is too small.")

scale = np.vdot(direct_ref, direct_rx) / energy
residual = received - scale * reference

# Correlate the received residual with the reference.
# Use a short segment to limit memory consumption.
segment_length = min(20_000, n - 2000)
template = reference[:segment_length]
residual_segment = residual[:segment_length + 2000]

corr = correlate(
    residual_segment,
    template,
    mode="full",
    method="fft"
)
lags = correlation_lags(
    len(residual_segment),
    len(template),
    mode="full"
)

mask = (lags >= 0) & (lags <= 2000)
positive_lags = lags[mask]
positive_corr = np.abs(corr[mask])

peak_index = int(np.argmax(positive_corr))
estimated_lag = int(positive_lags[peak_index])
delay_seconds = estimated_lag / FS
delay_ms = delay_seconds * 1000

print("=" * 55)
print("STAGE 41: ECHO DELAY VISUALIZATION")
print("=" * 55)
print(f"Samples analyzed: {n:,}")
print(f"Sample rate: {FS:,} samples/s")
print(f"Estimated direct scale: {scale}")
print(f"Expected delay: {EXPECTED_DELAY} samples")
print(f"Estimated delay: {estimated_lag} samples")
print(f"Estimated delay: {delay_ms:.6f} ms")
print(f"Correlation peak: {positive_corr[peak_index]:.3f}")

if estimated_lag == EXPECTED_DELAY:
    print("Delay check: PASS")
else:
    print("Delay check: CHECK")

# Plot a short segment so the delay is visible.
show = min(3000, n)
time_ms = np.arange(show) / FS * 1000

fig, axes = plt.subplots(3, 1, figsize=(12, 9))

axes[0].plot(time_ms, np.abs(reference[:show]),
             label="Reference")
axes[0].plot(time_ms, np.abs(received[:show]),
             label="Received", alpha=0.8)
axes[0].set_title("Stage 41 — Reference and Received Magnitudes")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Magnitude")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(time_ms, np.abs(residual[:show]),
             label="Residual magnitude")
axes[1].axvline(
    EXPECTED_DELAY / FS * 1000,
    color="red", linestyle="--",
    label="Expected delay"
)
axes[1].set_title("Residual After Direct-Path Subtraction")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Magnitude")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(positive_lags, positive_corr)
axes[2].axvline(
    EXPECTED_DELAY, color="red",
    linestyle="--", label="Expected lag"
)
axes[2].axvline(
    estimated_lag, color="green",
    linestyle=":", label="Estimated lag"
)
axes[2].set_title("Residual-to-Reference Correlation")
axes[2].set_xlabel("Lag (samples)")
axes[2].set_ylabel("Correlation magnitude")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()

png_path = os.path.join(
    FOLDER, "stage41_echo_delay_visualization.png"
)
csv_path = os.path.join(
    FOLDER, "stage41_echo_delay_summary.csv"
)

plt.savefig(png_path, dpi=150)
plt.close(fig)

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["samples_analyzed", n])
    writer.writerow(["sample_rate_hz", FS])
    writer.writerow(["expected_delay_samples", EXPECTED_DELAY])
    writer.writerow(["estimated_delay_samples", estimated_lag])
    writer.writerow(["estimated_delay_ms", delay_ms])
    writer.writerow(["direct_scale_real", scale.real])
    writer.writerow(["direct_scale_imag", scale.imag])
    writer.writerow(["correlation_peak", positive_corr[peak_index]])

print("\nSaved:")
print(png_path)
print(csv_path)
print("Note: this is a software echo simulation, not measured GPR data.")
