
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

FOLDER = r"C:\Users\tharunkumar\Desktop\FSK_GPR"
FS = 500_000
COUNT = 100_000
EXPECTED_DELAY = 250

REF_FILE = os.path.join(FOLDER, "reference_stage35.dat")
RX_FILE = os.path.join(FOLDER, "received_stage35.dat")
SAMPLES_FILE = os.path.join(FOLDER, "gr_samples_stage35.dat")

PNG_FILE = os.path.join(FOLDER, "stage40_fresh_capture_validation.png")
CSV_FILE = os.path.join(FOLDER, "stage40_validation_summary.csv")


def load_complex_file(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Missing file: {filename}")

    file_size = os.path.getsize(filename)

    if file_size % 8 != 0:
        raise ValueError(
            f"{os.path.basename(filename)} is not aligned "
            "to complete complex64 samples."
        )

    data = np.fromfile(filename, dtype=np.complex64, count=COUNT)

    if len(data) == 0:
        raise ValueError(f"No samples found in {filename}")

    if not np.all(np.isfinite(data)):
        raise ValueError(f"Invalid samples found in {filename}")

    return data, file_size


print("=" * 55)
print("STAGE 40: FRESH CAPTURE VALIDATION")
print("=" * 55)

ref, ref_size = load_complex_file(REF_FILE)
rx, rx_size = load_complex_file(RX_FILE)
gr, gr_size = load_complex_file(SAMPLES_FILE)

n = min(len(ref), len(rx), len(gr))
ref, rx, gr = ref[:n], rx[:n], gr[:n]

print(f"Samples analyzed per file: {n:,}")
print(f"Sample rate: {FS:,} samples/s")
print(f"Analysis duration: {n / FS * 1000:.3f} ms")

for name, size, data in [
    ("Reference", ref_size, ref),
    ("Received", rx_size, rx),
    ("GR samples", gr_size, gr),
]:
    print(f"\n{name}:")
    print(f"  File size: {size:,} bytes")
    print(f"  Mean magnitude: {np.mean(np.abs(data)):.6f}")
    print(f"  Maximum magnitude: {np.max(np.abs(data)):.6f}")

# Check whether GR samples matches the received capture.
gr_error = np.mean(np.abs(gr - rx))
print(f"\nMean |gr_samples - received|: {gr_error:.8f}")

# Estimate direct-path scale using the first 250 samples.
n_direct = min(EXPECTED_DELAY, n)
denominator = np.vdot(ref[:n_direct], ref[:n_direct])

if abs(denominator) < 1e-12:
    raise ValueError("Reference direct-path segment has near-zero energy.")

scale = np.vdot(ref[:n_direct], rx[:n_direct]) / denominator
residual = rx - scale * ref

before = np.abs(residual[:n_direct])
after_end = min(n, EXPECTED_DELAY + 1000)
after = np.abs(residual[EXPECTED_DELAY:after_end])

print("\nDirect-path and delay checks:")
print(f"  Estimated direct scale: {scale}")
print(f"  Mean residual before delay: {np.mean(before):.8f}")
print(f"  Mean residual after delay: {np.mean(after):.8f}")

# Correlation using a bounded segment for manageable memory use.
SEGMENT = min(20_000, n - EXPECTED_DELAY)

if SEGMENT < 2000:
    raise ValueError("Not enough samples for reliable correlation analysis.")

template = ref[:SEGMENT]
residual_segment = residual[:SEGMENT + 2000]

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
lags_positive = lags[mask]
corr_positive = np.abs(corr[mask])

peak_index = int(np.argmax(corr_positive))
estimated_lag = int(lags_positive[peak_index])
peak_value = float(corr_positive[peak_index])

print("\nCorrelation delay estimate:")
print(f"  Expected delay: {EXPECTED_DELAY} samples")
print(f"  Expected delay time: {EXPECTED_DELAY / FS * 1000:.6f} ms")
print(f"  Estimated strongest lag: {estimated_lag} samples")
print(f"  Estimated lag time: {estimated_lag / FS * 1000:.6f} ms")
print(f"  Correlation peak: {peak_value:.3f}")

if estimated_lag == EXPECTED_DELAY:
    print("  Result: PASS — strongest lag matches configured software delay.")
else:
    print("  Result: CHECK — strongest lag does not match expected delay.")

print("\nImportant:")
print("This validates a software capture, not a physical radar echo.")
print("At 500 kS/s, this setup cannot resolve nanosecond-scale GPR depths.")

# Plot signals and correlation
show = min(2000, n)
time_ms = np.arange(show) / FS * 1000

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(time_ms, np.abs(ref[:show]), label="Reference")
axes[0].plot(time_ms, np.abs(rx[:show]), label="Received", alpha=0.8)
axes[0].set_title("Stage 40 — Fresh Reference and Received Signals")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Magnitude")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(time_ms, np.abs(residual[:show]), label="Residual")
axes[1].axvline(
    EXPECTED_DELAY / FS * 1000,
    linestyle="--",
    label="Expected delay"
)
axes[1].set_title("Residual After Direct-Path Subtraction")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Magnitude")
axes[1].grid(True)
axes[1].legend()

axes[2].plot(lags_positive, corr_positive)
axes[2].axvline(
    EXPECTED_DELAY,
    color="red",
    linestyle="--",
    label="Expected lag"
)
axes[2].axvline(
    estimated_lag,
    color="green",
    linestyle=":",
    label="Estimated lag"
)
axes[2].set_title("Residual-to-Reference Correlation")
axes[2].set_xlabel("Lag (samples)")
axes[2].set_ylabel("Correlation magnitude")
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.savefig(PNG_FILE, dpi=150)
plt.close(fig)

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["samples_analyzed", n])
    writer.writerow(["reference_file_bytes", ref_size])
    writer.writerow(["received_file_bytes", rx_size])
    writer.writerow(["gr_samples_file_bytes", gr_size])
    writer.writerow(["sample_rate_hz", FS])
    writer.writerow(["expected_delay_samples", EXPECTED_DELAY])
    writer.writerow(["estimated_delay_samples", estimated_lag])
    writer.writerow(["estimated_delay_ms", estimated_lag / FS * 1000])
    writer.writerow(["mean_gr_received_difference", gr_error])
    writer.writerow(["direct_scale_real", scale.real])
    writer.writerow(["direct_scale_imag", scale.imag])

print("\nSaved:")
print(PNG_FILE)
print(CSV_FILE)
