
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# ==========================================
# STAGE 15: GPR RECEIVER PROCESSING
# ==========================================

FS = 500_000  # Sample rate: 500 kS/s
PROPAGATION_VELOCITY = 1e8  # Assumed soil velocity (m/s)

FOLDER = os.path.dirname(os.path.abspath(__file__))
REFERENCE_FILE = os.path.join(FOLDER, "reference.dat")
RECEIVED_FILE = os.path.join(FOLDER, "received.dat")

# GNU Radio complex float files normally store
# each sample as two float32 values: I and Q.
def load_complex_file(filename, max_samples=2_000_000):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    data = np.fromfile(filename, dtype=np.complex64,
                       count=max_samples)

    if data.size == 0:
        raise ValueError(f"No samples could be read: {filename}")

    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite samples found: {filename}")

    return data

print("Loading GNU Radio samples...")

reference = load_complex_file(REFERENCE_FILE)
received = load_complex_file(RECEIVED_FILE)

# Use equal-length sections for a fair comparison.
length = min(len(reference), len(received))
reference = reference[:length]
received = received[:length]

# Remove DC components.
reference = reference - np.mean(reference)
received = received - np.mean(received)

energy = np.sum(np.abs(reference) ** 2)

if energy <= 0:
    raise ValueError("Reference signal has zero energy.")

# ==========================================
# MATCHED FILTER
# ==========================================

print("Calculating matched-filter response...")

# Correlation using FFT is faster for long records.
nfft = 1 << (2 * length - 1).bit_length()

corr = np.fft.ifft(
    np.fft.fft(received, nfft)
    * np.conj(np.fft.fft(reference, nfft))
)

# Keep non-negative lags.
corr = corr[:length]

response = np.abs(corr) / energy

# Ignore lag zero when searching for a delayed echo.
search_start = 1
peak_index = search_start + np.argmax(response[search_start:])

delay_seconds = peak_index / FS
delay_ms = delay_seconds * 1000
apparent_depth = delay_seconds * PROPAGATION_VELOCITY / 2

print("\n========== STAGE 15 RESULTS ==========")
print(f"Samples processed: {length:,}")
print(f"Sample rate: {FS:,} samples/s")
print(f"Estimated lag: {peak_index} samples")
print(f"Estimated delay: {delay_ms:.6f} ms")
print(f"Apparent depth: {apparent_depth:.3f} m")
print("This is a software estimate, not a physical measurement.")

# ==========================================
# SAVE A-SCAN
# ==========================================

time_ms = np.arange(length) / FS * 1000

plt.figure(figsize=(11, 5))
plt.plot(time_ms, response, linewidth=0.8)
plt.axvline(
    delay_ms,
    color="red",
    linestyle="--",
    label=f"Peak delay = {delay_ms:.6f} ms"
)
plt.xlabel("Delay (ms)")
plt.ylabel("Normalized correlation")
plt.title("Stage 15: GNU Radio FSK-GPR Receiver A-scan")
plt.grid(True)
plt.legend()
plt.tight_layout()

ascan_path = os.path.join(FOLDER, "stage15_ascan.png")
plt.savefig(ascan_path, dpi=150)
plt.show()

# ==========================================
# SAVE DETECTION REPORT
# ==========================================

csv_path = os.path.join(FOLDER, "stage15_report.csv")

with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Peak lag (samples)",
        "Delay (ms)",
        "Apparent depth (m)",
        "Sample rate (samples/s)"
    ])
    writer.writerow([
        peak_index,
        delay_ms,
        apparent_depth,
        FS
    ])

print(f"\nA-scan saved: {ascan_path}")
print(f"CSV report saved: {csv_path}")
