import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags, find_peaks

# Stage 16: FSK-GPR A-scan receiver (software delay demonstration)
FS = 500_000                 # GNU Radio sample rate: 500 kS/s
EXPECTED_DELAY = 250         # configured delay in samples
COUNT = 100_000              # read only a small part of each large file
SOIL_VELOCITY = 1.0e8        # assumed velocity (m/s); change for your model

REFERENCE_FILE = "reference.dat"
RECEIVED_FILE = "received.dat"

for path in (REFERENCE_FILE, RECEIVED_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find {path}. Put this script in the folder containing both .dat files."
        )

reference = np.fromfile(REFERENCE_FILE, dtype=np.complex64, count=COUNT)
received = np.fromfile(RECEIVED_FILE, dtype=np.complex64, count=COUNT)

n = min(len(reference), len(received))
if n <= EXPECTED_DELAY + 10:
    raise ValueError("Not enough samples were read. Check that the .dat files are not empty.")

reference = reference[:n]
received = received[:n]

# Residual approximates the echo contribution after subtracting the direct path.
residual = received - reference

# Use a shorter segment to keep the correlation calculation manageable.
segment_len = min(20_000, n - EXPECTED_DELAY)
ref_seg = reference[:segment_len]
res_seg = residual[:segment_len]

corr = correlate(res_seg, ref_seg, mode="full", method="fft")
lags = correlation_lags(len(res_seg), len(ref_seg), mode="full")

# Restrict to plausible non-negative delays for this demonstration.
mask = (lags >= 0) & (lags <= min(2_000, segment_len - 1))
positive_lags = lags[mask]
positive_corr = np.abs(corr[mask])

if len(positive_corr) == 0:
    raise RuntimeError("No valid correlation lags were available.")

peak_index = int(np.argmax(positive_corr))
estimated_lag = int(positive_lags[peak_index])
estimated_time_s = estimated_lag / FS
estimated_depth_m = SOIL_VELOCITY * estimated_time_s / 2

print("===== STAGE 16: A-SCAN RECEIVER =====")
print(f"Samples processed: {n:,}")
print(f"Sample rate: {FS:,} samples/s")
print(f"Expected delay: {EXPECTED_DELAY} samples")
print(f"Estimated delay: {estimated_lag} samples")
print(f"Estimated delay time: {estimated_time_s * 1e3:.6f} ms")
print(f"Depth from assumed velocity ({SOIL_VELOCITY:.2e} m/s): {estimated_depth_m:.3f} m")
print("NOTE: At 500 kS/s, this is a software delay demo, not a realistic shallow-GPR depth.")

# Plot received magnitude and residual magnitude over the first 5 ms or available data.
plot_count = min(n, int(0.005 * FS))
time_ms = np.arange(plot_count) / FS * 1e3

fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)

axes[0].plot(time_ms, np.abs(received[:plot_count]), linewidth=0.8)
axes[0].set_title("Received Signal — A-scan")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Magnitude")
axes[0].grid(True)

axes[1].plot(time_ms, np.abs(residual[:plot_count]), linewidth=0.8)
axes[1].axvline(estimated_time_s * 1e3, linestyle="--",
                label=f"Estimated delay: {estimated_lag} samples")
axes[1].set_title("Estimated Echo Component (received − reference)")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Magnitude")
axes[1].legend()
axes[1].grid(True)

fig.savefig("stage16_ascan_receiver.png", dpi=150)
plt.show()
print("Saved plot: stage16_ascan_receiver.png")
