
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

FS = 500_000
EXPECTED_DELAY = 250
COUNT = 100_000
MAX_LAG = 2_000

reference = np.fromfile("reference.dat", dtype=np.complex64, count=COUNT)
received = np.fromfile("received.dat", dtype=np.complex64, count=COUNT)

n = min(len(reference), len(received))
reference = reference[:n]
received = received[:n]

# Estimate the echo contribution
echo = received - reference

# Use a limited segment to reduce processing time
N = min(20_000, n - EXPECTED_DELAY)
ref = reference[:N]
echo_segment = echo[:N]

# Cross-correlate the echo component with the reference
corr = correlate(echo_segment, ref, mode="full", method="fft")
lags = correlation_lags(len(echo_segment), len(ref), mode="full")

mask = (lags >= 0) & (lags <= MAX_LAG)
lags = lags[mask]
response = np.abs(corr[mask])

# Normalize for easier visualization
if response.max() > 0:
    response = response / response.max()

peak_index = np.argmax(response)
estimated_delay = int(lags[peak_index])
estimated_time_ms = estimated_delay / FS * 1000

print("===== STAGE 17: CORRELATION A-SCAN =====")
print(f"Expected delay: {EXPECTED_DELAY} samples")
print(f"Estimated delay: {estimated_delay} samples")
print(f"Estimated delay time: {estimated_time_ms:.6f} ms")

plt.figure(figsize=(11, 5))
plt.plot(lags, response, linewidth=1)
plt.axvline(
    EXPECTED_DELAY,
    color="red",
    linestyle="--",
    label=f"Expected: {EXPECTED_DELAY} samples"
)
plt.axvline(
    estimated_delay,
    color="green",
    linestyle=":",
    label=f"Estimated: {estimated_delay} samples"
)
plt.title("FSK-GPR A-scan — Correlation Response")
plt.xlabel("Delay (samples)")
plt.ylabel("Normalized correlation magnitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("stage17_correlation_ascan.png", dpi=150)
plt.show()

print("Saved: stage17_correlation_ascan.png")
