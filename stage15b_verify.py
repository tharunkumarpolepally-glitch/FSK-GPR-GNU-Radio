
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

FS = 500_000
EXPECTED_DELAY = 250
COUNT = 100_000

reference = np.fromfile(
    "reference.dat", dtype=np.complex64, count=COUNT
)
received = np.fromfile(
    "received.dat", dtype=np.complex64, count=COUNT
)

n = min(len(reference), len(received))
reference = reference[:n]
received = received[:n]

# The received file should contain the direct signal
# plus a delayed, attenuated copy.
# Remove the direct path to expose the echo.
echo_residual = received - reference

# Correlate the residual against the reference.
corr = correlate(
    echo_residual, reference,
    mode="full", method="fft"
)
lags = correlation_lags(
    len(echo_residual), len(reference), mode="full"
)

# Examine positive lags from 0 to 1000 samples.
mask = (lags >= 0) & (lags <= 1000)
positive_lags = lags[mask]
positive_corr = np.abs(corr[mask])

peak_index = int(np.argmax(positive_corr))
estimated_lag = int(positive_lags[peak_index])

print("===== DELAY VERIFICATION =====")
print(f"Expected delay: {EXPECTED_DELAY} samples")
print(f"Estimated delay: {estimated_lag} samples")
print(f"Expected time: {EXPECTED_DELAY / FS * 1000:.3f} ms")
print(f"Estimated time: {estimated_lag / FS * 1000:.3f} ms")
print(f"Peak correlation: {positive_corr[peak_index]:.3f}")

plt.figure(figsize=(10, 4))
plt.plot(positive_lags, positive_corr)
plt.axvline(
    EXPECTED_DELAY, color="red", linestyle="--",
    label="Expected: 250 samples"
)
plt.axvline(
    estimated_lag, color="green", linestyle=":",
    label=f"Estimated: {estimated_lag} samples"
)
plt.xlabel("Lag (samples)")
plt.ylabel("Correlation magnitude")
plt.title("Reference-to-Echo Delay Verification")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("stage15b_delay.png", dpi=150)
plt.show()
