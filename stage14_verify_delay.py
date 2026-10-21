
import numpy as np
from pathlib import Path
from scipy.signal import correlate, correlation_lags, find_peaks
import matplotlib.pyplot as plt

folder = Path(r"C:\Users\tharunkumar\Desktop\FSK_GPR")
fs = 500_000
expected_delay = 250
count = 100_000

# Read only the first 100,000 complex samples
x = np.fromfile(
    folder / "reference.dat",
    dtype=np.complex64,
    count=count
)
y = np.fromfile(
    folder / "received.dat",
    dtype=np.complex64,
    count=count
)

n = min(len(x), len(y))
x = x[:n]
y = y[:n]

if n < 1000:
    raise ValueError("Not enough samples in the capture.")

# Remove DC offsets
x = x - np.mean(x)
y = y - np.mean(y)

# The received signal should equal:
# direct signal + delayed signal
residual = y - x

# Compare the residual with the reference
corr = correlate(
    residual, x, mode="full", method="fft"
)
lags = correlation_lags(
    len(residual), len(x), mode="full"
)

# Inspect only delays near zero to +1000 samples
mask = (lags >= 0) & (lags <= 1000)
lags = lags[mask]
corr = np.abs(corr[mask])

corr /= np.max(corr) + 1e-12

peaks, _ = find_peaks(corr, distance=10)
strongest = peaks[np.argsort(corr[peaks])[-10:][::-1]]

print("===== STAGE 14: DELAY VERIFICATION =====")
print(f"Samples compared: {n}")
print(f"Expected delay: {expected_delay} samples")
print(f"Expected time delay: {expected_delay/fs*1000:.3f} ms")
print("\nStrongest correlation peaks:")

for p in strongest:
    lag = int(lags[p])
    print(
        f"Lag: {lag:4d} samples | "
        f"Delay: {lag/fs*1000:.3f} ms | "
        f"Response: {corr[p]:.4f}"
    )

plt.figure(figsize=(10, 4))
plt.plot(lags, corr)
plt.axvline(
    expected_delay,
    color="red",
    linestyle="--",
    label="Expected delay: 250 samples"
)
plt.xlabel("Delay (samples)")
plt.ylabel("Normalized correlation")
plt.title("Stage 14: Delayed Echo Verification")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(folder / "stage14_delay_check.png", dpi=150)
plt.show()
