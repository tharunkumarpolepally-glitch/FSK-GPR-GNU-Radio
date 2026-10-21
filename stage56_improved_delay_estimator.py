
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks

FS = 500_000
EXPECTED_DELAY = 250
REF_FILE = "reference_stage53.dat"
RX_FILE = "received_stage53.dat"

x = np.fromfile(REF_FILE, dtype=np.complex64)
y = np.fromfile(RX_FILE, dtype=np.complex64)

if len(x) != len(y) or len(x) == 0:
    raise ValueError("Reference and received files must have equal, nonzero lengths.")

# Estimate and remove the direct component.
direct_gain = np.vdot(x, y) / np.vdot(x, x)
residual = y - direct_gain * x

# Correlate residual with the reference.
corr = correlate(residual, x, mode="full", method="fft")
lags = np.arange(-len(x) + 1, len(x))

# Search positive delays only, up to 2000 samples.
mask = (lags >= 1) & (lags <= 2000)
positive_lags = lags[mask]
positive_corr = np.abs(corr[mask])

# Normalize by the reference and residual energy.
norm = np.sqrt(np.sum(np.abs(x)**2) *
               np.sum(np.abs(residual)**2))
normalized = positive_corr / norm if norm else positive_corr

# Find prominent local peaks.
peaks, properties = find_peaks(normalized, distance=5)
order = np.argsort(normalized[peaks])[::-1]
top = peaks[order[:10]]

print("=" * 58)
print("STAGE 56: IMPROVED DELAY ESTIMATION")
print("=" * 58)
print(f"Samples: {len(x):,}")
print(f"Estimated direct gain: {direct_gain}")
print(f"Expected delay: {EXPECTED_DELAY} samples")
print("\nTop correlation peaks:")

for rank, index in enumerate(top, start=1):
    lag = int(positive_lags[index])
    time_ms = lag / FS * 1000
    print(
        f"{rank:2d}. Lag={lag:4d} samples, "
        f"time={time_ms:.6f} ms, "
        f"normalized correlation={normalized[index]:.6f}"
    )

best_index = int(np.argmax(normalized))
best_lag = int(positive_lags[best_index])

print(f"\nStrongest searched peak: {best_lag} samples")
print(f"Estimated delay: {best_lag / FS * 1000:.6f} ms")
print(f"Delay error: {best_lag - EXPECTED_DELAY} samples")

plt.figure(figsize=(12, 5))
plt.plot(positive_lags, normalized, linewidth=0.8)
plt.axvline(EXPECTED_DELAY, linestyle="--",
            label="Expected delay: 250 samples")
plt.axvline(best_lag, linestyle=":",
            label=f"Strongest peak: {best_lag} samples")
plt.xlabel("Positive lag (samples)")
plt.ylabel("Normalized correlation")
plt.title("Stage 56: Delay Estimation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("stage56_improved_delay.png", dpi=150)
plt.show()

print("\nSaved: stage56_improved_delay.png")
