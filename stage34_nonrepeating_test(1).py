
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

FS = 500_000
EXPECTED_DELAY = 250
COUNT = 100_000

REFERENCE_FILE = "reference.dat"
RECEIVED_FILE = "received.dat"

print("===== STAGE 34: NON-REPEATING CAPTURE TEST =====")

for filename in (REFERENCE_FILE, RECEIVED_FILE):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing file: {filename}")

    print(f"{filename}: {os.path.getsize(filename):,} bytes")

reference = np.fromfile(
    REFERENCE_FILE, dtype=np.complex64, count=COUNT
)
received = np.fromfile(
    RECEIVED_FILE, dtype=np.complex64, count=COUNT
)

n = min(len(reference), len(received))
reference = reference[:n]
received = received[:n]

if n <= EXPECTED_DELAY + 1000:
    raise ValueError("Not enough aligned samples for this test.")

print(f"\nAligned samples analyzed: {n:,}")

# Estimate direct-path scale using the initial samples.
direct_count = min(200, n)
denominator = np.vdot(reference[:direct_count],
                      reference[:direct_count])

if abs(denominator) < 1e-12:
    raise ValueError("Reference signal is too small.")

scale = np.vdot(reference[:direct_count],
                received[:direct_count]) / denominator

residual = received - scale * reference

# Inspect the residual before and after the configured delay.
before = np.mean(np.abs(residual[:EXPECTED_DELAY]))
after_end = min(n, EXPECTED_DELAY + 5000)
after = np.mean(np.abs(residual[EXPECTED_DELAY:after_end]))

print(f"Estimated direct scale: {scale}")
print(f"Residual before delay: {before:.6f}")
print(f"Residual after delay:  {after:.6f}")

# Compare reference with residual at candidate lags.
max_lag = min(2000, n // 4)
lags = np.arange(0, max_lag + 1)
corr_values = []

for lag in lags:
    count = min(5000, n - lag)
    if count <= 0:
        corr_values.append(0.0)
        continue

    a = residual[lag:lag + count]
    b = reference[:count]

    value = abs(np.vdot(b, a))
    corr_values.append(value)

corr_values = np.asarray(corr_values, dtype=float)

expected_peak = corr_values[EXPECTED_DELAY]
secondary_lag = 1250 if 1250 <= max_lag else None
secondary_peak = (
    corr_values[secondary_lag]
    if secondary_lag is not None else np.nan
)

best_lag = int(lags[np.argmax(corr_values)])

print("\nCorrelation results:")
print(f"Expected delay: {EXPECTED_DELAY} samples "
      f"({1000 * EXPECTED_DELAY / FS:.3f} ms)")
print(f"Strongest candidate lag: {best_lag} samples "
      f"({1000 * best_lag / FS:.3f} ms)")
print(f"Correlation at expected delay: {expected_peak:.3f}")

if secondary_lag is not None:
    ratio = secondary_peak / max(expected_peak, 1e-12)
    print(f"Correlation at {secondary_lag}: {secondary_peak:.3f}")
    print(f"Secondary/expected ratio: {ratio:.4f}")

# Plot correlation candidates.
plt.figure(figsize=(12, 5))
plt.plot(lags, corr_values, linewidth=1)
plt.axvline(EXPECTED_DELAY, linestyle="--",
            label="Expected delay: 250")
if secondary_lag is not None:
    plt.axvline(secondary_lag, linestyle=":",
                label="Previous secondary candidate: 1250")
plt.xlabel("Lag (samples)")
plt.ylabel("Correlation magnitude")
plt.title("Stage 34 — Correlation of Residual and Reference")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("stage34_correlation.png", dpi=150)
plt.show()

print("\nSaved: stage34_correlation.png")
print("NOTE: Correlation alone cannot prove a physical radar echo.")
print("This script cannot prove Repeat=No; verify the GRC setting.")
