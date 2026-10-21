
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

FS = 500_000
DELAY_EXPECTED = 250
REF_FILE = "reference_stage53.dat"
RX_FILE = "received_stage53.dat"

for filename in (REF_FILE, RX_FILE):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing file: {filename}")

ref = np.fromfile(REF_FILE, dtype=np.complex64)
rx = np.fromfile(RX_FILE, dtype=np.complex64)

if len(ref) != len(rx) or len(ref) == 0:
    raise ValueError("Reference and received files must have equal, nonzero lengths.")

print("=" * 58)
print("STAGE 54: SIMULATED ECHO DELAY ANALYSIS")
print("=" * 58)
print(f"Samples per file: {len(ref):,}")
print(f"Sample rate: {FS:,} samples/s")
print(f"Capture duration: {len(ref)/FS:.6f} seconds")
print(f"Reference RMS: {np.sqrt(np.mean(np.abs(ref)**2)):.6f}")
print(f"Received RMS:  {np.sqrt(np.mean(np.abs(rx)**2)):.6f}")

# Estimate the direct-path complex scale, then subtract it.
direct_scale = np.vdot(ref, rx) / np.vdot(ref, ref)
residual = rx - direct_scale * ref

print(f"\nEstimated direct-path scale: {direct_scale}")
print(f"Residual RMS after direct-path subtraction: "
      f"{np.sqrt(np.mean(np.abs(residual)**2)):.6f}")

# Correlate the residual with the reference.
# Search positive lags only, where the delayed echo is expected.
corr = correlate(residual, ref, mode="full", method="fft")
lags = correlation_lags(len(residual), len(ref), mode="full")

search = (lags >= 1) & (lags <= 2000)
positive_corr = np.abs(corr[search])
positive_lags = lags[search]

idx = np.argmax(positive_corr)
estimated_delay = int(positive_lags[idx])
estimated_time_ms = estimated_delay / FS * 1000

print(f"\nExpected delay: {DELAY_EXPECTED} samples")
print(f"Estimated positive delay: {estimated_delay} samples")
print(f"Estimated delay time: {estimated_time_ms:.6f} ms")

# Compare the received signal with a direct-plus-delayed model.
if estimated_delay < len(ref):
    delayed = np.zeros_like(ref)
    delayed[estimated_delay:] = ref[:-estimated_delay]
    scale_echo = np.vdot(delayed, residual) / np.vdot(delayed, delayed)
    model = direct_scale * ref + scale_echo * delayed
    model_error = rx - model

    print(f"Estimated delayed-path scale: {scale_echo}")
    print(f"Model residual RMS: "
          f"{np.sqrt(np.mean(np.abs(model_error)**2)):.8g}")

# Plot correlation strength around the estimated delay.
plt.figure(figsize=(11, 5))
plt.plot(positive_lags, positive_corr, linewidth=0.8)
plt.axvline(DELAY_EXPECTED, linestyle="--",
            label="Expected: 250 samples")
plt.axvline(estimated_delay, linestyle=":",
            label=f"Estimated: {estimated_delay} samples")
plt.xlabel("Positive lag (samples)")
plt.ylabel("Correlation magnitude")
plt.title("Stage 54: Residual-to-Reference Correlation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("stage54_echo_delay_correlation.png", dpi=150)
plt.show()

print("\nSaved: stage54_echo_delay_correlation.png")
