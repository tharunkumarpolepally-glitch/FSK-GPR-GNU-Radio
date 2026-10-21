
import numpy as np
from scipy.signal import correlate, correlation_lags

FS = 500_000
EXPECTED_DELAY = 250

x = np.fromfile("reference_stage53.dat", dtype=np.complex64)
y = np.fromfile("received_stage53.dat", dtype=np.complex64)

if len(x) != len(y) or len(x) == 0:
    raise ValueError("Reference and received files must have equal, nonzero lengths.")

# Remove the estimated direct component.
direct_gain = np.vdot(x, y) / np.vdot(x, x)
residual = y - direct_gain * x

rng = np.random.default_rng(57)
residual_rms = np.sqrt(np.mean(np.abs(residual)**2))

noise_levels = [0.0, 0.01, 0.05, 0.10]

print("=" * 58)
print("STAGE 57: SYNTHETIC NOISE ROBUSTNESS TEST")
print("=" * 58)
print(f"Expected delay: {EXPECTED_DELAY} samples")
print(f"Residual RMS: {residual_rms:.6f}")
print("\nNoise RMS | Estimated delay | Error (samples)")

for level in noise_levels:
    noise = (
        rng.normal(size=len(residual))
        + 1j * rng.normal(size=len(residual))
    ) * (level / np.sqrt(2))

    test_signal = residual + noise.astype(np.complex64)

    corr = correlate(
        test_signal, x, mode="full", method="fft"
    )
    lags = correlation_lags(
        len(test_signal), len(x), mode="full"
    )

    mask = (lags >= 1) & (lags <= 2000)
    valid_lags = lags[mask]
    values = np.abs(corr[mask])

    estimated = int(valid_lags[np.argmax(values)])
    error = estimated - EXPECTED_DELAY

    print(f"{level:9.3f} | {estimated:15d} | {error:14d}")

print("\nNote: Noise levels are synthetic complex-sample RMS values.")
