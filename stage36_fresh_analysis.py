
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags, find_peaks

# Stage 36: analyze the fresh GNU Radio capture
FOLDER = r"C:\Users\tharunkumar\Desktop\FSK_GPR"
FS = 500_000
EXPECTED_DELAY = 250
COUNT = 100_000

REF_FILE = os.path.join(FOLDER, "reference_stage35.dat")
RX_FILE = os.path.join(FOLDER, "received_stage35.dat")

for path in (REF_FILE, RX_FILE):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")

reference = np.fromfile(REF_FILE, dtype=np.complex64)
received = np.fromfile(RX_FILE, dtype=np.complex64)

n = min(len(reference), len(received), COUNT)
reference = reference[:n]
received = received[:n]

if n <= EXPECTED_DELAY + 1000:
    raise ValueError("Not enough samples for delay analysis.")

print("===== STAGE 36: FRESH CAPTURE ANALYSIS =====")
print(f"Reference samples: {len(reference):,}")
print(f"Received samples:  {len(received):,}")
print(f"Sample rate: {FS:,} samples/s")
print(f"Expected delay: {EXPECTED_DELAY} samples")
print(f"Expected delay time: {1000*EXPECTED_DELAY/FS:.6f} ms")

print("\nSignal checks:")
print(f"Reference mean magnitude: {np.mean(np.abs(reference)):.6f}")
print(f"Received mean magnitude:  {np.mean(np.abs(received)):.6f}")
print(f"Reference finite: {np.all(np.isfinite(reference))}")
print(f"Received finite:  {np.all(np.isfinite(received))}")

# Estimate the direct-path scale from the initial samples.
# The delayed branch is assumed to be zero during the first 250 samples.
initial_ref = reference[:EXPECTED_DELAY]
initial_rx = received[:EXPECTED_DELAY]

denom = np.vdot(initial_ref, initial_ref)
if abs(denom) < 1e-12:
    raise ValueError("Reference signal is too small to estimate direct path.")

scale = np.vdot(initial_ref, initial_rx) / denom
residual = received - scale * reference

before = np.mean(np.abs(residual[:EXPECTED_DELAY]))
after_end = min(n, EXPECTED_DELAY + 1000)
after = np.mean(np.abs(residual[EXPECTED_DELAY:after_end]))

print("\nDirect-path check:")
print(f"Estimated scale: {scale.real:.6f} {scale.imag:+.6f}j")
print(f"Residual before delay: {before:.6f}")
print(f"Residual after delay:  {after:.6f}")

# Correlate the residual with the reference over positive lags.
# This is a diagnostic, not proof of a physical radar target.
max_lag = min(2000, n // 4)
corr = correlate(
    residual,
    reference,
    mode="full",
    method="fft"
)
lags = correlation_lags(n, n, mode="full")

mask = (lags >= 0) & (lags <= max_lag)
lag_values = lags[mask]
corr_values = np.abs(corr[mask])

# Normalize by the reference energy for each lag.
# Use only a fixed-length overlap for a fair comparison.
norm_corr = np.zeros_like(corr_values)
for i, lag in enumerate(lag_values):
    a = residual[lag:]
    b = reference[:len(a)]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    norm_corr[i] = abs(np.vdot(b, a)) / denom if denom > 0 else 0

peaks, _ = find_peaks(norm_corr, distance=20)
top = sorted(peaks, key=lambda i: norm_corr[i], reverse=True)[:10]

print("\nTop normalized correlation candidates:")
print("Rank | Lag | Time (ms) | Normalized correlation")
for rank, i in enumerate(top, 1):
    lag = int(lag_values[i])
    time_ms = 1000 * lag / FS
    print(f"{rank:>4} | {lag:>4} | {time_ms:>9.6f} | {norm_corr[i]:.6f}")

# Plot a short section of both signals and the correlation.
t_ms = np.arange(n) / FS * 1000
show = min(n, 2000)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(t_ms[:show], np.abs(reference[:show]), label="Reference")
axes[0].plot(t_ms[:show], np.abs(received[:show]), label="Received", alpha=0.8)
axes[0].set_title("Stage 36 — Fresh Reference and Received Signals")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Magnitude")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(t_ms[:show], np.abs(residual[:show]))
axes[1].axvline(1000 * EXPECTED_DELAY / FS, linestyle="--",
                label="Expected delay")
axes[1].set_title("Residual after direct-path subtraction")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Magnitude")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(lag_values, norm_corr)
axes[2].axvline(EXPECTED_DELAY, color="red", linestyle="--",
                label="Expected lag")
axes[2].set_title("Normalized residual-to-reference correlation")
axes[2].set_xlabel("Lag (samples)")
axes[2].set_ylabel("Normalized correlation")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
out = os.path.join(FOLDER, "stage36_fresh_analysis.png")
plt.savefig(out, dpi=150)
print(f"\nSaved plot: {out}")
plt.show()
