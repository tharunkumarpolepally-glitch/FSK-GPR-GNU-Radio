
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

FOLDER = r"C:\Users\tharunkumar\Desktop\FSK_GPR"
FS = 500_000
COUNT = 100_000
DELAY = 250
NOISE_LEVELS = [0.001, 0.01, 0.05, 0.10]
SEED = 43

ref_path = os.path.join(FOLDER, "reference_stage35.dat")
rx_path = os.path.join(FOLDER, "received_stage35.dat")

reference = np.fromfile(ref_path, dtype=np.complex64, count=COUNT)
received = np.fromfile(rx_path, dtype=np.complex64, count=COUNT)

n = min(len(reference), len(received))
reference = reference[:n]
received = received[:n]

if n <= DELAY + 2000:
    raise ValueError("Not enough samples for the test.")

# Estimate and remove the direct path using the initial samples.
direct_ref = reference[:DELAY]
direct_rx = received[:DELAY]
denom = np.vdot(direct_ref, direct_ref)

if abs(denom) < 1e-12:
    raise ValueError("Reference signal energy is too small.")

scale = np.vdot(direct_ref, direct_rx) / denom
residual = received - scale * reference

# Use a fixed-length reference template and positive lag search.
template_len = min(20_000, n - 2000)
template = reference[:template_len]
rng = np.random.default_rng(SEED)

results = []

print("=" * 60)
print("STAGE 43: CONTROLLED NOISE ECHO TEST")
print("=" * 60)
print(f"Configured delay: {DELAY} samples ({DELAY / FS * 1000:.3f} ms)")
print("Noise levels are complex-noise standard deviations.")

for noise_std in NOISE_LEVELS:
    noise = (
        rng.normal(0, noise_std / np.sqrt(2), n)
        + 1j * rng.normal(0, noise_std / np.sqrt(2), n)
    ).astype(np.complex64)

    noisy_received = received + noise
    noisy_residual = noisy_received - scale * reference
    segment = noisy_residual[:template_len + 2000]

    corr = correlate(
        segment, template, mode="full", method="fft"
    )
    lags = correlation_lags(
        len(segment), len(template), mode="full"
    )

    mask = (lags >= 0) & (lags <= 2000)
    candidate_lags = lags[mask]
    candidate_corr = np.abs(corr[mask])

    idx = int(np.argmax(candidate_corr))
    estimated_delay = int(candidate_lags[idx])
    peak = float(candidate_corr[idx])

    print(
        f"Noise std={noise_std:.3f} | "
        f"Estimated delay={estimated_delay:4d} samples | "
        f"Error={estimated_delay - DELAY:+d} samples | "
        f"Correlation={peak:.2f}"
    )

    results.append([
        noise_std, DELAY, estimated_delay,
        estimated_delay - DELAY, peak
    ])

csv_path = os.path.join(FOLDER, "stage43_noise_echo_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "noise_std", "expected_delay", "estimated_delay",
        "delay_error_samples", "correlation_peak"
    ])
    writer.writerows(results)

noise_values = [r[0] for r in results]
delay_errors = [r[3] for r in results]

plt.figure(figsize=(9, 5))
plt.plot(noise_values, delay_errors, marker="o")
plt.axhline(0, linestyle="--")
plt.xlabel("Complex noise standard deviation")
plt.ylabel("Delay error (samples)")
plt.title("Stage 43 — Delay Error Under Controlled Noise")
plt.grid(True)
png_path = os.path.join(FOLDER, "stage43_noise_echo_test.png")
plt.tight_layout()
plt.savefig(png_path, dpi=150)
plt.close()

print("\nSaved:")
print(csv_path)
print(png_path)
print("Synthetic-noise test only; not measured GPR performance.")
