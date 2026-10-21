
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

FOLDER = r"C:\Users\tharunkumar\Desktop\FSK_GPR"
FS = 500_000
COUNT = 100_000
DELAY = 250

REF_FILE = os.path.join(FOLDER, "reference_stage35.dat")
RX_FILE = os.path.join(FOLDER, "received_stage35.dat")

for path in (REF_FILE, RX_FILE):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")

reference = np.fromfile(
    REF_FILE, dtype=np.complex64, count=COUNT
)
received = np.fromfile(
    RX_FILE, dtype=np.complex64, count=COUNT
)

n = min(len(reference), len(received))
reference = reference[:n]
received = received[:n]

if n <= DELAY + 1000:
    raise ValueError("Not enough samples for analysis.")

# Estimate direct-path scale using the first DELAY samples.
ref_direct = reference[:DELAY]
rx_direct = received[:DELAY]

energy = np.vdot(ref_direct, ref_direct)
if abs(energy) < 1e-12:
    raise ValueError("Reference segment has insufficient energy.")

scale = np.vdot(ref_direct, rx_direct) / energy
residual = received - scale * reference

# Compare the known delayed segment with the residual.
echo = 0.4 * reference[:n - DELAY]
observed = residual[DELAY:]

m = min(len(echo), len(observed))
echo = echo[:m]
observed = observed[:m]

# Least-squares estimate of the residual's relationship to
# the delayed reference. This is valid for the constructed model
# only when the stream alignment and delay are correct.
denominator = np.vdot(echo, echo)
echo_scale = (
    np.vdot(echo, observed) / denominator
    if abs(denominator) > 1e-12 else 0j
)
error = observed - echo_scale * echo

echo_rms = float(np.sqrt(np.mean(np.abs(observed) ** 2)))
error_rms = float(np.sqrt(np.mean(np.abs(error) ** 2)))
reference_rms = float(np.sqrt(np.mean(np.abs(reference) ** 2)))

if error_rms > 0:
    snr_db = 20 * np.log10(echo_rms / error_rms)
else:
    snr_db = float("inf")

print("=" * 55)
print("STAGE 42: ECHO AMPLITUDE AND ERROR ANALYSIS")
print("=" * 55)
print(f"Samples analyzed: {n:,}")
print(f"Sample rate: {FS:,} samples/s")
print(f"Configured delay: {DELAY} samples ({DELAY / FS * 1000:.6f} ms)")
print(f"Estimated direct scale: {scale}")
print(f"Reference RMS magnitude: {reference_rms:.6f}")
print(f"Residual RMS after delay: {echo_rms:.6f}")
print(f"Estimated delayed-component scale: {echo_scale}")
print(f"Model residual error RMS: {error_rms:.8f}")
print(f"Model-based SNR estimate: {snr_db:.2f} dB")
print("\nThese metrics describe the captured software signal path.")
print("They are not measured physical GPR sensitivity or SNR.")

# Visualize the residual around the programmed delay.
show_start = max(0, DELAY - 100)
show_end = min(n, DELAY + 1000)
idx = np.arange(show_start, show_end)
time_ms = idx / FS * 1000

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(
    time_ms, np.abs(received[show_start:show_end]),
    label="Received magnitude"
)
axes[0].plot(
    time_ms, np.abs(residual[show_start:show_end]),
    label="Residual magnitude"
)
axes[0].axvline(
    DELAY / FS * 1000,
    color="red", linestyle="--", label="Configured delay"
)
axes[0].set_title("Stage 42 — Received Signal and Residual")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Magnitude")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(
    np.abs(observed[:2000]), label="Observed residual"
)
axes[1].plot(
    np.abs((echo_scale * echo)[:2000]),
    label="Scaled delayed reference"
)
axes[1].set_title("Delayed-Component Model Comparison")
axes[1].set_xlabel("Sample index after delay")
axes[1].set_ylabel("Magnitude")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
png_path = os.path.join(
    FOLDER, "stage42_echo_amplitude_snr.png"
)
plt.savefig(png_path, dpi=150)
plt.close(fig)

csv_path = os.path.join(
    FOLDER, "stage42_echo_amplitude_snr.csv"
)
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["samples_analyzed", n])
    writer.writerow(["sample_rate_hz", FS])
    writer.writerow(["delay_samples", DELAY])
    writer.writerow(["delay_ms", DELAY / FS * 1000])
    writer.writerow(["reference_rms", reference_rms])
    writer.writerow(["residual_rms", echo_rms])
    writer.writerow(["delayed_component_scale_real", echo_scale.real])
    writer.writerow(["delayed_component_scale_imag", echo_scale.imag])
    writer.writerow(["model_error_rms", error_rms])
    writer.writerow(["model_based_snr_db", snr_db])

print("\nSaved:")
print(png_path)
print(csv_path)
