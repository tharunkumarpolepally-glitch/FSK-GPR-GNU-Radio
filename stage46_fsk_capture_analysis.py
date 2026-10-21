
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv

# Stage 46: FSK capture analysis
FOLDER = Path(r"C:\Users\tharunkumar\Desktop\FSK_GPR")
FS = 500_000
EXPECTED_DELAY = 250

reference_file = FOLDER / "reference_stage45.dat"
received_file = FOLDER / "received_stage45.dat"
samples_file = FOLDER / "gr_samples_stage45.dat"

def load_complex(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw = np.fromfile(path, dtype=np.complex64)

    if raw.size == 0:
        raise ValueError(f"File is empty: {path}")

    return raw

reference = load_complex(reference_file)
received = load_complex(received_file)
gr_samples = load_complex(samples_file)

count = min(len(reference), len(received), len(gr_samples))
reference = reference[:count]
received = received[:count]
gr_samples = gr_samples[:count]

# Check file consistency
file_difference = np.max(np.abs(received - gr_samples))

# Estimate the direct-path scale using the first samples.
# This is a software model, not a physical radar calibration.
calibration_count = min(EXPECTED_DELAY, count)
denominator = np.vdot(
    reference[:calibration_count],
    reference[:calibration_count]
)

if abs(denominator) > 1e-12:
    direct_scale = np.vdot(
        reference[:calibration_count],
        received[:calibration_count]
    ) / denominator
else:
    direct_scale = 0j

residual = received - direct_scale * reference

# Estimate delay using normalized correlation.
max_lag = min(2000, count // 2)
template_length = min(20_000, count - max_lag)

if template_length <= 0:
    raise ValueError("Not enough samples for delay analysis.")

template = reference[:template_length]
template_norm = np.linalg.norm(template)

scores = []
lags = range(max_lag + 1)

for lag in lags:
    segment = residual[lag:lag + template_length]

    if len(segment) != template_length:
        scores.append(0.0)
        continue

    norm = np.linalg.norm(segment) * template_norm

    if norm < 1e-12:
        scores.append(0.0)
    else:
        scores.append(abs(np.vdot(template, segment)) / norm)

estimated_delay = int(np.argmax(scores))
delay_ms = estimated_delay / FS * 1000

# Estimate instantaneous frequency from phase differences.
phase_step = np.angle(received[1:] * np.conj(received[:-1]))
instantaneous_frequency = phase_step * FS / (2 * np.pi)

# Remove extreme phase-difference samples for summary statistics.
valid_frequency = instantaneous_frequency[
    np.isfinite(instantaneous_frequency)
]

print("=" * 60)
print("STAGE 46: FSK CAPTURE ANALYSIS")
print("=" * 60)
print(f"Sample count: {count:,}")
print(f"Sample rate: {FS:,} samples/s")
print(f"Capture duration: {count / FS:.6f} s")
print(f"Reference file: {reference_file.stat().st_size:,} bytes")
print(f"Received file:  {received_file.stat().st_size:,} bytes")
print(f"gr_samples match received (max difference): {file_difference:.6g}")
print(f"Estimated direct-path scale: {direct_scale}")
print(f"Expected delay: {EXPECTED_DELAY} samples")
print(f"Estimated correlation delay: {estimated_delay} samples")
print(f"Estimated delay time: {delay_ms:.6f} ms")
print(f"Correlation score: {scores[estimated_delay]:.6f}")
print()
print("Instantaneous-frequency statistics (received signal):")
print(f"Mean: {np.mean(valid_frequency):.2f} Hz")
print(f"Standard deviation: {np.std(valid_frequency):.2f} Hz")
print(f"Minimum: {np.min(valid_frequency):.2f} Hz")
print(f"Maximum: {np.max(valid_frequency):.2f} Hz")
print()
print("Note: This is a digital simulation, not measured GPR data.")

# Save summary CSV
csv_path = FOLDER / "stage46_fsk_capture_summary.csv"

with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Sample count", count])
    writer.writerow(["Sample rate Hz", FS])
    writer.writerow(["Capture duration s", count / FS])
    writer.writerow(["File match max difference", file_difference])
    writer.writerow(["Expected delay samples", EXPECTED_DELAY])
    writer.writerow(["Estimated delay samples", estimated_delay])
    writer.writerow(["Estimated delay ms", delay_ms])
    writer.writerow(["Correlation score", scores[estimated_delay]])
    writer.writerow(["Mean instantaneous frequency Hz",
                     np.mean(valid_frequency)])
    writer.writerow(["Frequency standard deviation Hz",
                     np.std(valid_frequency)])
    writer.writerow(["Minimum instantaneous frequency Hz",
                     np.min(valid_frequency)])
    writer.writerow(["Maximum instantaneous frequency Hz",
                     np.max(valid_frequency)])

# Plot a short section of the received waveform and frequency estimate.
plot_count = min(1000, count)
time_ms = np.arange(plot_count) / FS * 1000

fig, axes = plt.subplots(2, 1, figsize=(11, 7))

axes[0].plot(time_ms, received[:plot_count].real)
axes[0].set_title("Stage 46: Received Signal — Real Part")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

freq_count = min(1000, len(valid_frequency))
freq_time_ms = np.arange(freq_count) / FS * 1000

axes[1].plot(freq_time_ms, valid_frequency[:freq_count] / 1000)
axes[1].set_title("Instantaneous Frequency Estimate")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Frequency (kHz)")
axes[1].grid(True)

plt.tight_layout()

plot_path = FOLDER / "stage46_fsk_capture_analysis.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print(f"\nSaved summary: {csv_path}")
print(f"Saved plot:    {plot_path}")
