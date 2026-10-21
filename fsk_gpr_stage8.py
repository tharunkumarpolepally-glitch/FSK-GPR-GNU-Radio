
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path

# =====================================================
# 1. CONFIGURATION
# =====================================================
fs = 10e9
symbol_rate = 1e9
freq_dev = 1e9

num_symbols = 2
num_positions = 201
scan_width = 5.0
velocity = 1e8

noise_level = 0.015
threshold = 0.08

# Synthetic targets: (position_m, depth_m, amplitude)
targets = [
    (1.5, 0.15, 0.40),
    (3.5, 0.30, 0.30)
]

output_dir = Path(__file__).resolve().parent
rng = np.random.default_rng(12)

# =====================================================
# 2. GENERATE FSK REFERENCE
# =====================================================
samples_per_symbol = int(fs / symbol_rate)
bits = np.array([0, 1])
symbols = 2 * bits - 1
symbol_stream = np.repeat(symbols, samples_per_symbol)

phase = np.cumsum(
    2 * np.pi * freq_dev * symbol_stream / fs
)
tx = np.exp(1j * phase)

edge = min(3, len(tx) // 4)
window = np.ones(len(tx))
window[:edge] = np.linspace(0, 1, edge)
window[-edge:] = np.linspace(1, 0, edge)
tx *= window

energy = np.sum(np.abs(tx) ** 2)

# =====================================================
# 3. SIMULATE THE RECEIVED TRACES
# =====================================================
positions = np.linspace(0, scan_width, num_positions)

# Use target geometry only to size the receive buffer.
geometry_delays = []

for xpos in positions:
    for target_x, depth, amplitude in targets:
        distance = np.sqrt(depth**2 + (xpos - target_x)**2)
        geometry_delays.append(round(2 * distance / velocity * fs))

trace_length = int(max(geometry_delays)) + len(tx) + 100

received = np.zeros(
    (num_positions, trace_length), dtype=complex
)

for j, xpos in enumerate(positions):
    for target_x, depth, amplitude in targets:
        distance = np.sqrt(depth**2 + (xpos - target_x)**2)
        delay = int(round(2 * distance / velocity * fs))
        received[j, delay:delay + len(tx)] += amplitude * tx

    noise = noise_level / np.sqrt(2) * (
        rng.standard_normal(trace_length)
        + 1j * rng.standard_normal(trace_length)
    )
    received[j] += noise

# =====================================================
# 4. BLIND MATCHED-FILTER PROCESSING
# =====================================================
response_matrix = np.zeros(
    (trace_length - len(tx) + 1, num_positions)
)

for j in range(num_positions):
    corr = np.correlate(received[j], tx, mode="valid")
    response_matrix[:, j] = np.abs(corr) / energy

# Detection is based on the received signal, not true delays.
detections = []
min_peak_distance = 3  # samples

for j, xpos in enumerate(positions):
    response = response_matrix[:, j]

    peaks, properties = find_peaks(
        response,
        height=threshold,
        distance=min_peak_distance,
        prominence=threshold * 0.25
    )

    for peak in peaks:
        delay_s = peak / fs
        depth_m = delay_s * velocity / 2

        detections.append({
            "scan_position_m": float(xpos),
            "delay_ns": float(delay_s * 1e9),
            "apparent_depth_m": float(depth_m),
            "response": float(response[peak])
        })

# =====================================================
# 5. DISPLAY CENTRE-POSITION A-SCAN
# =====================================================
centre = int(np.argmin(np.abs(positions - 2.5)))
response = response_matrix[:, centre]
time_ns = np.arange(len(response)) / fs * 1e9

plt.figure(figsize=(10, 4))
plt.plot(time_ns, response, label="Matched-filter response")
plt.axhline(
    threshold, color="red", linestyle="--",
    label="Detection threshold"
)
plt.xlabel("Two-way delay (ns)")
plt.ylabel("Normalized correlation")
plt.title("Stage 8: A-scan and Detection Threshold")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "fsk_gpr_ascan.png", dpi=200)

# =====================================================
# 6. DISPLAY AND SAVE B-SCAN
# =====================================================
max_display_ns = 100
display_samples = min(
    response_matrix.shape[0],
    int(max_display_ns * 1e-9 * fs) + 1
)

bscan = response_matrix[:display_samples, :].copy()
bscan[bscan < threshold] = 0

plt.figure(figsize=(10, 5))
plt.imshow(
    bscan,
    extent=[
        positions[0], positions[-1],
        (display_samples - 1) / fs * 1e9, 0
    ],
    aspect="auto",
    interpolation="nearest",
    cmap="turbo"
)
plt.xlabel("Scan position (m)")
plt.ylabel("Two-way delay (ns)")
plt.title("Stage 8: Multi-Target FSK-GPR B-scan")
plt.colorbar(label="Normalized matched-filter response")
plt.tight_layout()
plt.savefig(output_dir / "fsk_gpr_bscan.png", dpi=200)
plt.show()

# =====================================================
# 7. EXPORT DETECTIONS TO CSV
# =====================================================
csv_path = output_dir / "fsk_gpr_detections.csv"

with open(csv_path, "w", newline="") as file:
    fieldnames = [
        "scan_position_m",
        "delay_ns",
        "apparent_depth_m",
        "response"
    ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(detections)

# =====================================================
# 8. PRINT SUMMARY
# =====================================================
print("\n========== STAGE 8 DETECTION REPORT ==========")
print(f"Scan positions: {num_positions}")
print(f"Detected peaks: {len(detections)}")
print(f"Detection threshold: {threshold:.3f}")
print(f"A-scan image: {output_dir / 'fsk_gpr_ascan.png'}")
print(f"B-scan image: {output_dir / 'fsk_gpr_bscan.png'}")
print(f"CSV report: {csv_path}")

print("\nStrongest detections:")
for item in sorted(
    detections, key=lambda d: d["response"], reverse=True
)[:10]:
    print(
        f"Position={item['scan_position_m']:.2f} m, "
        f"delay={item['delay_ns']:.2f} ns, "
        f"apparent depth={item['apparent_depth_m']:.3f} m, "
        f"response={item['response']:.3f}"
    )

print("\nNote: These are simulated peak detections.")
print("Sidelobes can create false detections.")
print("Apparent depth assumes the configured propagation velocity.")
