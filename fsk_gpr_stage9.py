
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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
minimum_peak_distance = 20
minimum_prominence = 0.025

targets = [
    (1.5, 0.15, 0.40),
    (3.5, 0.30, 0.30)
]

output_dir = Path(__file__).resolve().parent
rng = np.random.default_rng(12)

# =====================================================
# 2. GENERATE FSK TRANSMITTED REFERENCE
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
# 3. SIMULATE RECEIVED TRACES
# =====================================================
positions = np.linspace(0, scan_width, num_positions)

geometry_delays = []

for xpos in positions:
    for target_x, depth, amplitude in targets:
        distance = np.sqrt(
            depth**2 + (xpos - target_x)**2
        )
        delay = int(round(2 * distance / velocity * fs))
        geometry_delays.append(delay)

trace_length = max(geometry_delays) + len(tx) + 100

received = np.zeros(
    (num_positions, trace_length),
    dtype=complex
)

for j, xpos in enumerate(positions):
    for target_x, depth, amplitude in targets:
        distance = np.sqrt(
            depth**2 + (xpos - target_x)**2
        )
        delay = int(round(2 * distance / velocity * fs))

        received[j, delay:delay + len(tx)] += amplitude * tx

    noise = noise_level / np.sqrt(2) * (
        rng.standard_normal(trace_length)
        + 1j * rng.standard_normal(trace_length)
    )
    received[j] += noise

# =====================================================
# 4. MATCHED-FILTER PROCESSING
# =====================================================
response_matrix = np.zeros(
    (trace_length - len(tx) + 1, num_positions)
)

for j in range(num_positions):
    corr = np.correlate(received[j], tx, mode="valid")
    response_matrix[:, j] = np.abs(corr) / energy

# =====================================================
# 5. STAGE 9: FILTER AND GROUP PEAKS
# =====================================================
detections = []

for j, xpos in enumerate(positions):
    response = response_matrix[:, j]

    peaks, properties = find_peaks(
        response,
        height=threshold,
        distance=minimum_peak_distance,
        prominence=minimum_prominence
    )

    for peak in peaks:
        delay_ns = peak / fs * 1e9
        apparent_depth = peak / fs * velocity / 2

        detections.append({
            "scan_position_m": float(xpos),
            "delay_ns": float(delay_ns),
            "apparent_depth_m": float(apparent_depth),
            "response": float(response[peak])
        })

# =====================================================
# 6. A-SCAN: CENTRE OF THE SCAN
# =====================================================


# =====================================================
# 6. A-SCANS AT BOTH TARGET POSITIONS
# =====================================================
target_scan_positions = [1.5, 3.5]

plt.figure(figsize=(10, 5))

for target_x in target_scan_positions:
    j = int(np.argmin(np.abs(positions - target_x)))

    response = response_matrix[:, j]
    time_ns = np.arange(len(response)) / fs * 1e9

    peaks, _ = find_peaks(
        response,
        height=threshold,
        distance=minimum_peak_distance,
        prominence=minimum_prominence
    )

    plt.plot(
        time_ns,
        response,
        label=f"Scan position {positions[j]:.2f} m"
    )

    plt.scatter(
        time_ns[peaks],
        response[peaks],
        s=20,
        label=f"Detected peaks at {positions[j]:.2f} m"
    )

plt.axhline(
    threshold,
    color="red",
    linestyle="--",
    label="Detection threshold"
)

plt.xlim(0, 15)
plt.xlabel("Two-way delay (ns)")
plt.ylabel("Normalized correlation")
plt.title("Stage 9: A-scans at Both Target Positions")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "stage9_ascan.png", dpi=200)


# =====================================================
# 7. B-SCAN: SHOW FILTERED RESPONSES
# =====================================================
display_max_ns = 80
display_samples = min(
    response_matrix.shape[0],
    int(display_max_ns * 1e-9 * fs) + 1
)

bscan = response_matrix[:display_samples, :].copy()
bscan[bscan < threshold] = 0

plt.figure(figsize=(11, 6))
plt.imshow(
    bscan,
    extent=[
        positions[0],
        positions[-1],
        (display_samples - 1) / fs * 1e9,
        0
    ],
    aspect="auto",
    interpolation="nearest",
    cmap="turbo",
    vmin=0,
    vmax=max(0.4, float(bscan.max()))
)

# Overlay retained detections
if detections:
    det_x = np.array([
        d["scan_position_m"] for d in detections
    ])
    det_t = np.array([
        d["delay_ns"] for d in detections
    ])

    plt.scatter(
        det_x, det_t,
        s=5, facecolors="none",
        edgecolors="white", linewidths=0.4,
        label="Retained detections"
    )
    plt.legend(loc="upper right")

plt.xlabel("Scan position (m)")
plt.ylabel("Two-way delay (ns)")
plt.title("Stage 9: Filtered FSK-GPR B-scan")
plt.colorbar(label="Normalized matched-filter response")
plt.tight_layout()
plt.savefig(output_dir / "stage9_bscan.png", dpi=200)
plt.show()

# =====================================================
# 8. SAVE DETECTION REPORT
# =====================================================
csv_path = output_dir / "stage9_detections.csv"

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

print("\n========== STAGE 9 RESULTS ==========")
print(f"Scan positions: {num_positions}")
print(f"Retained peak detections: {len(detections)}")
print(f"Detection threshold: {threshold:.3f}")
print(f"Minimum peak separation: {minimum_peak_distance} samples")
print(f"Minimum prominence: {minimum_prominence:.3f}")
print(f"A-scan: {output_dir / 'stage9_ascan.png'}")
print(f"B-scan: {output_dir / 'stage9_bscan.png'}")
print(f"CSV report: {csv_path}")

print("\nStrongest retained detections:")
for d in sorted(
    detections,
    key=lambda item: item["response"],
    reverse=True
)[:10]:
    print(
        f"Position={d['scan_position_m']:.2f} m, "
        f"delay={d['delay_ns']:.2f} ns, "
        f"apparent depth={d['apparent_depth_m']:.3f} m, "
        f"response={d['response']:.3f}"
    )

print("\nSynthetic simulation only.")
print("Thresholding reduces weak peaks but does not eliminate")
print("all sidelobes or prove that every detection is a real target.")
