
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ======================================================
# 1. SIMULATION SETTINGS
# ======================================================
fs = 10e9                    # 10 GHz simulation sample rate
symbol_rate = 1e9            # 1 Gsymbol/s
freq_dev = 1e9               # FSK frequency deviation

samples_per_symbol = int(fs / symbol_rate)
num_symbols = 2

scan_width = 5.0
num_positions = 201
velocity = 1e8               # Assumed propagation speed (m/s)

noise_level = 0.015

# Two simulated point targets:
# (horizontal position in metres, depth in metres, amplitude)
targets = [
    (1.5, 0.15, 0.40),
    (3.5, 0.30, 0.30)
]

rng = np.random.default_rng(12)

# ======================================================
# 2. GENERATE A SHORT FSK REFERENCE BURST
# ======================================================
bits = np.array([0, 1])
symbols = 2 * bits - 1

symbol_stream = np.repeat(symbols, samples_per_symbol)

phase = np.cumsum(
    2 * np.pi * freq_dev * symbol_stream / fs
)

tx = np.exp(1j * phase)

# Smooth burst edges
edge = 3
window = np.ones(len(tx))
window[:edge] = np.linspace(0, 1, edge)
window[-edge:] = np.linspace(1, 0, edge)
tx *= window

energy = np.sum(np.abs(tx)**2)

# ======================================================
# 3. SCAN POSITIONS AND ECHO MODEL
# ======================================================
positions = np.linspace(0, scan_width, num_positions)

# Compute delays for sizing only.
# These delays are not used by the detector.
all_true_delays = []

for xpos in positions:
    row = []

    for tx_pos, depth, amplitude in targets:
        distance = np.sqrt(depth**2 + (xpos - tx_pos)**2)
        delay = int(round(2 * distance / velocity * fs))
        row.append(delay)

    all_true_delays.append(row)

all_true_delays = np.array(all_true_delays)

max_delay = int(np.max(all_true_delays)) + 20
trace_length = max_delay + len(tx) + 100

received = np.zeros(
    (num_positions, trace_length),
    dtype=complex
)

for j, xpos in enumerate(positions):

    for tx_pos, depth, amplitude in targets:
        distance = np.sqrt(depth**2 + (xpos - tx_pos)**2)
        delay = int(round(2 * distance / velocity * fs))

        received[j, delay:delay + len(tx)] += amplitude * tx

    noise = noise_level / np.sqrt(2) * (
        rng.standard_normal(trace_length)
        + 1j * rng.standard_normal(trace_length)
    )

    received[j] += noise

# ======================================================
# 4. BLIND MATCHED-FILTER RESPONSE
# ======================================================
# Do not use all_true_delays in this detection section.
response_matrix = np.zeros(
    (trace_length - len(tx) + 1, num_positions)
)

for j in range(num_positions):

    corr = np.correlate(
        received[j], tx, mode="valid"
    )

    response_matrix[:, j] = np.abs(corr) / energy

# Threshold based on the overall response distribution.
# This is a demonstration threshold, not a calibrated detector.
threshold = max(
    0.08,
    5 * noise_level / np.sqrt(len(tx))
)

# ======================================================
# 5. SHOW A-SCAN AT EACH TARGET'S NEAREST SCAN POSITION
# ======================================================
plt.figure(figsize=(10, 5))

for tx_pos, depth, amplitude in targets:
    j = int(np.argmin(np.abs(positions - tx_pos)))

    response = response_matrix[:, j]
    time_ns = np.arange(len(response)) / fs * 1e9

    plt.plot(
        time_ns,
        response,
        label=f"Scan position {tx_pos:.1f} m"
    )

plt.xlabel("Two-way delay (ns)")
plt.ylabel("Normalized correlation")
plt.title("Multi-Target FSK-GPR A-scans")
plt.grid(True)
plt.legend()
plt.tight_layout()

# ======================================================
# 6. DISPLAY THE MULTI-TARGET B-SCAN
# ======================================================
display_max_ns = 100
display_samples = min(
    response_matrix.shape[0],
    int(display_max_ns * 1e-9 * fs) + 1
)

bscan = response_matrix[:display_samples, :].copy()

# Display only responses above the demonstration threshold.
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
    cmap="turbo"
)

plt.xlabel("Scan position (m)")
plt.ylabel("Two-way delay (ns)")
plt.title("FSK-GPR Multi-Target B-scan")
plt.colorbar(label="Normalized matched-filter response")
plt.tight_layout()
plt.show()

# ======================================================
# 7. DETECT PEAKS AT THE TWO TARGET POSITIONS
# ======================================================
print("\n======= MULTI-TARGET SIMULATION RESULTS =======")

for tx_pos, depth, amplitude in targets:

    j = int(np.argmin(np.abs(positions - tx_pos)))
    response = response_matrix[:, j]

    peaks, properties = find_peaks(
        response,
        height=threshold,
        distance=max(1, int(1e-9 * fs))
    )

    if len(peaks) == 0:
        print(f"\nScan position: {tx_pos:.2f} m")
        print("No peak exceeded the selected threshold.")
        continue

    # Report strongest detected peaks, not assumed target delays.
    order = np.argsort(response[peaks])[::-1]
    peaks = peaks[order[:5]]

    print(f"\nScan position: {tx_pos:.2f} m")
    print(f"Programmed target depth: {depth:.3f} m")
    print("Strongest detected peaks:")

    for peak in peaks:
        delay_ns = peak / fs * 1e9
        estimated_depth = delay_ns * 1e-9 * velocity / 2

        print(
            f"  Delay = {delay_ns:.2f} ns, "
            f"equivalent depth = {estimated_depth:.3f} m, "
            f"response = {response[peak]:.3f}"
        )

print("\nSynthetic software simulation only.")
print("Peak depths are apparent depths; sidelobes and overlapping echoes")
print("can produce detections that are not separate physical targets.")
