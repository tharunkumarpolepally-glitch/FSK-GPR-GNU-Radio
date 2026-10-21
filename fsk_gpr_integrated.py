
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. PARAMETERS
# =====================================================
fs = 1e9
symbol_rate = 10e6
freq_dev = 25e6
samples_per_symbol = int(fs / symbol_rate)

num_symbols = 100
num_positions = 201
scan_width = 5.0

# Simulated point target
target_x = 2.5
target_depth = 0.15
velocity = 1e8

echo_amplitude = 0.4
noise_level = 0.015
detection_threshold = 0.10

rng = np.random.default_rng(7)

# =====================================================
# 2. GENERATE TRANSMITTED FSK REFERENCE
# =====================================================
bits = rng.integers(0, 2, num_symbols)
symbols = 2 * bits - 1
symbol_stream = np.repeat(symbols, samples_per_symbol)

phase = np.cumsum(
    2 * np.pi * freq_dev * symbol_stream / fs
)
tx = np.exp(1j * phase)

# Fade burst edges
edge = min(100, len(tx) // 4)
window = np.ones(len(tx))
window[:edge] = np.linspace(0, 1, edge)
window[-edge:] = np.linspace(1, 0, edge)
tx *= window

energy = np.sum(np.abs(tx)**2)

# =====================================================
# 3. SIMULATE RECEIVED ECHOES
# =====================================================
positions = np.linspace(0, scan_width, num_positions)

distance = np.sqrt(
    target_depth**2 + (positions - target_x)**2
)
true_delays = np.rint(
    2 * distance / velocity * fs
).astype(int)

max_delay = int(np.max(true_delays)) + 10
trace_length = max_delay + len(tx) + 100

received = np.zeros(
    (num_positions, trace_length),
    dtype=complex
)

for j, delay in enumerate(true_delays):
    received[j, delay:delay + len(tx)] += echo_amplitude * tx

    noise = noise_level / np.sqrt(2) * (
        rng.standard_normal(trace_length)
        + 1j * rng.standard_normal(trace_length)
    )
    received[j] += noise

# =====================================================
# 4. BLIND MATCHED-FILTER DETECTION
#    IMPORTANT: true_delays are NOT used in this section.
# =====================================================
estimated_delays = np.zeros(num_positions, dtype=int)
peak_strength = np.zeros(num_positions)

for j in range(num_positions):
    corr = np.correlate(received[j], tx, mode="valid")
    magnitude = np.abs(corr) / energy

    # Search the full valid delay range
    idx = int(np.argmax(magnitude))

    estimated_delays[j] = idx
    peak_strength[j] = magnitude[idx]

# Apply a detection threshold
detected = peak_strength >= detection_threshold

# =====================================================
# 5. A-SCAN AT CENTRE POSITION
# =====================================================
centre = int(np.argmin(np.abs(positions - target_x)))
trace = received[centre]

corr = np.correlate(trace, tx, mode="valid")
response = np.abs(corr) / energy

max_display_ns = 100
limit = min(len(response), int(max_display_ns * 1e-9 * fs) + 1)
delay_axis_ns = np.arange(limit) / fs * 1e9

plt.figure(figsize=(10, 4))
plt.plot(delay_axis_ns, response[:limit],
         label="Blind matched-filter response")
plt.axvline(
    estimated_delays[centre] / fs * 1e9,
    color="green", linestyle="--",
    label="Detected delay"
)
plt.xlabel("Delay (ns)")
plt.ylabel("Normalized correlation")
plt.title("FSK-GPR A-scan — Blind Detection")
plt.grid(True)
plt.legend()
plt.tight_layout()

# =====================================================
# 6. B-SCAN FROM DETECTED ECHOES
# =====================================================
display_max_ns = 110
display_samples = min(
    trace_length, int(display_max_ns * 1e-9 * fs)
)

bscan = np.zeros((display_samples, num_positions))

for j in range(num_positions):
    if detected[j]:
        idx = estimated_delays[j]
        if idx < display_samples:
            bscan[idx, j] = peak_strength[j]

            # Make detected responses easier to see
            for offset, weight in [
                (-2, 0.25), (-1, 0.65),
                (1, 0.65), (2, 0.25)
            ]:
                k = idx + offset
                if 0 <= k < display_samples:
                    bscan[k, j] = max(
                        bscan[k, j],
                        peak_strength[j] * weight
                    )

plt.figure(figsize=(10, 5))
plt.imshow(
    bscan,
    extent=[
        positions[0], positions[-1],
        display_max_ns, 0
    ],
    aspect="auto",
    interpolation="nearest",
    cmap="turbo"
)
plt.xlabel("Scan position (m)")
plt.ylabel("Two-way travel time (ns)")
plt.title("FSK-GPR B-scan — Blind Detection")
plt.colorbar(label="Matched-filter response")
plt.tight_layout()
plt.show()

# =====================================================
# 7. EVALUATE DETECTION AFTERWARDS
#    Known delays are used only for evaluation.
# =====================================================
estimated_delay_ns = estimated_delays / fs * 1e9
true_delay_ns = true_delays / fs * 1e9

errors_ns = estimated_delay_ns - true_delay_ns
depth_estimates = estimated_delays / fs * velocity / 2

print("\n========== BLIND DETECTION RESULTS ==========")
print(f"Target position: {target_x:.2f} m")
print(f"Known target depth: {target_depth:.3f} m")
print(f"Programmed centre delay: {true_delay_ns[centre]:.2f} ns")
print(f"Blind estimated centre delay: {estimated_delay_ns[centre]:.2f} ns")
print(f"Centre delay error: {errors_ns[centre]:.2f} ns")
print(f"Centre estimated depth: {depth_estimates[centre]:.3f} m")
print(f"Detected scan traces: {np.count_nonzero(detected)}/{num_positions}")
print(f"Detection threshold: {detection_threshold:.2f}")
print("Software simulation only; not a physical radar measurement.")
