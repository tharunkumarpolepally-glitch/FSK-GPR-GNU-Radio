
import os
import numpy as np
import matplotlib.pyplot as plt

folder = os.path.dirname(os.path.abspath(__file__))

reference_file = os.path.join(folder, "reference_stage53.dat")
received_file = os.path.join(folder, "received_stage53.dat")

sample_rate = 500_000  # 500 kS/s

for filename in (reference_file, received_file):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Missing capture file: {filename}"
        )

reference = np.fromfile(reference_file, dtype=np.complex64)
received = np.fromfile(received_file, dtype=np.complex64)

if len(reference) == 0 or len(received) == 0:
    raise ValueError("A capture file is empty.")

n = min(len(reference), len(received))
reference = reference[:n]
received = received[:n]

time_ms = np.arange(n) / sample_rate * 1000

fig, axes = plt.subplots(2, 1, figsize=(12, 7))

# Display a short section so the waveform is readable.
count = min(n, 4096)

axes[0].plot(
    time_ms[:count], reference.real[:count],
    label="Reference I"
)
axes[0].plot(
    time_ms[:count], reference.imag[:count],
    label="Reference Q", alpha=0.8
)
axes[0].set_title("Captured FSK Reference — A-scan")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Amplitude")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(
    time_ms[:count], received.real[:count],
    label="Received I"
)
axes[1].plot(
    time_ms[:count], received.imag[:count],
    label="Received Q", alpha=0.8
)
axes[1].set_title("Captured Received Signal — A-scan")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Amplitude")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

output = os.path.join(folder, "captured_gpr_ascan.png")
plt.savefig(output, dpi=200)
print(f"A-scan saved to: {output}")

plt.show()
