
import numpy as np
import matplotlib.pyplot as plt

FS = 500_000
DELAY = 250
COUNT = 2000

reference = np.fromfile("reference_stage53.dat", dtype=np.complex64)
received = np.fromfile("received_stage53.dat", dtype=np.complex64)
demod = np.fromfile("demod_echo_stage51.dat", dtype=np.float32)

count = min(COUNT, len(reference), len(received), len(demod))
time_ms = np.arange(count) / FS * 1000
delay_ms = DELAY / FS * 1000

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

axes[0].plot(time_ms, np.real(reference[:count]), linewidth=0.8)
axes[0].set_title("FSK Reference Signal")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

axes[1].plot(time_ms, np.real(received[:count]), linewidth=0.8)
axes[1].set_title("Simulated Received Signal")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True)

axes[2].plot(time_ms, demod[:count], linewidth=0.8)
axes[2].set_title("GNU Radio Quadrature Demodulator Output")
axes[2].set_ylabel("Output")
axes[2].set_xlabel("Time (ms)")
axes[2].grid(True)

for ax in axes:
    ax.axvline(
        delay_ms,
        linestyle="--",
        linewidth=1.5,
        label=f"Programmed delay = {delay_ms:.3f} ms"
    )
    ax.legend(loc="upper right")

fig.suptitle(
    "FSK-GPR Software Simulation — Stage 62",
    fontsize=15
)
fig.tight_layout()

fig.savefig("stage62_final_visualization.png", dpi=180)
print("Saved: stage62_final_visualization.png")
print(f"Programmed delay: {DELAY} samples = {delay_ms:.3f} ms")
print("Note: Software simulation, not measured physical GPR data.")

plt.show()
