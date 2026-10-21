
import os
import numpy as np
import matplotlib.pyplot as plt

FS = 500_000
DELAY = 250
ECHO_GAIN = 0.4

FILES = {
    "reference": "reference_stage53.dat",
    "received": "received_stage53.dat",
    "demodulated": "demod_echo_stage51.dat",
}

print("=" * 65)
print("STAGE 61: DIRECT CAPTURE COMPARISON")
print("=" * 65)

for name, filename in FILES.items():
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing file: {filename}")

reference = np.fromfile(FILES["reference"], dtype=np.complex64)
received = np.fromfile(FILES["received"], dtype=np.complex64)
demod = np.fromfile(FILES["demodulated"], dtype=np.float32)

print(f"Reference samples:   {len(reference):,}")
print(f"Received samples:    {len(received):,}")
print(f"Demodulated samples: {len(demod):,}")
print(f"Sample rate:         {FS:,} samples/s")
print(f"Programmed delay:    {DELAY} samples ({DELAY / FS * 1000:.3f} ms)")

if len(reference) != len(received):
    raise ValueError("Reference and received lengths differ.")

# Compare received signal with the programmed echo model.
expected = reference.copy()
expected[DELAY:] += ECHO_GAIN * reference[:-DELAY]

error = received - expected
error_rms = float(np.sqrt(np.mean(np.abs(error) ** 2)))

print("\nEcho-model comparison")
print(f"Reference RMS: {np.sqrt(np.mean(np.abs(reference)**2)):.6f}")
print(f"Received RMS:  {np.sqrt(np.mean(np.abs(received)**2)):.6f}")
print(f"Model error RMS: {error_rms:.8g}")

print("\nDemodulator statistics")
print(f"Minimum: {np.min(demod):.6f}")
print(f"Maximum: {np.max(demod):.6f}")
print(f"Mean:    {np.mean(demod):.6f}")
print(f"Std dev: {np.std(demod):.6f}")

# Plot a short window to make transitions visible.
count = min(2000, len(reference), len(demod))
time_ms = np.arange(count) / FS * 1000

fig, axes = plt.subplots(3, 1, figsize=(12, 9))

axes[0].plot(time_ms, np.real(reference[:count]))
axes[0].set_title("Stage 61: Reference signal")
axes[0].set_ylabel("Real amplitude")
axes[0].grid(True)

axes[1].plot(time_ms, np.real(received[:count]))
axes[1].set_title("Simulated received signal")
axes[1].set_ylabel("Real amplitude")
axes[1].grid(True)

axes[2].step(
    time_ms[:min(count, len(demod))],
    demod[:count],
    where="post"
)
axes[2].set_title("Actual GNU Radio demodulator capture")
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("Demod output")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("stage61_capture_comparison.png", dpi=160)
print("\nSaved: stage61_capture_comparison.png")
plt.show()
