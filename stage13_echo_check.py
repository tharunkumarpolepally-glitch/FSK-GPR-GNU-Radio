
import numpy as np
import matplotlib.pyplot as plt

filename = "gr_samples.dat"
fs = 500_000

# Read a manageable section
with open(filename, "rb") as f:
    samples = np.fromfile(
        f, dtype=np.complex64, count=100_000
    )

if len(samples) < 1000:
    raise ValueError("Not enough samples in the file.")

# Remove the DC component
samples = samples - np.mean(samples)

# Autocorrelation: compare signal with delayed copies of itself
corr = np.correlate(
    samples[:5000],
    samples[:5000],
    mode="full"
)

corr = corr[len(corr)//2:]
corr = np.abs(corr)

if corr[0] > 0:
    corr = corr / corr[0]

delay_samples = np.arange(len(corr))
delay_ms = delay_samples / fs * 1000

plt.figure(figsize=(10, 4))
plt.plot(delay_ms, corr)
plt.xlabel("Lag (ms)")
plt.ylabel("Normalized autocorrelation")
plt.title("Stage 13: GNU Radio Signal Autocorrelation")
plt.xlim(0, 5)
plt.grid(True)
plt.tight_layout()
plt.savefig("stage13_autocorrelation.png", dpi=150)
plt.show()

print("Saved: stage13_autocorrelation.png")
print("Sample rate:", fs, "samples/second")
print("Lag resolution:", 1e6/fs, "microseconds")
