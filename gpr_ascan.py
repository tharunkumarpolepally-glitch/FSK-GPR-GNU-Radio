
import numpy as np
import matplotlib.pyplot as plt

# 1. Radar simulation parameters
fs = 500_000
samples_per_symbol = 50
delay_samples = 250
echo_amplitude = 0.4
num_samples = 10000

# 2. Generate a reproducible binary FSK sequence
rng = np.random.default_rng(7)
num_symbols = (num_samples + samples_per_symbol - 1) // samples_per_symbol
bits = rng.integers(0, 2, num_symbols)
symbols = 2 * bits - 1
symbol_stream = np.repeat(symbols, samples_per_symbol)[:num_samples]

# 3. Generate complex FSK baseband signal
freq_dev = 25_000
phase = np.cumsum(2 * np.pi * freq_dev * symbol_stream / fs)
tx = np.exp(1j * phase)

# 4. Simulate direct signal and delayed echo
rx = np.zeros_like(tx)
rx[delay_samples:] += echo_amplitude * tx[:-delay_samples]

# 5. Correlate received signal with a reference segment
template_length = 1000
template = tx[:template_length]
corr = np.correlate(rx, template, mode="full")
lags = np.arange(-(template_length - 1), len(rx))

# 6. Display the positive-delay correlation region
max_delay = 1000
mask = (lags >= 0) & (lags <= max_delay)
delay_ms = lags[mask] / fs * 1000
magnitude = np.abs(corr[mask])

plt.figure(figsize=(10, 5))
plt.plot(delay_ms, magnitude)
plt.axvline(
    delay_samples / fs * 1000,
    linestyle="--",
    label="Programmed echo delay: 0.5 ms"
)
plt.xlabel("Delay (ms)")
plt.ylabel("Correlation magnitude")
plt.title("FSK-GPR Simulated A-scan")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

peak_index = np.argmax(magnitude)
estimated_delay = lags[mask][peak_index] / fs
print(f"Strongest correlation delay: {estimated_delay * 1000:.3f} ms")
print("Programmed echo delay: 0.500 ms")
