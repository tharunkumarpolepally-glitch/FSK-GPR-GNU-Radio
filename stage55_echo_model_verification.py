
import numpy as np
import matplotlib.pyplot as plt
import os

FS = 500_000
DELAY = 250
ECHO_GAIN = 0.4

REF_FILE = "reference_stage53.dat"
RX_FILE = "received_stage53.dat"

for filename in (REF_FILE, RX_FILE):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing file: {filename}")

x = np.fromfile(REF_FILE, dtype=np.complex64)
y = np.fromfile(RX_FILE, dtype=np.complex64)

if len(x) != len(y) or len(x) <= DELAY:
    raise ValueError("Unexpected capture lengths.")

# Construct the expected delayed reference.
delayed = np.zeros_like(x)
delayed[DELAY:] = x[:-DELAY]

# Expected signal under the programmed model.
expected = x + ECHO_GAIN * delayed

# Compare captured received data with the model.
error = y - expected

rms = lambda z: float(np.sqrt(np.mean(np.abs(z)**2)))

print("=" * 58)
print("STAGE 55: ECHO MODEL VERIFICATION")
print("=" * 58)
print(f"Reference samples: {len(x):,}")
print(f"Received samples:  {len(y):,}")
print(f"Reference RMS:     {rms(x):.6f}")
print(f"Received RMS:      {rms(y):.6f}")
print(f"Expected RMS:      {rms(expected):.6f}")
print(f"Model error RMS:   {rms(error):.6f}")

# Estimate the best-fit direct and delayed complex coefficients.
# Use only samples after the initial delay so both components exist.
X = np.column_stack((x[DELAY:], delayed[DELAY:]))
target = y[DELAY:]

coefficients, *_ = np.linalg.lstsq(X, target, rcond=None)
direct_gain, echo_gain = coefficients

fitted = X @ coefficients
fit_error = target - fitted

print("\nBest-fit model coefficients:")
print(f"Direct gain: {direct_gain}")
print(f"Echo gain:   {echo_gain}")
print(f"Best-fit error RMS: {rms(fit_error):.6f}")

# Plot the magnitude of the signals for the first 2 ms.
n = min(1000, len(x))
time_ms = np.arange(n) / FS * 1000

plt.figure(figsize=(12, 5))
plt.plot(time_ms, np.abs(y[:n]), label="Captured received")
plt.plot(time_ms, np.abs(expected[:n]), label="Expected model",
         alpha=0.8)
plt.xlabel("Time (ms)")
plt.ylabel("Magnitude")
plt.title("Stage 55: Captured vs Expected Received Signal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("stage55_echo_model_comparison.png", dpi=150)
plt.show()

print("\nSaved: stage55_echo_model_comparison.png")
