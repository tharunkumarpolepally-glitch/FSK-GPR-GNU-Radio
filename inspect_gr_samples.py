
import os
import numpy as np

filename = "gr_samples.dat"

file_size = os.path.getsize(filename)

print("========== GNU RADIO SAMPLE CHECK ==========")
print(f"File size: {file_size:,} bytes")

# GNU Radio File Sink with complex-float input
bytes_per_sample = 8
sample_count = file_size // bytes_per_sample

print(f"Estimated complex samples: {sample_count:,}")
print(f"Duration at 500 kS/s: {sample_count / 500_000:.2f} seconds")

# Read only the first 1 million samples
with open(filename, "rb") as f:
    samples = np.fromfile(f, dtype=np.complex64, count=1_000_000)

print(f"Samples loaded: {len(samples):,}")

if len(samples) > 0:
    print(f"Maximum magnitude: {np.max(np.abs(samples)):.6f}")
    print(f"Mean magnitude: {np.mean(np.abs(samples)):.6f}")
    print(f"Finite samples: {np.isfinite(samples).all()}")
