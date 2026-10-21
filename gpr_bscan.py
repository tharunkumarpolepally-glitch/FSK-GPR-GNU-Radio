
import numpy as np
import matplotlib.pyplot as plt

# Scan configuration
x = np.linspace(0, 5, 301)
t_ns = np.linspace(0, 100, 1001)
t = t_ns * 1e-9

# Assumed target and ground properties
x0 = 2.5
depth = 0.15
v = 1.0e8
pulse_width = 1.5e-9

# Simulated B-scan
scan = np.zeros((len(t), len(x)))
rng = np.random.default_rng(10)

for j, xpos in enumerate(x):
    # Two-way travel time for a point target
    tau = 2 * np.sqrt(depth**2 + (xpos - x0)**2) / v

    # Gaussian reflection pulse
    scan[:, j] = np.exp(
        -0.5 * ((t - tau) / pulse_width)**2
    )

# Add background noise
scan += 0.005 * rng.standard_normal(scan.shape)

# Display time increasing downward
plt.figure(figsize=(10, 5))
plt.imshow(
    scan,
    extent=[x.min(), x.max(), t_ns.max(), t_ns.min()],
    aspect="auto",
    interpolation="bilinear",
    cmap="turbo",
    vmin=-0.02,
    vmax=1.0
)

plt.xlabel("Scan position (m)")
plt.ylabel("Two-way travel time (ns)")
plt.title("Synthetic GPR B-scan — Point Target")
plt.colorbar(label="Relative echo amplitude")
plt.tight_layout()
plt.show()

print(f"Target horizontal position: {x0:.2f} m")
print(f"Assumed target depth: {depth:.2f} m")
print(f"Propagation velocity: {v:.2e} m/s")
print(f"Minimum travel time: {2 * depth / v * 1e9:.2f} ns")
