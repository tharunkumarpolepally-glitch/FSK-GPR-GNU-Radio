
import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STAGE 58: SIMULATED FSK-GPR A-SCAN DISPLAY
# ============================================================

FS = 500_000                 # Sample rate: 500 kS/s
VELOCITY = 1e8               # Assumed velocity (m/s), illustrative only
DISPLAY_MS = 2.0             # Display the first 2 milliseconds
RX_FILE = "received_stage53.dat"

TIME_OUTPUT = "stage58_ascan_time.png"
DEPTH_OUTPUT = "stage58_ascan_depth.png"

print("=" * 60)
print("STAGE 58: SIMULATED GPR A-SCAN")
print("=" * 60)

# ------------------------------------------------------------
# 1. Check and load the received-signal file
# ------------------------------------------------------------

if not os.path.exists(RX_FILE):
    raise FileNotFoundError(
        f"Cannot find {RX_FILE}. "
        "Run this script from your FSK_GPR project folder."
    )

rx = np.fromfile(RX_FILE, dtype=np.complex64)

if len(rx) == 0:
    raise ValueError("The received-signal file is empty.")

# Signal magnitude
amplitude = np.abs(rx)

# ------------------------------------------------------------
# 2. Select the first 2 ms
# ------------------------------------------------------------

total_samples = len(rx)
display_samples = min(
    total_samples,
    int(DISPLAY_MS * 1e-3 * FS)
)

rx_view = rx[:display_samples]
amplitude_view = np.abs(rx_view)

# Time axis in milliseconds
time_s = np.arange(display_samples) / FS
time_ms = time_s * 1000

# ------------------------------------------------------------
# 3. Plot A-scan against time
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))
plt.plot(time_ms, amplitude_view, linewidth=0.8)

plt.xlabel("Time (ms)")
plt.ylabel("Signal magnitude")
plt.title("Stage 58: Simulated FSK-GPR A-Scan")
plt.xlim(0, DISPLAY_MS)
plt.grid(True)
plt.tight_layout()

plt.savefig(TIME_OUTPUT, dpi=150)
print(f"Saved: {TIME_OUTPUT}")

# ------------------------------------------------------------
# 4. Convert time to illustrative depth
#    d = velocity * time / 2
# ------------------------------------------------------------

depth_m = time_s * VELOCITY / 2

plt.figure(figsize=(12, 5))
plt.plot(depth_m, amplitude_view, linewidth=0.8)

plt.xlabel("Illustrative depth (m)")
plt.ylabel("Signal magnitude")
plt.title("Stage 58: A-Scan with Assumed Depth Conversion")
plt.xlim(0, depth_m[-1] if len(depth_m) > 1 else 1)
plt.grid(True)
plt.tight_layout()

plt.savefig(DEPTH_OUTPUT, dpi=150)
print(f"Saved: {DEPTH_OUTPUT}")

# ------------------------------------------------------------
# 5. Print summary
# ------------------------------------------------------------

print("\nCAPTURE SUMMARY")
print("-" * 40)
print(f"Total samples:       {total_samples:,}")
print(f"Sample rate:         {FS:,} samples/s")
print(f"Full capture time:   {total_samples / FS:.6f} s")
print(f"Displayed samples:   {display_samples:,}")
print(f"Displayed time:      {time_ms[-1]:.6f} ms")
print(f"Maximum magnitude:   {np.max(amplitude):.6f}")
print(f"Assumed velocity:    {VELOCITY:.3g} m/s")
print(f"Displayed depth end: {depth_m[-1]:.3f} m")

# The programmed delay from earlier stages
delay_samples = 250
delay_ms = delay_samples / FS * 1000
illustrative_delay_depth = VELOCITY * delay_samples / FS / 2

print(f"\nProgrammed delay:    {delay_samples} samples")
print(f"Delay time:          {delay_ms:.6f} ms")
print(f"Illustrative depth:  {illustrative_delay_depth:.3f} m")

print("\nIMPORTANT:")
print("The depth conversion is illustrative only.")
print("This is a software-generated signal, not a physical")
print("GPR measurement or validated buried-target depth.")

plt.show()
