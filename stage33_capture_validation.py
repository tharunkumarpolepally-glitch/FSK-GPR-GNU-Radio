"""
Stage 33: GNU Radio flowgraph validation checklist and capture diagnostics.

This stage does not change the existing .dat files or pretend to alter the
GNU Radio flowgraph automatically. It checks file alignment, direct-path
agreement, the residual echo onset, and correlation peaks in the current capture.
It also writes a checklist for the next manual GRC change.

Expected project files:
  reference.dat
  received.dat

All files are interpreted as complex64. Software-delay demonstration only.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags, find_peaks

REF_FILE = Path("reference.dat")
RX_FILE = Path("received.dat")
FS = 500_000.0
EXPECTED_DELAY = 250
COUNT = 100_000
MAX_LAG = 2_000
PERIOD_LAG = 1_000

def load(path, count):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.resolve()}. Run this script from your FSK_GPR folder."
        )
    if path.stat().st_size < 8:
        raise ValueError(f"{path.name} is too small to contain complex64 samples.")
    x = np.fromfile(path, dtype=np.complex64, count=count)
    if len(x) == 0:
        raise ValueError(f"No samples read from {path.name}.")
    return x

def main():
    print("===== STAGE 33: CAPTURE AND FLOWGRAPH VALIDATION =====")
    ref = load(REF_FILE, COUNT)
    rx = load(RX_FILE, COUNT)
    n = min(len(ref), len(rx))
    ref, rx = ref[:n], rx[:n]
    valid = (np.isfinite(ref.real) & np.isfinite(ref.imag) &
             np.isfinite(rx.real) & np.isfinite(rx.imag))
    ref, rx = ref[valid], rx[valid]
    if len(ref) <= EXPECTED_DELAY + 1000:
        raise ValueError("Not enough valid paired samples.")

    # Estimate complex direct-path scale from the first delay interval.
    pre = EXPECTED_DELAY
    denom = np.vdot(ref[:pre], ref[:pre])
    scale = np.vdot(ref[:pre], rx[:pre]) / denom if abs(denom) > 1e-12 else 1+0j
    residual = rx - scale*ref

    # Quantify direct-path alignment and the residual onset.
    rel_err = np.linalg.norm(rx[:pre] - scale*ref[:pre]) / max(np.linalg.norm(rx[:pre]), 1e-12)
    residual_mag = np.abs(residual)
    before = float(np.mean(residual_mag[:pre]))
    after_window = residual_mag[pre:min(pre+1000, len(residual_mag))]
    after = float(np.mean(after_window)) if len(after_window) else float("nan")
    ratio = after / max(before, 1e-12)

    # Correlate residual with reference.
    c = correlate(residual, ref, mode="full", method="fft")
    lag_all = correlation_lags(len(residual), len(ref), mode="full")
    mask = (lag_all >= 1) & (lag_all <= min(MAX_LAG, len(ref)-2))
    lags, cmag = lag_all[mask], np.abs(c[mask])
    idx = int(np.argmax(cmag))
    best_lag = int(lags[idx])
    at_expected = float(cmag[np.argmin(abs(lags-EXPECTED_DELAY))])
    at_period = float(cmag[np.argmin(abs(lags-(EXPECTED_DELAY+PERIOD_LAG)))])
    period_ratio = at_period / max(at_expected, 1e-12)

    print(f"Valid aligned sample pairs: {len(ref):,}")
    print(f"Reference file size: {REF_FILE.stat().st_size:,} bytes")
    print(f"Received file size: {RX_FILE.stat().st_size:,} bytes")
    print(f"Estimated direct-path scale: {scale.real:.6f} + {scale.imag:.6f}j")
    print(f"Relative direct-path mismatch over first {pre} samples: {rel_err:.6g}")
    print(f"Mean residual magnitude before expected delay: {before:.6f}")
    print(f"Mean residual magnitude in next 1,000 samples: {after:.6f}")
    print(f"After/before residual ratio: {ratio:.3f}")
    print(f"Correlation at {EXPECTED_DELAY} samples: {at_expected:.3f}")
    print(f"Correlation at {EXPECTED_DELAY+PERIOD_LAG} samples: {at_period:.3f}")
    print(f"Secondary/main ratio: {period_ratio:.4f}")
    print(f"Strongest lag in search window: {best_lag} samples ({best_lag/FS*1e3:.6f} ms)")

    print("\n--- GRC checks to perform manually ---")
    checklist = [
        "Stop the flowgraph before changing any blocks.",
        "Confirm the reference File Sink is connected to the transmitter/modulator output.",
        "Confirm the received File Sink is connected after the Add block.",
        "Confirm the delayed branch contains exactly one Delay block set to 250 samples.",
        "Confirm the delayed branch has one Multiply Const block set to 0.4.",
        "Confirm Add inputs are direct signal and delayed/scaled signal, with no unintended feedback loop.",
        "For the next capture, use a finite capture or stop the flowgraph promptly to avoid huge files.",
        "Use a non-repeating/random symbol sequence for the diagnostic capture if the source supports it.",
        "Restart both files from a fresh capture so their first samples are aligned.",
        "Do not infer physical target depth from the 500 kS/s sample rate."
    ]
    for i, item in enumerate(checklist, 1):
        print(f"{i:2d}. {item}")

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    count_plot = min(5000, len(ref))
    t_ms = np.arange(count_plot)/FS*1e3
    ax[0].plot(t_ms, np.abs(ref[:count_plot]), label="Reference", lw=0.8)
    ax[0].plot(t_ms, np.abs(rx[:count_plot]), label="Received", lw=0.8, alpha=0.8)
    ax[0].set_title("Stage 33 — Reference and Received Capture")
    ax[0].set_xlabel("Time (ms)")
    ax[0].set_ylabel("Magnitude")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(t_ms, residual_mag[:count_plot], color="tab:orange", lw=0.8)
    ax[1].axvline(EXPECTED_DELAY/FS*1e3, color="red", ls="--",
                  label=f"Expected delay: {EXPECTED_DELAY} samples")
    ax[1].set_title("Direct-Path-Subtracted Residual")
    ax[1].set_xlabel("Time (ms)")
    ax[1].set_ylabel("Magnitude")
    ax[1].grid(True)
    ax[1].legend()

    ax[2].plot(lags, cmag, lw=0.9)
    ax[2].axvline(EXPECTED_DELAY, color="red", ls="--", label="Expected 250")
    ax[2].axvline(EXPECTED_DELAY+PERIOD_LAG, color="orange", ls=":",
                  label="Secondary candidate 1250")
    ax[2].set_title("Residual-to-Reference Correlation")
    ax[2].set_xlabel("Lag (samples)")
    ax[2].set_ylabel("Correlation magnitude")
    ax[2].grid(True)
    ax[2].legend()
    fig.savefig("stage33_capture_validation.png", dpi=160)
    plt.show()

    with open("stage33_capture_validation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["valid_pairs", "sample_rate_hz", "expected_delay_samples",
                         "estimated_strongest_lag", "direct_scale_real", "direct_scale_imag",
                         "relative_direct_mismatch", "mean_residual_before",
                         "mean_residual_after", "secondary_to_main_ratio"])
        writer.writerow([len(ref), FS, EXPECTED_DELAY, best_lag, scale.real, scale.imag,
                         rel_err, before, after, period_ratio])

    with open("stage33_grc_checklist.txt", "w", encoding="utf-8") as f:
        f.write("STAGE 33 — MANUAL GNU RADIO COMPANION CHECKLIST\n\n")
        for i, item in enumerate(checklist, 1):
            f.write(f"{i}. {item}\n")

    print("\nSaved:")
    print("  stage33_capture_validation.png")
    print("  stage33_capture_validation.csv")
    print("  stage33_grc_checklist.txt")
    print("\nThis validates the existing software capture; it does not prove physical GPR performance.")

if __name__ == "__main__":
    main()
