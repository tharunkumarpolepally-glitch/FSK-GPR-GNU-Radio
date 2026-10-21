"""
Stage 32: Test for correlation ambiguity using a non-repeating reference segment.

This script analyzes the existing GNU Radio files first and then creates a
software-only comparison using a deterministic pseudo-random complex reference
sequence and a known delayed copy. It does NOT modify reference.dat or received.dat.

Purpose:
- demonstrate how a non-repeating sequence helps distinguish a single delay
- compare the current repeated reference with a non-repeating test sequence
- confirm the known software delay with a controlled synthetic test

This is not physical GPR validation.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

REFERENCE_FILE = Path("reference.dat")
RECEIVED_FILE = Path("received.dat")
FS = 500_000.0
EXPECTED_DELAY = 250
READ_COUNT = 20_000
TEST_LENGTH = 20_000
ECHO_GAIN = 0.4
RANDOM_SEED = 32
MAX_LAG = 2_000

def load_complex(path, count):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.resolve()}")
    values = np.fromfile(path, dtype=np.complex64, count=count)
    if values.size < 5_000:
        raise ValueError(f"Not enough samples in {path.name}")
    return values

def positive_correlation(a, b, max_lag):
    c = correlate(a, b, mode="full", method="fft")
    lag_all = correlation_lags(len(a), len(b), mode="full")
    mask = (lag_all >= 0) & (lag_all <= min(max_lag, len(a)-2))
    return lag_all[mask], np.abs(c[mask])

def main():
    print("===== STAGE 32: REDUCE DELAY AMBIGUITY =====")
    print("Part A: inspect current GNU Radio reference")
    ref = load_complex(REFERENCE_FILE, READ_COUNT)
    rx = load_complex(RECEIVED_FILE, READ_COUNT)
    n = min(len(ref), len(rx))
    ref, rx = ref[:n], rx[:n]
    valid = (np.isfinite(ref.real) & np.isfinite(ref.imag) &
             np.isfinite(rx.real) & np.isfinite(rx.imag))
    ref, rx = ref[valid], rx[valid]

    # Estimate the direct-path scale using the first 250 samples.
    d = min(EXPECTED_DELAY, len(ref))
    den = np.vdot(ref[:d], ref[:d])
    scale = np.vdot(ref[:d], rx[:d]) / den if abs(den) > 1e-12 else 1+0j
    residual = rx - scale*ref
    current_lags, current_corr = positive_correlation(residual, ref, MAX_LAG)
    current_main = float(current_corr[np.argmin(abs(current_lags-EXPECTED_DELAY))])
    current_repeat = float(current_corr[np.argmin(abs(current_lags-(EXPECTED_DELAY+1000)))])
    print(f"Current peak at {EXPECTED_DELAY}: {current_main:.3f}")
    print(f"Current peak at {EXPECTED_DELAY+1000}: {current_repeat:.3f}")
    print(f"Secondary/main ratio: {current_repeat/max(current_main,1e-12):.4f}")

    print("\nPart B: controlled non-repeating sequence test")
    rng = np.random.default_rng(RANDOM_SEED)
    # Complex random sequence with unit average power.
    tx = ((rng.standard_normal(TEST_LENGTH) +
           1j*rng.standard_normal(TEST_LENGTH)) / np.sqrt(2)).astype(np.complex64)
    # Create delayed echo; samples before delay contain only direct path.
    echo = np.zeros(TEST_LENGTH, dtype=np.complex64)
    echo[EXPECTED_DELAY:] = ECHO_GAIN * tx[:-EXPECTED_DELAY]
    rx_test = tx + echo
    residual_test = rx_test - tx
    test_lags, test_corr = positive_correlation(residual_test, tx, MAX_LAG)

    # Correlation has finite-window edge effects; report strongest lag.
    idx = int(np.argmax(test_corr))
    est = int(test_lags[idx])
    main = float(test_corr[np.argmin(abs(test_lags-EXPECTED_DELAY))])
    other_mask = np.abs(test_lags-EXPECTED_DELAY) > 50
    max_other = float(np.max(test_corr[other_mask])) if np.any(other_mask) else float("nan")
    print(f"Expected test delay: {EXPECTED_DELAY} samples ({EXPECTED_DELAY/FS*1e3:.3f} ms)")
    print(f"Estimated test delay: {est} samples ({est/FS*1e3:.3f} ms)")
    print(f"Correlation at expected delay: {main:.3f}")
    print(f"Strongest correlation outside ±50 samples: {max_other:.3f}")
    print("Controlled test: " + ("PASS" if est == EXPECTED_DELAY else "REVIEW"))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(current_lags, current_corr, label="Current GNU Radio files")
    axes[0].axvline(EXPECTED_DELAY, color="red", linestyle="--", label="Expected 250")
    axes[0].axvline(EXPECTED_DELAY+1000, color="orange", linestyle=":",
                    label="Repeated-pattern candidate 1250")
    axes[0].set_title("Current capture: correlation ambiguity")
    axes[0].set_xlabel("Lag (samples)")
    axes[0].set_ylabel("Correlation magnitude")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(test_lags, test_corr, label="Non-repeating random reference test")
    axes[1].axvline(EXPECTED_DELAY, color="red", linestyle="--",
                    label=f"Expected delay {EXPECTED_DELAY}")
    axes[1].axvline(est, color="green", linestyle=":",
                    label=f"Estimated delay {est}")
    axes[1].set_title("Controlled test: non-repeating reference")
    axes[1].set_xlabel("Lag (samples)")
    axes[1].set_ylabel("Correlation magnitude")
    axes[1].grid(True)
    axes[1].legend()
    fig.savefig("stage32_delay_ambiguity_comparison.png", dpi=160)
    plt.show()

    with open("stage32_delay_ambiguity_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test", "expected_delay_samples", "estimated_delay_samples",
                         "correlation_at_expected", "largest_other_peak", "secondary_to_main_ratio"])
        writer.writerow(["existing GNU Radio files", EXPECTED_DELAY, EXPECTED_DELAY,
                         current_main, current_repeat, current_repeat/max(current_main,1e-12)])
        writer.writerow(["controlled random sequence", EXPECTED_DELAY, est,
                         main, max_other, max_other/max(main,1e-12)])

    print("\nSaved:")
    print("  stage32_delay_ambiguity_comparison.png")
    print("  stage32_delay_ambiguity_summary.csv")
    print("\nNote: the controlled random sequence is synthetic and is not a replacement")
    print("for validating the actual GNU Radio flowgraph or measured radar data.")

if __name__ == "__main__":
    main()
