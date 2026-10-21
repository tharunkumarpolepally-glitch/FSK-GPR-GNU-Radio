"""
Stage 31: Check whether the secondary correlation peak is caused by
periodicity in the reference signal or by the delayed branch.

Reads reference.dat and received.dat as complex64. It:
- compares reference autocorrelation peaks with residual/reference correlation
- checks reference similarity at a 1,000-sample shift
- reports whether the 1,250-sample correlation peak has a corresponding
  1,000-sample periodic pattern in the reference

This is a software-signal diagnostic, not physical GPR validation.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags, find_peaks

REFERENCE_FILE = Path("reference.dat")
RECEIVED_FILE = Path("received.dat")
FS = 500_000.0
EXPECTED_DELAY = 250
SECONDARY_LAG = 1250
PERIOD_CANDIDATE = SECONDARY_LAG - EXPECTED_DELAY  # 1000 samples
READ_COUNT = 100_000
MAX_LAG = 2_000
SEGMENT_LENGTH = 20_000

def read_complex(path, count):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.resolve()}")
    data = np.fromfile(path, dtype=np.complex64, count=count)
    if len(data) < 5_000:
        raise ValueError(f"Not enough samples in {path.name}")
    return data

def normalized_similarity(a, b):
    a = a.astype(np.complex128, copy=False)
    b = b.astype(np.complex128, copy=False)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 1e-12:
        return float("nan")
    return float(abs(np.vdot(a, b)) / den)

def main():
    print("===== STAGE 31: SECONDARY PEAK / PERIODICITY CHECK =====")
    ref = read_complex(REFERENCE_FILE, READ_COUNT)
    rx = read_complex(RECEIVED_FILE, READ_COUNT)
    n = min(len(ref), len(rx))
    ref, rx = ref[:n], rx[:n]
    valid = (np.isfinite(ref.real) & np.isfinite(ref.imag) &
             np.isfinite(rx.real) & np.isfinite(rx.imag))
    ref, rx = ref[valid], rx[valid]
    if len(ref) <= PERIOD_CANDIDATE + SEGMENT_LENGTH:
        raise ValueError("Not enough samples for the 1,000-sample periodicity test.")

    # Use first 250 samples to estimate direct-path scale before the configured echo.
    pre = EXPECTED_DELAY
    den = np.vdot(ref[:pre], ref[:pre])
    scale = np.vdot(ref[:pre], rx[:pre]) / den if abs(den) > 1e-12 else 1+0j
    residual = rx - scale * ref

    # Correlation curves (positive lags only).
    max_lag = min(MAX_LAG, len(ref)-2)
    ar = correlate(ref, ref, mode="full", method="fft")
    ar_lags = correlation_lags(len(ref), len(ref), mode="full")
    ar_mask = (ar_lags >= 0) & (ar_lags <= max_lag)
    ar_lags, ar_mag = ar_lags[ar_mask], np.abs(ar[ar_mask])

    cr = correlate(residual, ref, mode="full", method="fft")
    cr_lags = correlation_lags(len(residual), len(ref), mode="full")
    cr_mask = (cr_lags >= 0) & (cr_lags <= max_lag)
    cr_lags, cr_mag = cr_lags[cr_mask], np.abs(cr[cr_mask])

    # Similarity at candidate periodic shift using aligned windows.
    L = min(SEGMENT_LENGTH, len(ref)-PERIOD_CANDIDATE)
    sim = normalized_similarity(ref[:L], ref[PERIOD_CANDIDATE:PERIOD_CANDIDATE+L])
    corr_peak_250 = float(cr_mag[np.argmin(np.abs(cr_lags-EXPECTED_DELAY))])
    corr_peak_1250 = float(cr_mag[np.argmin(np.abs(cr_lags-SECONDARY_LAG))])
    ref_peak_1000 = float(ar_mag[np.argmin(np.abs(ar_lags-PERIOD_CANDIDATE))])
    ref_zero = float(ar_mag[0])
    normalized_ref_corr_1000 = ref_peak_1000 / max(ref_zero, 1e-12)

    # Find notable reference autocorrelation lags excluding lag 0.
    search = (ar_lags >= 1) & (ar_lags <= max_lag)
    peaks, props = find_peaks(ar_mag[search], distance=50,
                              prominence=max(float(ar_mag.max())*0.03, 1e-9))
    actual_lags = ar_lags[search]
    actual_mag = ar_mag[search]
    ranked = peaks[np.argsort(actual_mag[peaks])[::-1]][:10]

    print(f"Samples analyzed: {len(ref):,}")
    print(f"Candidate period: {PERIOD_CANDIDATE} samples ({PERIOD_CANDIDATE/FS*1e3:.3f} ms)")
    print(f"Reference similarity across a {PERIOD_CANDIDATE}-sample shift: {sim:.4f}")
    print(f"Reference normalized autocorrelation at lag {PERIOD_CANDIDATE}: {normalized_ref_corr_1000:.4f}")
    print(f"Residual/reference correlation at lag {EXPECTED_DELAY}: {corr_peak_250:.3f}")
    print(f"Residual/reference correlation at lag {SECONDARY_LAG}: {corr_peak_1250:.3f}")
    print("\nStrong reference autocorrelation candidates:")
    print("Lag (samples) | Time (ms) | Relative autocorrelation")
    for i in ranked:
        lag = int(actual_lags[i])
        print(f"{lag:13d} | {lag/FS*1e3:9.6f} | {actual_mag[i]/max(ref_zero,1e-12):.4f}")

    if sim > 0.8 or normalized_ref_corr_1000 > 0.8:
        conclusion = "Strong reference repetition at 1,000 samples is plausible."
    else:
        conclusion = "No strong 1,000-sample repetition was detected by these metrics."
    print("\nInterpretation:", conclusion)
    print("A repeated correlation peak is not evidence of a second physical target.")
    print("Capture alignment, symbol sequence, and flowgraph wiring still need verification.")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), constrained_layout=True)
    axes[0].plot(ar_lags, ar_mag / max(ref_zero, 1e-12), label="Reference autocorrelation")
    axes[0].axvline(PERIOD_CANDIDATE, color="orange", linestyle="--",
                    label=f"Candidate period: {PERIOD_CANDIDATE}")
    axes[0].set_title("Stage 31 — Reference Autocorrelation")
    axes[0].set_xlabel("Lag (samples)")
    axes[0].set_ylabel("Normalized magnitude")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(cr_lags, cr_mag, label="Residual/reference correlation")
    axes[1].axvline(EXPECTED_DELAY, color="red", linestyle="--",
                    label=f"Expected echo delay: {EXPECTED_DELAY}")
    axes[1].axvline(SECONDARY_LAG, color="orange", linestyle=":",
                    label=f"Secondary peak: {SECONDARY_LAG}")
    axes[1].set_title("Stage 31 — Residual Correlation")
    axes[1].set_xlabel("Lag (samples)")
    axes[1].set_ylabel("Correlation magnitude")
    axes[1].grid(True)
    axes[1].legend()
    fig.savefig("stage31_periodicity_check.png", dpi=160)
    plt.show()

    with open("stage31_periodicity_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["samples_analyzed", "candidate_period_samples",
                         "reference_shift_similarity", "reference_norm_autocorr_at_period",
                         "residual_ref_corr_at_250", "residual_ref_corr_at_1250",
                         "conclusion"])
        writer.writerow([len(ref), PERIOD_CANDIDATE, sim, normalized_ref_corr_1000,
                         corr_peak_250, corr_peak_1250, conclusion])
    print("\nSaved:")
    print("  stage31_periodicity_check.png")
    print("  stage31_periodicity_summary.csv")

if __name__ == "__main__":
    main()
