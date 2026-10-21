"""
Stage 30: Correlation-peak analysis and delay ambiguity check.

Reads reference.dat and received.dat as complex64. It estimates a direct-path
scale from the first EXPECTED_DELAY_SAMPLES, subtracts the scaled reference,
then calculates residual/reference correlation. It lists prominent candidate
lags and compares their strengths.

IMPORTANT:
- A repeated correlation peak can be caused by periodic/repetitive signal structure,
  not necessarily a second physical target.
- This is a GNU Radio software-delay analysis, not validated physical GPR data.
- At 500 kS/s, 250 samples = 0.5 ms, not a realistic shallow-GPR round-trip time.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags, find_peaks

REFERENCE_FILE = Path("reference.dat")
RECEIVED_FILE = Path("received.dat")
SAMPLE_RATE_HZ = 500_000.0
EXPECTED_DELAY_SAMPLES = 250
SAMPLES_TO_READ = 100_000
MAX_SEARCH_LAG = 2_000
MIN_PEAK_DISTANCE_SAMPLES = 50
PEAK_PROMINENCE_FRACTION = 0.08

def load_complex64(path, count):
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path.resolve()}. Put this script beside reference.dat "
            "and received.dat."
        )
    if path.stat().st_size < 8:
        raise ValueError(f"{path.name} is empty or too small for complex64 samples.")
    x = np.fromfile(path, dtype=np.complex64, count=count)
    if x.size == 0:
        raise ValueError(f"No samples read from {path.name}.")
    return x

def main():
    print("===== STAGE 30: CORRELATION PEAK ANALYSIS =====")
    print(f"Sample rate: {SAMPLE_RATE_HZ:,.0f} samples/s")
    print(f"Expected software delay: {EXPECTED_DELAY_SAMPLES} samples")

    ref = load_complex64(REFERENCE_FILE, SAMPLES_TO_READ)
    rx = load_complex64(RECEIVED_FILE, SAMPLES_TO_READ)
    n = min(len(ref), len(rx))
    ref, rx = ref[:n], rx[:n]
    valid = (np.isfinite(ref.real) & np.isfinite(ref.imag) &
             np.isfinite(rx.real) & np.isfinite(rx.imag))
    ref, rx = ref[valid], rx[valid]
    if len(ref) < EXPECTED_DELAY_SAMPLES + 2_000:
        raise ValueError("Not enough valid samples for this analysis.")

    # Estimate direct path from the interval before the expected echo starts.
    n_pre = EXPECTED_DELAY_SAMPLES
    denom = np.vdot(ref[:n_pre], ref[:n_pre])
    scale = np.vdot(ref[:n_pre], rx[:n_pre]) / denom if abs(denom) > 1e-12 else 1+0j
    residual = rx - scale * ref

    max_lag = min(MAX_SEARCH_LAG, len(ref)-2)
    c = correlate(residual, ref, mode="full", method="fft")
    lag_all = correlation_lags(len(residual), len(ref), mode="full")
    mask = (lag_all >= 1) & (lag_all <= max_lag)
    lags = lag_all[mask]
    corr_mag = np.abs(c[mask])
    if corr_mag.size == 0:
        raise ValueError("Correlation range is empty.")

    peak_prominence = max(float(np.max(corr_mag))*PEAK_PROMINENCE_FRACTION, 1e-9)
    local_idx, props = find_peaks(
        corr_mag,
        distance=MIN_PEAK_DISTANCE_SAMPLES,
        prominence=peak_prominence
    )
    if len(local_idx) == 0:
        local_idx = np.array([int(np.argmax(corr_mag))])
    # Sort candidates strongest first, keep at most 12 for console/CSV readability.
    order = local_idx[np.argsort(corr_mag[local_idx])[::-1]][:12]
    candidates = []
    for j in order:
        lag = int(lags[j])
        candidates.append({
            "lag": lag,
            "time_ms": lag / SAMPLE_RATE_HZ * 1e3,
            "corr": float(corr_mag[j]),
            "relative_to_max": float(corr_mag[j] / np.max(corr_mag))
        })

    best_idx = int(np.argmax(corr_mag))
    best_lag = int(lags[best_idx])
    best_corr = float(corr_mag[best_idx])
    expected_ms = EXPECTED_DELAY_SAMPLES / SAMPLE_RATE_HZ * 1e3

    print("\nTop correlation candidates (strongest first):")
    print("Rank | Lag (samples) | Time (ms) | Correlation | Relative strength")
    for rank, item in enumerate(candidates, 1):
        print(f"{rank:4d} | {item['lag']:13d} | {item['time_ms']:9.6f} | "
              f"{item['corr']:11.3f} | {item['relative_to_max']:16.3f}")

    print("\n--- Interpretation ---")
    print(f"Expected lag: {EXPECTED_DELAY_SAMPLES} samples ({expected_ms:.6f} ms)")
    print(f"Strongest lag in search window: {best_lag} samples "
          f"({best_lag/SAMPLE_RATE_HZ*1e3:.6f} ms)")
    if best_lag == EXPECTED_DELAY_SAMPLES:
        print("PASS: the strongest peak matches the configured software delay.")
    else:
        print("REVIEW: strongest peak differs from the configured software delay.")
    secondary = [p for p in candidates if abs(p["lag"]-EXPECTED_DELAY_SAMPLES) > 20]
    if secondary:
        print("Other peaks are correlation candidates, not automatically separate targets.")
        print("They may result from repeated FSK symbols, sequence periodicity, or residual misalignment.")
    print("This script cannot prove that a peak corresponds to a physical object.")

    # Plot correlation, annotate candidate peaks, and show residual.
    pcount = min(5000, len(ref))
    time_ms = np.arange(pcount) / SAMPLE_RATE_HZ * 1e3
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), constrained_layout=True)
    axes[0].plot(lags, corr_mag, linewidth=0.9, label="Correlation magnitude")
    axes[0].axvline(EXPECTED_DELAY_SAMPLES, color="red", linestyle="--",
                    label=f"Expected delay: {EXPECTED_DELAY_SAMPLES}")
    for item in candidates[:5]:
        axes[0].annotate(str(item["lag"]), (item["lag"], item["corr"]),
                         xytext=(3, 5), textcoords="offset points", fontsize=8)
    axes[0].set_title("Stage 30 — Correlation Peaks and Candidate Lags")
    axes[0].set_xlabel("Lag (samples)")
    axes[0].set_ylabel("Correlation magnitude")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(time_ms, np.abs(residual[:pcount]), color="tab:orange", linewidth=0.8)
    axes[1].set_title("Residual Magnitude After Direct-Path Subtraction")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(True)
    fig.savefig("stage30_correlation_peak_analysis.png", dpi=160)
    plt.show()

    with open("stage30_correlation_candidates.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "lag_samples", "time_ms", "correlation_magnitude", "relative_strength"])
        for rank, item in enumerate(candidates, 1):
            writer.writerow([rank, item["lag"], item["time_ms"], item["corr"], item["relative_to_max"]])

    with open("stage30_delay_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_rate_hz", "expected_delay_samples", "strongest_lag_samples",
                         "expected_delay_ms", "strongest_lag_ms", "strongest_matches_expected"])
        writer.writerow([SAMPLE_RATE_HZ, EXPECTED_DELAY_SAMPLES, best_lag, expected_ms,
                         best_lag/SAMPLE_RATE_HZ*1e3, best_lag == EXPECTED_DELAY_SAMPLES])

    print("\nSaved:")
    print("  stage30_correlation_peak_analysis.png")
    print("  stage30_correlation_candidates.csv")
    print("  stage30_delay_summary.csv")

if __name__ == "__main__":
    main()
