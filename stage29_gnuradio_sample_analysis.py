"""Stage 29: Analyze GNU Radio complex samples and estimate software echo delay.
Reads reference.dat and received.dat as complex64. This is a software-delay test,
not proof of a physical GPR echo. At 500 kS/s, 250 samples = 0.5 ms.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

REFERENCE_FILE = Path("reference.dat")
RECEIVED_FILE = Path("received.dat")
SAMPLE_RATE_HZ = 500_000.0
EXPECTED_DELAY_SAMPLES = 250
SAMPLES_TO_READ = 100_000
MAX_SEARCH_LAG = 2_000
PLOT_SAMPLES = 5_000

def load_samples(path, count):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.resolve()}. Put this script in the FSK_GPR folder "
            "containing reference.dat and received.dat."
        )
    if path.stat().st_size < 8:
        raise ValueError(f"{path.name} is empty or too small for complex64 data.")
    values = np.fromfile(path, dtype=np.complex64, count=count)
    if values.size == 0:
        raise ValueError(f"No complex samples could be read from {path.name}.")
    return values

def main():
    print("===== STAGE 29: GNU RADIO SAMPLE ANALYSIS =====")
    print(f"Working folder: {Path.cwd()}")
    print(f"Sample rate: {SAMPLE_RATE_HZ:,.0f} samples/s")
    print(f"Expected delay: {EXPECTED_DELAY_SAMPLES} samples")

    ref = load_samples(REFERENCE_FILE, SAMPLES_TO_READ)
    rx = load_samples(RECEIVED_FILE, SAMPLES_TO_READ)
    n = min(len(ref), len(rx))
    ref, rx = ref[:n], rx[:n]
    valid = (np.isfinite(ref.real) & np.isfinite(ref.imag) &
             np.isfinite(rx.real) & np.isfinite(rx.imag))
    ref, rx = ref[valid], rx[valid]
    if len(ref) < 2_000:
        raise ValueError(f"Only {len(ref)} valid paired samples; need at least 2,000.")

    for label, values, path in [
        ("Reference", ref, REFERENCE_FILE), ("Received", rx, RECEIVED_FILE)
    ]:
        mag = np.abs(values)
        print(f"\n{label}:")
        print(f"  File size: {path.stat().st_size:,} bytes")
        print(f"  Samples analyzed: {len(values):,}")
        print(f"  Mean magnitude: {np.mean(mag):.6f}")
        print(f"  Maximum magnitude: {np.max(mag):.6f}")

    # Estimate direct-path scale from samples before the expected echo begins.
    pre = min(max(EXPECTED_DELAY_SAMPLES, 10), len(ref))
    denom = np.vdot(ref[:pre], ref[:pre])
    scale = np.vdot(ref[:pre], rx[:pre]) / denom if abs(denom) > 1e-12 else 1 + 0j
    residual = rx - scale * ref

    max_lag = min(MAX_SEARCH_LAG, len(ref) - 2)
    corr = correlate(residual, ref, mode="full", method="fft")
    lags_full = correlation_lags(len(residual), len(ref), mode="full")
    mask = (lags_full >= 1) & (lags_full <= max_lag)
    lags = lags_full[mask]
    corr_mag = np.abs(corr[mask])
    if not len(corr_mag):
        raise ValueError("Not enough data to estimate positive delay.")
    idx = int(np.argmax(corr_mag))
    estimated = int(lags[idx])
    peak = float(corr_mag[idx])
    median = float(np.median(corr_mag))
    ratio = peak / max(median, 1e-12)
    expected_ms = EXPECTED_DELAY_SAMPLES / SAMPLE_RATE_HZ * 1e3
    estimated_ms = estimated / SAMPLE_RATE_HZ * 1e3

    print("\n--- Delay estimate ---")
    print(f"Estimated direct-path scale: {scale.real:.6f} + {scale.imag:.6f}j")
    print(f"Expected delay: {EXPECTED_DELAY_SAMPLES} samples ({expected_ms:.6f} ms)")
    print(f"Estimated delay: {estimated} samples ({estimated_ms:.6f} ms)")
    print(f"Correlation peak magnitude: {peak:.3f}")
    print(f"Peak/median correlation: {ratio:.2f}x")
    print("Delay check: " + (
        "PASS — estimated lag matches the configured software delay."
        if estimated == EXPECTED_DELAY_SAMPLES else
        "REVIEW — estimated lag differs; check stream alignment, file capture, wiring, and delay block."
    ))
    print("\nThis checks a delay relationship in the files; it does not prove a physical radar echo.")
    print("At 500 kS/s, nanosecond-scale shallow-GPR depth estimation is not supported.")

    pcount = min(PLOT_SAMPLES, len(ref))
    time_ms = np.arange(pcount) / SAMPLE_RATE_HZ * 1e3
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    ax[0].plot(time_ms, np.abs(ref[:pcount]), label="Reference magnitude", lw=0.8)
    ax[0].plot(time_ms, np.abs(rx[:pcount]), label="Received magnitude", lw=0.8, alpha=0.8)
    ax[0].set_title("Stage 29 — Reference and Received Signals")
    ax[0].set_xlabel("Time (ms)")
    ax[0].set_ylabel("Magnitude")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(time_ms, np.abs(residual[:pcount]), color="tab:orange", lw=0.8)
    ax[1].set_title("Residual Magnitude: received − estimated direct component")
    ax[1].set_xlabel("Time (ms)")
    ax[1].set_ylabel("Magnitude")
    ax[1].grid(True)

    ax[2].plot(lags, corr_mag, lw=0.9)
    ax[2].axvline(EXPECTED_DELAY_SAMPLES, color="red", ls="--",
                  label=f"Expected: {EXPECTED_DELAY_SAMPLES} samples")
    ax[2].axvline(estimated, color="green", ls=":",
                  label=f"Estimated: {estimated} samples")
    ax[2].set_title("Residual-to-Reference Correlation")
    ax[2].set_xlabel("Lag (samples)")
    ax[2].set_ylabel("Correlation magnitude")
    ax[2].grid(True)
    ax[2].legend()
    fig.savefig("stage29_gnuradio_sample_analysis.png", dpi=160)
    plt.show()

    with open("stage29_delay_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_rate_hz", "samples_analyzed", "expected_delay_samples",
            "estimated_delay_samples", "expected_delay_ms", "estimated_delay_ms",
            "direct_scale_real", "direct_scale_imag", "correlation_peak",
            "peak_to_median_correlation", "delay_matches_expected"
        ])
        writer.writerow([
            SAMPLE_RATE_HZ, len(ref), EXPECTED_DELAY_SAMPLES, estimated,
            expected_ms, estimated_ms, scale.real, scale.imag, peak, ratio,
            estimated == EXPECTED_DELAY_SAMPLES
        ])
    print("\nSaved:")
    print("  stage29_gnuradio_sample_analysis.png")
    print("  stage29_delay_summary.csv")

if __name__ == "__main__":
    main()
