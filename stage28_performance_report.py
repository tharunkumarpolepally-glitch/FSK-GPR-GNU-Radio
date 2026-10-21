"""
Stage 28: Consolidated performance report for synthetic FSK-GPR stages
Summarizes reference values and the observed results shared from Stages 22–27.

IMPORTANT:
- Stage 22–27 values below are recorded results from the user's console outputs.
- The stage 25–27 runs used different random seeds, so comparisons are illustrative,
  not controlled head-to-head benchmarks.
- This is a report of synthetic-model tests, not measured radar performance.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np

# Reference target values used in the synthetic model
targets = {
    1: {"x_ref_m": 1.500, "depth_ref_m": 0.150},
    2: {"x_ref_m": 3.500, "depth_ref_m": 0.300},
}

# Results transcribed from the console outputs provided for each stage.
# None means the metric was not reported for that stage.
records = [
    # stage, method, noise, peaks, target, x_est, depth_est, rmse_ns, inlier_points
    (22, "Unfiltered noisy; curve fit", 0.015, 199, 1, 1.500, 0.150, 0.050, None),
    (22, "Unfiltered noisy; curve fit", 0.015, 199, 2, 3.499, 0.300, 0.066, None),
    (23, "No filtering; curve fit", 0.015, 199, 1, 1.500, 0.150, 0.051, None),
    (23, "No filtering; curve fit", 0.015, 199, 2, 3.500, 0.300, 0.069, None),
    (23, "No filtering; curve fit", 0.030, 781, 1, 1.500, 0.151, 0.067, None),
    (23, "No filtering; curve fit", 0.030, 781, 2, 3.501, 0.300, 0.234, None),
    (23, "No filtering; curve fit", 0.050, 1053, 1, 1.500, 0.151, 0.097, None),
    (23, "No filtering; curve fit", 0.050, 1053, 2, 3.501, 0.300, 0.377, None),
    (23, "No filtering; curve fit", 0.080, 1095, 1, 1.499, 0.147, 0.104, None),
    (23, "No filtering; curve fit", 0.080, 1095, 2, 3.500, 0.298, 0.410, None),
    (24, "Gaussian filtered", 0.080, 231, 1, 1.501, 0.148, 0.320, None),
    (24, "Gaussian filtered", 0.080, 231, 2, 3.498, 0.295, 0.258, None),
    (25, "No filtering", 0.080, 1102, 1, 1.501, 0.150, 0.104, None),
    (25, "No filtering", 0.080, 1102, 2, 3.500, 0.300, 0.175, None),
    (25, "Gaussian", 0.080, 214, 1, 1.500, 0.149, 0.169, None),
    (25, "Gaussian", 0.080, 214, 2, 3.502, 0.295, 0.316, None),
    (25, "Median", 0.080, 775, 1, 1.501, 0.151, 0.853, None),
    (25, "Median", 0.080, 775, 2, 3.490, 0.294, 0.603, None),
    (26, "Gaussian + curve gate", 0.080, 216, 1, 1.501, 0.149, 0.150, 101),
    (26, "Gaussian + curve gate", 0.080, 216, 2, 3.500, 0.301, 0.512, 100),
    (27, "Gaussian + iterative rejection", 0.080, 220, 1, 1.500, 0.149, 0.086, 97),
    (27, "Gaussian + iterative rejection", 0.080, 220, 2, 3.499, 0.301, 0.156, 87),
]

csv_path = "stage28_performance_summary.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "stage", "method", "noise_std", "detected_peak_count", "target_id",
        "x_reference_m", "x_estimate_m", "x_absolute_error_m",
        "depth_reference_m", "depth_estimate_m", "depth_absolute_error_m",
        "fit_rmse_ns", "inlier_points"
    ])
    for stage, method, noise, peak_count, target_id, x_est, d_est, rmse, inliers in records:
        ref = targets[target_id]
        writer.writerow([
            stage, method, noise, peak_count, target_id,
            ref["x_ref_m"], x_est, abs(x_est-ref["x_ref_m"]),
            ref["depth_ref_m"], d_est, abs(d_est-ref["depth_ref_m"]),
            rmse, "" if inliers is None else inliers
        ])

# Stage 25 filter comparison plot (reported results, same stated noise level;
# different stage runs/random seeds mean illustrative comparison).
methods = ["No filtering", "Gaussian", "Median"]
counts = [1102, 214, 775]
rmse_t1 = [0.104, 0.169, 0.853]
rmse_t2 = [0.175, 0.316, 0.603]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
axes[0].bar(methods, counts)
axes[0].set_title("Stage 25 — Detected peaks")
axes[0].set_ylabel("Peak count")
axes[0].tick_params(axis="x", rotation=15)

x = np.arange(len(methods))
width = 0.34
axes[1].bar(x-width/2, rmse_t1, width, label="Target 1")
axes[1].bar(x+width/2, rmse_t2, width, label="Target 2")
axes[1].set_xticks(x, methods, rotation=15)
axes[1].set_ylabel("Fit RMSE (ns)")
axes[1].set_title("Stage 25 — Fit error")
axes[1].legend()
fig.savefig("stage28_filter_comparison.png", dpi=160)
plt.close(fig)

# Noise sweep plot from Stage 23 (recorded outputs)
noise_levels = np.array([0.015, 0.030, 0.050, 0.080])
peak_counts = np.array([199, 781, 1053, 1095])
rmse_t1_noise = np.array([0.051, 0.067, 0.097, 0.104])
rmse_t2_noise = np.array([0.069, 0.234, 0.377, 0.410])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
axes[0].plot(noise_levels, peak_counts, marker="o")
axes[0].set_title("Stage 23 — Peak count vs noise")
axes[0].set_xlabel("Noise standard deviation")
axes[0].set_ylabel("Peak detections")
axes[0].grid(True)
axes[1].plot(noise_levels, rmse_t1_noise, marker="o", label="Target 1")
axes[1].plot(noise_levels, rmse_t2_noise, marker="o", label="Target 2")
axes[1].set_title("Stage 23 — Fit RMSE vs noise")
axes[1].set_xlabel("Noise standard deviation")
axes[1].set_ylabel("RMSE (ns)")
axes[1].grid(True)
axes[1].legend()
fig.savefig("stage28_noise_sweep_summary.png", dpi=160)
plt.close(fig)

# Final stage 27 summary plot
labels = ["Target 1", "Target 2"]
x_err_mm = [abs(1.500-1.500)*1000, abs(3.499-3.500)*1000]
depth_err_mm = [abs(0.149-0.150)*1000, abs(0.301-0.300)*1000]
rmse_final = [0.086, 0.156]
fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
axes[0].bar(labels, x_err_mm)
axes[0].set_title("Stage 27: |x error|")
axes[0].set_ylabel("mm")
axes[1].bar(labels, depth_err_mm)
axes[1].set_title("Stage 27: |depth error|")
axes[1].set_ylabel("mm")
axes[2].bar(labels, rmse_final)
axes[2].set_title("Stage 27: fit RMSE")
axes[2].set_ylabel("ns")
fig.savefig("stage28_final_estimation_errors.png", dpi=160)
plt.close(fig)

print("===== STAGE 28: PERFORMANCE REPORT =====")
print(f"Rows exported: {len(records)}")
print("Synthetic reference targets:")
print("  Target 1: x=1.500 m, depth=0.150 m")
print("  Target 2: x=3.500 m, depth=0.300 m")
print("\nLatest recorded Stage 27 estimates:")
print("  Target 1: x=1.500 m, depth=0.149 m, RMSE=0.086 ns, retained points=97")
print("  Target 2: x=3.499 m, depth=0.301 m, RMSE=0.156 ns, retained points=87")
print("\nFiles saved:")
print(f"  {csv_path}")
print("  stage28_filter_comparison.png")
print("  stage28_noise_sweep_summary.png")
print("  stage28_final_estimation_errors.png")
print("\nCaution: stages used different random seeds/settings; this is a summary of recorded synthetic results, not a controlled benchmark or real GPR validation.")
