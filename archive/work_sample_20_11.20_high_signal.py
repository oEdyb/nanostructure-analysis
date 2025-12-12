from spectra_functions import *
from confocal_functions import *
from sem_functions import *
from spectra_plotting_functions import *
import numpy as np
import matplotlib.pyplot as plt
import os
import re

output_folder = r"plots/sample_20_11.20_high_signal"
os.makedirs(output_folder, exist_ok=True)

def normalize_against_reference(spectra_dict, reference_dict, lam=1e7, p=0.5):
    reference_key = list(reference_dict.keys())[0]
    reference_intensity = reference_dict[reference_key][:, 1]

    normalized_data = {}
    for key in spectra_dict.keys():
        intensity = spectra_dict[key][:, 1] / reference_intensity
        intensity = baseline_als(intensity, lam, p)
        intensity = intensity / np.max(intensity)
        normalized_data[key] = np.column_stack((spectra_dict[key][:, 0], intensity))
    return normalized_data




spectra_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251120 - Sample 20 White Laser Scan - High Signal"
reference_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251120 - Sample 20 White Laser Scan - High Signal"
sem_csv_path = r"Data/SEM/SEM_measurements_20251029_sample_21_gap_widths.csv"
confocal_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\20251129 - Sample 20 White Laser scans - Good Signal"

sensitivity_range = (837, 871)

spectra_data_raw, spectra_params_raw = spectra_main(spectra_path)
spectra_data_ref, spectra_params_ref = spectra_main(reference_path)

spectra_data_raw, spectra_params_raw = filter_spectra(spectra_data_raw, spectra_params_raw, "[A-D]*", average=False, exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*bad*"])
spectra_data_ref, spectra_params_ref = filter_spectra(spectra_data_ref, spectra_params_ref, "*30um*", average=False, exclude=["*bias*", "*baseline*"])

spectra_data = normalize_against_reference(spectra_data_raw, spectra_data_ref)


confocal_data = load_with_cache(confocal_path, confocal_main)
(filtered_image, filtered_apd, filtered_monitor, filtered_xy, filtered_z) = confocal_data
print(filtered_z.keys())
confocal_data_before = filter_confocal(confocal_data, "*variance*", exclude=["after"])
(filtered_image, filtered_apd, filtered_monitor, filtered_xy, filtered_z) = confocal_data_before
print(filtered_z.keys())
confocal_results = analyze_confocal(confocal_data_before)

plot_spectra(spectra_data, spectra_params_raw, cutoff=1000, new_fig=True)
plt.savefig(os.path.join(output_folder, "spectra_data_normalized.png"))
plt.close()

plot_spectra_derivative(spectra_data, spectra_params_raw, cutoff=950, new_fig=True, linestyle='-')
plt.savefig(os.path.join(output_folder, f"spectra_derivative.png"))
plt.close()

plot_integrated_transmission_scatter(spectra_data_raw, cutoff=950, new_fig=True)
plt.savefig(os.path.join(output_folder, "integrated_transmission_histogram.png"))
plt.close()

    
plot_spectra_std_scatter(spectra_data, cutoff=1000, new_fig=True, normalize_by_mean=True)
plt.savefig(os.path.join(output_folder, "spectra_std_scatter.png"))
plt.close()


############################################################################
# Variance Analysis: Peak Value Comparison (50ms vs 200ms) - Z-Scan Data
############################################################################

# Separate 50ms and 200ms measurements (exclude traces)
variance_50ms = {}
variance_200ms = {}

for key in filtered_z.keys():
    if "traces" not in key:  # Only process z-scans, not traces
        if "200ms" in key:
            variance_200ms[key] = filtered_z[key]
        else:
            variance_50ms[key] = filtered_z[key]

# Extract peak values from each z-scan (skip None or empty data)
peak_values_50ms = []
peak_values_200ms = []

for key, z_scan in variance_50ms.items():
    if z_scan is not None and len(z_scan) > 0:
        peak_values_50ms.append(np.max(z_scan))
    else:
        print(f"Warning: Skipping {key} - invalid z-scan data")

for key, z_scan in variance_200ms.items():
    if z_scan is not None and len(z_scan) > 0:
        peak_values_200ms.append(np.max(z_scan))
    else:
        print(f"Warning: Skipping {key} - invalid z-scan data")

# Calculate statistics
print("\n=== Variance Analysis ===")

if len(peak_values_50ms) == 0 and len(peak_values_200ms) == 0:
    print("ERROR: No valid z-scan data found!")
else:
    if len(peak_values_50ms) > 0:
        print(f"\n50ms measurements (n={len(peak_values_50ms)}):")
        print(f"  Mean peak value: {np.mean(peak_values_50ms):.2f}")
        print(f"  Std dev: {np.std(peak_values_50ms):.2f}")
        print(f"  Variance: {np.var(peak_values_50ms):.2f}")
        print(f"  Relative std (CV): {100*np.std(peak_values_50ms)/np.mean(peak_values_50ms):.2f}%")
    else:
        print("\nNo valid 50ms measurements found!")

    if len(peak_values_200ms) > 0:
        print(f"\n200ms measurements (n={len(peak_values_200ms)}):")
        print(f"  Mean peak value: {np.mean(peak_values_200ms):.2f}")
        print(f"  Std dev: {np.std(peak_values_200ms):.2f}")
        print(f"  Variance: {np.var(peak_values_200ms):.2f}")
        print(f"  Relative std (CV): {100*np.std(peak_values_200ms)/np.mean(peak_values_200ms):.2f}%")
    else:
        print("\nNo valid 200ms measurements found!")

# Only create plots if we have valid data
if len(peak_values_50ms) > 0 or len(peak_values_200ms) > 0:

    # Plot 1: Histogram comparison (only if we have data for both)
    if len(peak_values_50ms) > 0 and len(peak_values_200ms) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(peak_values_50ms, bins=15, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(np.mean(peak_values_50ms), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(peak_values_50ms):.1f}')
        axes[0].set_xlabel('Peak Value (counts)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'50ms Integration Time (n={len(peak_values_50ms)})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(peak_values_200ms, bins=15, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(np.mean(peak_values_200ms), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(peak_values_200ms):.1f}')
        axes[1].set_xlabel('Peak Value (counts)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'200ms Integration Time (n={len(peak_values_200ms)})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "variance_peak_histogram.png"), dpi=300)
        plt.close()

    # Plot 2: Scatter plot with index
    if len(peak_values_50ms) > 0 or len(peak_values_200ms) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        if len(peak_values_50ms) > 0:
            ax.scatter(range(len(peak_values_50ms)), peak_values_50ms, alpha=0.6, s=100, color='blue', label=f'50ms (std={np.std(peak_values_50ms):.1f})')
            ax.axhline(np.mean(peak_values_50ms), color='blue', linestyle='--', alpha=0.5, linewidth=2)

        if len(peak_values_200ms) > 0:
            ax.scatter(range(len(peak_values_200ms)), peak_values_200ms, alpha=0.6, s=100, color='green', label=f'200ms (std={np.std(peak_values_200ms):.1f})')
            ax.axhline(np.mean(peak_values_200ms), color='green', linestyle='--', alpha=0.5, linewidth=2)

        ax.set_xlabel('Measurement Index')
        ax.set_ylabel('Peak Value (counts)')
        ax.set_title('Peak Value Variation: 50ms vs 200ms Integration Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "variance_peak_scatter.png"), dpi=300)
        plt.close()

    # Plot 3: Box plot comparison (only if we have data for both)
    if len(peak_values_50ms) > 0 and len(peak_values_200ms) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))

        data_to_plot = [peak_values_50ms, peak_values_200ms]
        bp = ax.boxplot(data_to_plot, labels=['50ms', '200ms'], patch_artist=True, showmeans=True)

        bp['boxes'][0].set_facecolor('blue')
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor('green')
        bp['boxes'][1].set_alpha(0.5)

        ax.set_ylabel('Peak Value (counts)')
        ax.set_xlabel('Integration Time')
        ax.set_title('Peak Value Distribution Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "variance_peak_boxplot.png"), dpi=300)
        plt.close()

    # Plot 4: Relative variance (coefficient of variation)
    if len(peak_values_50ms) > 0 and len(peak_values_200ms) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))

        cv_50ms = 100 * np.std(peak_values_50ms) / np.mean(peak_values_50ms)
        cv_200ms = 100 * np.std(peak_values_200ms) / np.mean(peak_values_200ms)

        categories = ['50ms', '200ms']
        cv_values = [cv_50ms, cv_200ms]
        colors = ['blue', 'green']

        bars = ax.bar(categories, cv_values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Coefficient of Variation (%)')
        ax.set_xlabel('Integration Time')
        ax.set_title('Relative Variance (CV) Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, cv_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "variance_coefficient_comparison.png"), dpi=300)
        plt.close()

    print("\nVariance analysis plots saved!")

############################################################################

