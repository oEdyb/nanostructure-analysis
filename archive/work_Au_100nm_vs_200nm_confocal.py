from apd_functions import * 
from ALL_plotting_functions_OLD import *
from confocal_functions import *
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import re
import pandas as pd
import itertools



# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Output folder for saving plots
output_folder = "plots/Au_100nm_vs_200nm_confocal"
os.makedirs(output_folder, exist_ok=True)

# ==============================================================================
# DATA PATHS CONFIGURATION
# ==============================================================================

# Sample 13 data paths
sample13_apd_path = "./Data/APD/2025.08.21 - Sample 13 Power Threshold"
sample13_pt_apd_path = r"\\AMIPC045962\Cache (D)\daily_data\apd_traces\2025.09.02 - Sample 13 PT high power"
sample13_confocal_path = r"Data\Confocal\2025.08.21 - Sample 13 Power Threshold box1"
sample13_pt_confocal_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.09.02 - Sample 13 before after break"

# Sample 6 and 7 data paths
sample6_apd_path = r"Data\APD\2025.06.11 - Sample 6 Power Threshold"
sample7_apd_path = r"Data\APD\2025.06.03 - Power Threshold Sample 7 box 4"
sample6_confocal_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.06.11 - Sample 6 Power Threshold"
sample7_confocal_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.06.03 - Power Threshold Sample 7 box 4"

# ==============================================================================
# SAMPLE 13 DATA PROCESSING
# ==============================================================================

# Load and filter APD data from both normal and PT datasets
sample13_apd_data, sample13_monitor_data, sample13_apd_params = apd_load_main(sample13_apd_path)
sample13_pt_apd_data, sample13_pt_monitor_data, sample13_pt_apd_params = apd_load_main(sample13_pt_apd_path)

# Combine normal and PT APD data
sample13_combined_apd = {**sample13_apd_data, **sample13_pt_apd_data}
sample13_combined_monitor = {**sample13_monitor_data, **sample13_pt_monitor_data}
sample13_combined_params = {**sample13_apd_params, **sample13_pt_apd_params}

# Filter combined datasets for Box1 and Box4
sample13_box1_apd, sample13_box1_monitor, sample13_box1_params = filter_apd(
    sample13_combined_apd, sample13_combined_monitor, sample13_combined_params, "*box1*")
sample13_box4_apd, sample13_box4_monitor, sample13_box4_params = filter_apd(
    sample13_combined_apd, sample13_combined_monitor, sample13_combined_params, "*box4*[!_D4_*]*")

# Plot combined APD traces for Box1 (includes both normal and PT data)
plot_apd(sample13_box1_apd, sample13_box1_monitor, sample13_box1_params, new_fig=True)
plt.savefig(os.path.join(output_folder, "sample13_apd_box1_combined.png"))
plt.close()

plot_apd(sample13_box1_apd, sample13_box1_monitor, sample13_box1_params, new_fig=True, time=30)
plt.savefig(os.path.join(output_folder, "sample13_apd_box1_combined_30s.png"))
plt.close()

# Plot combined APD traces for Box4 (includes both normal and PT data)
plot_apd(sample13_box4_apd, sample13_box4_monitor, sample13_box4_params, new_fig=True)
plt.savefig(os.path.join(output_folder, "sample13_apd_box4_combined.png"))
plt.close()

plot_apd(sample13_box4_apd, sample13_box4_monitor, sample13_box4_params, new_fig=True, time=30)
plt.savefig(os.path.join(output_folder, "sample13_apd_box4_combined_30s.png"))
plt.close()

# Load and combine confocal data from both datasets
sample13_confocal_data = load_with_cache(sample13_confocal_path, confocal_main)
sample13_pt_confocal_data = load_with_cache(sample13_pt_confocal_path, confocal_main)

print(f"Sample 13 PT confocal data: {list(sample13_pt_confocal_data[0].keys())}")

# Combine confocal datasets (merge all dictionaries in the tuple)
sample13_combined_confocal = (
    {**sample13_confocal_data[0], **sample13_pt_confocal_data[0]},  # image_dict
    {**sample13_confocal_data[1], **sample13_pt_confocal_data[1]},  # apd_dict
    {**sample13_confocal_data[2], **sample13_pt_confocal_data[2]},  # monitor_dict
    {**sample13_confocal_data[3], **sample13_pt_confocal_data[3]},  # xy_dict
    {**sample13_confocal_data[4], **sample13_pt_confocal_data[4]}   # z_dict
)

# Filter combined confocal images
sample13_box1_confocal_before = filter_confocal(sample13_combined_confocal, "*box1*", exclude=["after", "C2", "C1"])
sample13_box1_confocal_after = filter_confocal(sample13_combined_confocal, "*box1*after*", exclude=["C2"])
sample13_box4_confocal_before = filter_confocal(sample13_combined_confocal, "*box4*", exclude=["after", "C2"])
sample13_box4_confocal_after = filter_confocal(sample13_combined_confocal, "*box4*after*", exclude=["C2"])

print(f"Sample 13 Box1 confocal before: {list(sample13_box1_confocal_before[0].keys())}")
print(f"Sample 13 Box1 confocal after: {list(sample13_box1_confocal_after[0].keys())}")
print(f"Sample 13 Box4 confocal before: {list(sample13_box4_confocal_before[0].keys())}")
print(f"Sample 13 Box4 confocal after: {list(sample13_box4_confocal_after[0].keys())}")

# Analyze combined confocal data (includes APD trace statistics and max values)
sample13_box1_results_before = analyze_confocal(sample13_box1_confocal_before)
sample13_box1_results_after = analyze_confocal(sample13_box1_confocal_after)
sample13_box4_results_before = analyze_confocal(sample13_box4_confocal_before)
sample13_box4_results_after = analyze_confocal(sample13_box4_confocal_after)



# ==============================================================================
# SAMPLE 6 AND 7 DATA PROCESSING
# ==============================================================================



# Load and filter APD data for Sample 6 (Box 1)
sample6_apd_data, sample6_monitor_data, sample6_apd_params = apd_load_main(sample6_apd_path)
sample6_apd_data, sample6_monitor_data, sample6_apd_params = filter_apd(
    sample6_apd_data, sample6_monitor_data, sample6_apd_params, "*[A-D][1-6]*")

# Load and filter APD data for Sample 7 (Box 4)
sample7_apd_data, sample7_monitor_data, sample7_apd_params = apd_load_main(sample7_apd_path)
sample7_apd_data, sample7_monitor_data, sample7_apd_params = filter_apd(
    sample7_apd_data, sample7_monitor_data, sample7_apd_params, "*[A-D][1-6]*", exclude=["C4"])

# Load confocal data for both samples
sample6_confocal_data = load_with_cache(sample6_confocal_path, confocal_main)
sample7_confocal_data = load_with_cache(sample7_confocal_path, confocal_main)

    # Filter confocal images
sample6_confocal_before = filter_confocal(sample6_confocal_data, "*", exclude=["after"])
sample6_confocal_after = filter_confocal(sample6_confocal_data, "*after*")
sample7_confocal_before = filter_confocal(sample7_confocal_data, "*", exclude=["after"])
sample7_confocal_after = filter_confocal(sample7_confocal_data, "*after*")

# Analyze confocal data (includes APD trace statistics and max values)
sample6_results_before = analyze_confocal(sample6_confocal_before)
sample6_results_after = analyze_confocal(sample6_confocal_after)
sample7_results_before = analyze_confocal(sample7_confocal_before)
sample7_results_after = analyze_confocal(sample7_confocal_after)

print(f"Sample 6 images before: {list(sample6_confocal_before[0].keys())}")
print(f"Sample 6 images after: {list(sample6_confocal_after[0].keys())}")
print(f"Sample 7 images before: {list(sample7_confocal_before[0].keys())}")
print(f"Sample 7 images after: {list(sample7_confocal_after[0].keys())}")
    
    # Fix monitor data for missing power measurements
# D6 Sample 6: set to 100mW equivalent if signal too low
for key in sample6_monitor_data.keys():
    if 'D6' in key and np.mean(sample6_monitor_data[key]) < 0.05:
        sample6_monitor_data[key] = np.full_like(sample6_monitor_data[key], 100 / 50)  # 100mW with 50 power_factor
        sample6_apd_params[key]['power'] = 100.0

# C6 Sample 7: set to 100mW equivalent if signal too low
for key in sample7_monitor_data.keys():
    if 'C6' in key and np.mean(sample7_monitor_data[key]) < 0.05:
        sample7_monitor_data[key] = np.full_like(sample7_monitor_data[key], 100 / 50)  # 100mW with 50 power_factor
        sample7_apd_params[key]['power'] = 100.0

# ==============================================================================
# SNR VALUES FROM ANALYZE_CONFOCAL
# ==============================================================================

print("\n" + "="*80)
print("SNR VALUES FROM ANALYZE_CONFOCAL RESULTS")
print("="*80)

# Helper function to print SNR values for a given result dictionary
def print_snr_results(sample_name, results_dict, condition):
    print(f"\n{sample_name} - {condition}:")
    print("-" * 40)
    if results_dict:
        for key, result in results_dict.items():
            if isinstance(result, dict) and 'snr_3x3' in result:
                print(f"  {key}: SNR = {result['snr_3x3']:.2f}")
    else:
        print("  No data available")

# Print SNR values for Sample 13 Box 1
print_snr_results("Sample 13 Box 1", sample13_box1_results_before, "Before")
print_snr_results("Sample 13 Box 1", sample13_box1_results_after, "After")

# Print SNR values for Sample 13 Box 4
print_snr_results("Sample 13 Box 4", sample13_box4_results_before, "Before")
print_snr_results("Sample 13 Box 4", sample13_box4_results_after, "After")

# Print SNR values for Sample 6
print_snr_results("Sample 6", sample6_results_before, "Before")
print_snr_results("Sample 6", sample6_results_after, "After")

# Print SNR values for Sample 7
print_snr_results("Sample 7", sample7_results_before, "Before")
print_snr_results("Sample 7", sample7_results_after, "After")

print("\n" + "="*80)
print("SNR ANALYSIS COMPLETE")
print("="*80)

# ==============================================================================
# APD PARAMETERS SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("APD PARAMETERS SUMMARY")
print("="*80)

# Helper function to print APD parameters for a given parameter dictionary
def print_apd_params(sample_name, params_dict):
    print(f"\n{sample_name} APD Parameters:")
    print("-" * 40)
    if params_dict:
        for key, params in params_dict.items():
            if isinstance(params, dict):
                print(f"  {key}:")
                for param_name, param_value in params.items():
                    print(f"    {param_name}: {param_value}")
    else:
        print("  No parameters available")

# Print APD parameters for Sample 13
print_apd_params("Sample 13 Combined", sample13_combined_params)
print_apd_params("Sample 13 Box 1", sample13_box1_params)
print_apd_params("Sample 13 Box 4", sample13_box4_params)

# Print APD parameters for Sample 6 and 7
print_apd_params("Sample 6", sample6_apd_params)
print_apd_params("Sample 7", sample7_apd_params)

print("\n" + "="*80)
print("APD PARAMETERS SUMMARY COMPLETE")
print("="*80)

# ==============================================================================
# POWER-SNR LINKED ANALYSIS
# ==============================================================================

print("\n" + "="*80)
print("POWER-SNR LINKED ANALYSIS")
print("="*80)

def link_power_to_snr(snr_results, apd_params):
    """Link power values to SNR results using smart key matching."""
    linked_data = {}

    for snr_key, snr_data in snr_results.items():
        if isinstance(snr_data, dict) and 'snr_3x3' in snr_data:
            # Try different methods to extract identifier (e.g., 'D1', 'C2', etc.)
            identifier = None

            # Method 1: Look for patterns like 'D1', 'C2', etc. using regex
            identifier_match = re.search(r'[A-Z]\d+', snr_key)  # Matches D1, C2, etc.
            if identifier_match:
                identifier = identifier_match.group()
            else:
                # Method 2: Fallback - try splitting by '_' and look for patterns
                parts = snr_key.split('_')
                for part in parts:
                    if re.match(r'[A-Z]\d+', part):  # Check if part matches pattern
                        identifier = part
                        break

            if identifier:
                # Find matching APD parameter key
                for apd_key, apd_data in apd_params.items():
                    if isinstance(apd_data, dict) and identifier in apd_key:
                        power = apd_data.get('power', 0)
                        linked_data[power] = {
                            'snr': snr_data['snr_3x3'],
                            'snr_key': snr_key,
                            'apd_key': apd_key
                        }
                        break

    return linked_data

# Create linked analysis for each sample/condition
def create_power_snr_summary(results_dict, params_dict, sample_name, condition):
    """Create summary linking power to SNR for a given sample/condition."""
    linked = link_power_to_snr(results_dict, params_dict)
    print(f"\n{sample_name} - {condition}:")
    print("-" * 40)
    if linked:
        for power, data in sorted(linked.items()):
            print(f"  Power: {power:.1f} mW → SNR: {data['snr']:.2f} ({data['snr_key']})")
    else:
        print("  No linked data available")

# Link and display for Sample 13
print("\nSAMPLE 13 BOX 1:")
print("="*30)
create_power_snr_summary(sample13_box1_results_before, sample13_box1_params, "Sample 13 Box 1", "Before")
create_power_snr_summary(sample13_box1_results_after, sample13_box1_params, "Sample 13 Box 1", "After")

print("\nSAMPLE 13 BOX 4:")
print("="*30)
create_power_snr_summary(sample13_box4_results_before, sample13_box4_params, "Sample 13 Box 4", "Before")
create_power_snr_summary(sample13_box4_results_after, sample13_box4_params, "Sample 13 Box 4", "After")

print("\nSAMPLE 6:")
print("="*30)
create_power_snr_summary(sample6_results_before, sample6_apd_params, "Sample 6", "Before")
create_power_snr_summary(sample6_results_after, sample6_apd_params, "Sample 6", "After")

print("\nSAMPLE 7:")
print("="*30)
create_power_snr_summary(sample7_results_before, sample7_apd_params, "Sample 7", "Before")
create_power_snr_summary(sample7_results_after, sample7_apd_params, "Sample 7", "After")

print("\n" + "="*80)
print("POWER-SNR LINKED ANALYSIS COMPLETE")
print("="*80)

# ==============================================================================
# POWER-SNR SCATTER PLOT
# ==============================================================================

def plot_power_snr_scatter(output_folder):
    """
    Create scatter plot showing SNR vs power with before/after comparison
    Split into 4 subplots: Sample 13 Box 1, Sample 13 Box 4, Sample 6&7 Box 1, Sample 6&7 Box 4
    With fitted dashed lines for each dataset

    Args:
        output_folder: Directory to save plots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Helper function to add fitted line
    def add_fitted_line(ax, x_data, y_data, color, label):
        if len(x_data) >= 2:  # Need at least 2 points for a line
            try:
                # Fit linear regression
                coeffs = np.polyfit(x_data, y_data, 1)
                x_line = np.linspace(min(x_data), max(x_data), 100)
                y_line = np.polyval(coeffs, x_line)

                # Plot fitted line
                ax.plot(x_line, y_line, color=color, linestyle='--', alpha=0.8, linewidth=2,
                       label=f'{label} (fitted)')

                # Calculate and display R-squared
                y_pred = np.polyval(coeffs, x_data)
                ss_res = np.sum((np.array(y_data) - y_pred)**2)
                ss_tot = np.sum((np.array(y_data) - np.mean(y_data))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                # Add R² text annotation
                ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes,
                       fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            except (np.RankWarning, TypeError):
                pass  # Skip if fitting fails

    # Sample 13 Box 1 data (200nm) - includes both regular and PT data
    box1_200nm_before_powers = [power for power, data in link_power_to_snr(sample13_box1_results_before, sample13_box1_params).items()]
    box1_200nm_before_snr = [data['snr'] for power, data in link_power_to_snr(sample13_box1_results_before, sample13_box1_params).items()]
    box1_200nm_after_powers = [power for power, data in link_power_to_snr(sample13_box1_results_after, sample13_box1_params).items()]
    box1_200nm_after_snr = [data['snr'] for power, data in link_power_to_snr(sample13_box1_results_after, sample13_box1_params).items()]

    # Sample 13 Box 4 data (200nm) - includes both regular and PT data
    box4_200nm_before_powers = [power for power, data in link_power_to_snr(sample13_box4_results_before, sample13_box4_params).items()]
    box4_200nm_before_snr = [data['snr'] for power, data in link_power_to_snr(sample13_box4_results_before, sample13_box4_params).items()]
    box4_200nm_after_powers = [power for power, data in link_power_to_snr(sample13_box4_results_after, sample13_box4_params).items()]
    box4_200nm_after_snr = [data['snr'] for power, data in link_power_to_snr(sample13_box4_results_after, sample13_box4_params).items()]

    # Sample 6&7 Box 1 data (100nm)
    box1_100nm_before_powers = [power for power, data in link_power_to_snr(sample6_results_before, sample6_apd_params).items()]
    box1_100nm_before_snr = [data['snr'] for power, data in link_power_to_snr(sample6_results_before, sample6_apd_params).items()]
    box1_100nm_after_powers = [power for power, data in link_power_to_snr(sample6_results_after, sample6_apd_params).items()]
    box1_100nm_after_snr = [data['snr'] for power, data in link_power_to_snr(sample6_results_after, sample6_apd_params).items()]

    # Sample 6&7 Box 4 data (100nm)
    box4_100nm_before_powers = [power for power, data in link_power_to_snr(sample7_results_before, sample7_apd_params).items()]
    box4_100nm_before_snr = [data['snr'] for power, data in link_power_to_snr(sample7_results_before, sample7_apd_params).items()]
    box4_100nm_after_powers = [power for power, data in link_power_to_snr(sample7_results_after, sample7_apd_params).items()]
    box4_100nm_after_snr = [data['snr'] for power, data in link_power_to_snr(sample7_results_after, sample7_apd_params).items()]

    # Plot Sample 13 Box 1 (200nm) - Top-left
    if box1_200nm_before_powers:
        ax1.scatter(box1_200nm_before_powers, box1_200nm_before_snr, s=100, alpha=0.7,
                   color='dodgerblue', marker='o', edgecolors='black', linewidths=1.5,
                   label='Before')
        add_fitted_line(ax1, box1_200nm_before_powers, box1_200nm_before_snr, 'dodgerblue', 'Before')

    if box1_200nm_after_powers:
        ax1.scatter(box1_200nm_after_powers, box1_200nm_after_snr, s=100, alpha=0.7,
                   color='crimson', marker='s', edgecolors='black', linewidths=1.5,
                   label='After')
        add_fitted_line(ax1, box1_200nm_after_powers, box1_200nm_after_snr, 'crimson', 'After')

    ax1.set_xlabel('Power (mW)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('SNR', fontsize=10, fontweight='bold')
    ax1.set_title('Sample 13 Box 1 (200nm)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=9)

    # Plot Sample 13 Box 4 (200nm) - Top-right
    if box4_200nm_before_powers:
        ax2.scatter(box4_200nm_before_powers, box4_200nm_before_snr, s=100, alpha=0.7,
                   color='dodgerblue', marker='o', edgecolors='black', linewidths=1.5,
                   label='Before')
        add_fitted_line(ax2, box4_200nm_before_powers, box4_200nm_before_snr, 'dodgerblue', 'Before')

    if box4_200nm_after_powers:
        ax2.scatter(box4_200nm_after_powers, box4_200nm_after_snr, s=100, alpha=0.7,
                   color='crimson', marker='s', edgecolors='black', linewidths=1.5,
                   label='After')
        add_fitted_line(ax2, box4_200nm_after_powers, box4_200nm_after_snr, 'crimson', 'After')

    ax2.set_xlabel('Power (mW)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('SNR', fontsize=10, fontweight='bold')
    ax2.set_title('Sample 13 Box 4 (200nm)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=9)

    # Plot Sample 6&7 Box 1 (100nm) - Bottom-left
    if box1_100nm_before_powers:
        ax3.scatter(box1_100nm_before_powers, box1_100nm_before_snr, s=100, alpha=0.7,
                   color='dodgerblue', marker='o', edgecolors='black', linewidths=1.5,
                   label='Before')
        add_fitted_line(ax3, box1_100nm_before_powers, box1_100nm_before_snr, 'dodgerblue', 'Before')

    if box1_100nm_after_powers:
        ax3.scatter(box1_100nm_after_powers, box1_100nm_after_snr, s=100, alpha=0.7,
                   color='crimson', marker='s', edgecolors='black', linewidths=1.5,
                   label='After')
        add_fitted_line(ax3, box1_100nm_after_powers, box1_100nm_after_snr, 'crimson', 'After')

    ax3.set_xlabel('Power (mW)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('SNR', fontsize=10, fontweight='bold')
    ax3.set_title('Sample 6 & 7 Box 1 (100nm)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=9)

    # Plot Sample 6&7 Box 4 (100nm) - Bottom-right
    if box4_100nm_before_powers:
        ax4.scatter(box4_100nm_before_powers, box4_100nm_before_snr, s=100, alpha=0.7,
                   color='dodgerblue', marker='o', edgecolors='black', linewidths=1.5,
                   label='Before')
        add_fitted_line(ax4, box4_100nm_before_powers, box4_100nm_before_snr, 'dodgerblue', 'Before')

    if box4_100nm_after_powers:
        ax4.scatter(box4_100nm_after_powers, box4_100nm_after_snr, s=100, alpha=0.7,
                   color='crimson', marker='s', edgecolors='black', linewidths=1.5,
                   label='After')
        add_fitted_line(ax4, box4_100nm_after_powers, box4_100nm_after_snr, 'crimson', 'After')

    ax4.set_xlabel('Power (mW)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('SNR', fontsize=10, fontweight='bold')
    ax4.set_title('Sample 6 & 7 Box 4 (100nm)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='both', which='major', labelsize=9)

    plt.suptitle('SNR vs Power: Before vs After Irradiation (with Linear Fits)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(os.path.join(output_folder, "power_snr_scatter_before_after.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Create the scatter plot
plot_power_snr_scatter(output_folder)


