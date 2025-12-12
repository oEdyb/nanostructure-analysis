import os
import matplotlib.pyplot as plt
import numpy as np
from nanostructure_analysis import *
from nanostructure_analysis.confocal_plotting_functions import calculate_center_snr

output_dir = "plots/100nm_snr_violin"
os.makedirs(output_dir, exist_ok=True)

# Load 100nm confocal data
# Sample 6 is box1, Sample 7 is box4 (implicit from sample, not in filenames)
sample6_confocal = confocal_main(r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.06.11 - Sample 6 Power Threshold")
sample6_box1_before = filter_confocal(sample6_confocal, "*", exclude=["after"])
sample6_box1_after = filter_confocal(sample6_confocal, "*after*")

sample7_confocal = confocal_main(r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.06.03 - Power Threshold Sample 7 box 4")
sample7_box4_before = filter_confocal(sample7_confocal, "*", exclude=["after"])
sample7_box4_after = filter_confocal(sample7_confocal, "*after*")

# Load 200nm confocal data (Sample 13 box1 + box4)
sample13_confocal = confocal_main(r"Data\Confocal\2025.08.21 - Sample 13 Power Threshold box1")
sample13_pt_confocal = confocal_main(r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.09.02 - Sample 13 before after break")

# Combine 200nm confocal datasets
combined_13_confocal = ConfocalData(
    images={**sample13_confocal.images, **sample13_pt_confocal.images},
    apd_traces={**sample13_confocal.apd_traces, **sample13_pt_confocal.apd_traces},
    monitor_traces={**sample13_confocal.monitor_traces, **sample13_pt_confocal.monitor_traces},
    xy_coords={**sample13_confocal.xy_coords, **sample13_pt_confocal.xy_coords},
    z_scans={**sample13_confocal.z_scans, **sample13_pt_confocal.z_scans},
    z_scan_traces={**sample13_confocal.z_scan_traces, **sample13_pt_confocal.z_scan_traces}
)

sample13_box1_before = filter_confocal(combined_13_confocal, "*box1*", exclude=["after", "C2"])
sample13_box1_after = filter_confocal(combined_13_confocal, "*box1*after*", exclude=["C2"])
sample13_box4_before = filter_confocal(combined_13_confocal, "*box4*", exclude=["after", "C2"])
sample13_box4_after = filter_confocal(combined_13_confocal, "*box4*after*", exclude=["C2"])

# Extract SNR values
def get_snr(confocal_data):
    snr_dict = calculate_center_snr(confocal_data.apd_traces)
    return [snr for indices in snr_dict.values() for snr in indices.values()]

# Get SNR for 100nm
data_100nm = [
    get_snr(sample6_box1_before),
    get_snr(sample6_box1_after),
    get_snr(sample7_box4_before),
    get_snr(sample7_box4_after)
]

labels_100nm = ['d_ih=190nm\nBefore', 'd_ih=190nm\nAfter', 'd_ih=220nm\nBefore', 'd_ih=220nm\nAfter']
colors_100nm = ['dodgerblue', 'crimson', 'dodgerblue', 'crimson']

# Get SNR for 200nm
data_200nm = [
    get_snr(sample13_box1_before),
    get_snr(sample13_box1_after),
    get_snr(sample13_box4_before),
    get_snr(sample13_box4_after)
]

labels_200nm = ['d_ih=190nm\nBefore', 'd_ih=190nm\nAfter', 'd_ih=220nm\nBefore', 'd_ih=220nm\nAfter']
colors_200nm = ['forestgreen', 'orange', 'forestgreen', 'orange']

# Create 100nm violin plot
plt.figure(figsize=(12, 8))
violin_parts = plt.violinplot(data_100nm, positions=range(len(labels_100nm)), showmeans=True, showmedians=True)

for i, (pc, color) in enumerate(zip(violin_parts['bodies'], colors_100nm)):
    pc.set_facecolor(color)
    pc.set_alpha(0.8)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)

for element in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
    if element in violin_parts:
        violin_parts[element].set_color('black')
        violin_parts[element].set_linewidth(2)

for i, (snr_vals, color) in enumerate(zip(data_100nm, colors_100nm)):
    x_jitter = np.random.normal(i, 0.05, len(snr_vals))
    plt.scatter(x_jitter, snr_vals, alpha=0.8, s=80, color=color,
               edgecolors='black', linewidths=1.5)

for i, snr_vals in enumerate(data_100nm):
    mean_val, std_val = np.mean(snr_vals), np.std(snr_vals)
    plt.text(i, max(snr_vals) + 0.5, f'μ={mean_val:.1f}\nσ={std_val:.1f}',
            ha='center', va='bottom', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

all_max_100nm = max(max(vals) for vals in data_100nm)
plt.ylim(bottom=0, top=all_max_100nm * 1.15)
plt.xticks(range(len(labels_100nm)), labels_100nm, fontsize=18, fontweight='bold')
plt.ylabel('SNR', fontsize=18, fontweight='bold')
plt.title('100nm SNR Comparison', fontsize=22, fontweight='bold')
plt.grid(True, alpha=0.7, linestyle='--')
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
plt.savefig(f"{output_dir}/snr_violin_100nm.png", dpi=300, bbox_inches='tight')
plt.close()

# Create 200nm violin plot
plt.figure(figsize=(12, 8))
violin_parts = plt.violinplot(data_200nm, positions=range(len(labels_200nm)), showmeans=True, showmedians=True)

for i, (pc, color) in enumerate(zip(violin_parts['bodies'], colors_200nm)):
    pc.set_facecolor(color)
    pc.set_alpha(0.8)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)

for element in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
    if element in violin_parts:
        violin_parts[element].set_color('black')
        violin_parts[element].set_linewidth(2)

for i, (snr_vals, color) in enumerate(zip(data_200nm, colors_200nm)):
    x_jitter = np.random.normal(i, 0.05, len(snr_vals))
    plt.scatter(x_jitter, snr_vals, alpha=0.8, s=80, color=color,
               edgecolors='black', linewidths=1.5)

for i, snr_vals in enumerate(data_200nm):
    mean_val, std_val = np.mean(snr_vals), np.std(snr_vals)
    plt.text(i, max(snr_vals) + 0.5, f'μ={mean_val:.1f}\nσ={std_val:.1f}',
            ha='center', va='bottom', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

all_max_200nm = max(max(vals) for vals in data_200nm)
plt.ylim(bottom=0, top=all_max_200nm * 1.15)
plt.xticks(range(len(labels_200nm)), labels_200nm, fontsize=18, fontweight='bold')
plt.ylabel('SNR', fontsize=18, fontweight='bold')
plt.title('200nm SNR Comparison', fontsize=22, fontweight='bold')
plt.grid(True, alpha=0.7, linestyle='--')
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
plt.savefig(f"{output_dir}/snr_violin_200nm.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"Plots saved to {output_dir}/snr_violin_100nm.png and {output_dir}/snr_violin_200nm.png")
