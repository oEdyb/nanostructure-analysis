from spectra_functions import *
from ALL_plotting_functions_OLD import *
from confocal_functions import *
from scipy.signal import savgol_filter
import os
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import numpy as np

def baseline_als(y, lam=1e6, p=0.5, niter=10):
    # "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens
    # For smoothing (not baseline removal), use high lambda (1e6) and p=0.5
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


output_folder = "plots/all_SNR"
os.makedirs(output_folder, exist_ok=True)


# Define samples: (name, data_path)
samples = [
    ("Sample 5", r"\\AMIPC045962\daily_data\confocal_data\20251013 - Sample 5 24 of the same"),
    ("Sample 23", r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\20251009 - Sample 23"),
    ("Sample 21", r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\20250922 - Sample 21"),
    ("Sample 20", r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\20250925 - Sample 20"),
    ("Sample 13", r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.08.21 - Sample 13 after"),
    ("Sample 6", r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.06.11 - Sample 6 Power Threshold"),
    ("Sample 7", r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.06.03 - Power Threshold Sample 7 box 4"),
]

# Process each sample and collect SNR values
all_snr_values = []
sample_names = []

for name, data_path in samples:
    # Load and analyze data
    confocal_data = load_with_cache(data_path, confocal_main)
    
    # Filter data - CUSTOMIZE THIS PER SAMPLE AS NEEDED
    if name == "Sample 5":
        confocal_data_filtered = filter_confocal(confocal_data, "*", exclude=["after"])
    elif name == "Sample 23":
        confocal_data_filtered = filter_confocal(confocal_data, "*z0*", exclude=["1x1"])
    elif name == "Sample 21":
        confocal_data_filtered = filter_confocal(confocal_data, "*", exclude=["after"])
    elif name == "Sample 20":
        confocal_data_filtered = filter_confocal(confocal_data, "[A-D]*", exclude=["after"])
    elif name == "Sample 13":
        confocal_data_filtered = filter_confocal(confocal_data, "*box1*", exclude=["after"])
    elif name == "Sample 6":
        confocal_data_filtered = filter_confocal(confocal_data, "*", exclude=["after"])
    elif name == "Sample 7":
        confocal_data_filtered = filter_confocal(confocal_data, "*", exclude=["after"])
    # Analyze and extract SNR values
    results_dict = analyze_confocal(confocal_data_filtered)
    snr_values = [v['snr_3x3'] for v in results_dict.values() if 'snr_3x3' in v]
    # Store results
    all_snr_values.append(snr_values)
    sample_names.append(name)
    print(f"{name}: {len(snr_values)} measurements, mean SNR = {np.mean(snr_values):.2f}")

# Create multi-violin plot
plt.figure(figsize=(12, 8))
positions = range(len(samples))
violin_parts = plt.violinplot(all_snr_values, positions=positions, widths=0.7, 
                               showmeans=True, showmedians=True)

# Style violin body
for pc in violin_parts['bodies']:
    pc.set_facecolor('dodgerblue')
    pc.set_alpha(0.8)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)

# Style violin elements
for element in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
    if element in violin_parts:
        violin_parts[element].set_color('black')
        violin_parts[element].set_linewidth(2)

# Add scatter points for each sample
for i, snr_values in enumerate(all_snr_values):
    x_jitter = np.random.normal(i, 0.03, len(snr_values))
    plt.scatter(x_jitter, snr_values, alpha=0.6, s=50, color='dodgerblue', 
                edgecolors='black', linewidths=1)

# Labels and styling
plt.xticks(positions, sample_names, fontsize=14)
plt.ylabel('SNR', fontsize=18, fontweight='bold')
plt.title('SNR Distribution Across Samples', fontsize=22, fontweight='bold')
plt.tick_params(axis='y', which='major', labelsize=16)

# Add statistics for each sample
for i, snr_values in enumerate(all_snr_values):
    mean_val = np.mean(snr_values)
    std_val = np.std(snr_values)
    y_pos = max(snr_values) + (max(max(s) for s in all_snr_values) - 
                                 min(min(s) for s in all_snr_values)) * 0.02
    plt.text(i, y_pos, f'μ={mean_val:.1f}\nσ={std_val:.1f}', 
             ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "snr_violin.png"))
plt.show()
