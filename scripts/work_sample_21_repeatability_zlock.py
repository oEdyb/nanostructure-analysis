from nanostructure_analysis import *

# `confocal_functions` imports `ALL_plotting_functions` which defines a different
# `plot_spectra` (without `legend_flag`). Re-import the correct implementation
# from `spectra_plotting_functions` so the function with the `legend_flag`
# parameter is available in this module's namespace.
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

output_folder = r"plots/sample_21_repeatability_zlock"
os.makedirs(output_folder, exist_ok=True)

path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251024 - Sample 21 Pol and Repeatability of DNH"
path_ref = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251013 - Sample 5 24 of the same"

spectra_data, spectra_params = spectra_main(path)
spectra_data_ref, spectra_params_ref = spectra_main(path_ref)

spectra_data, spectra_params = filter_spectra(spectra_data, spectra_params, "[A-D]*", average=False, exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*vertical*", "*horizontal*"])
spectra_data_ref, spectra_params_ref = filter_spectra(spectra_data_ref, spectra_params_ref, "*20um*", average=False, exclude=["*bias*", "*baseline*"])

reference_key = list(spectra_data_ref.keys())[0]
for key in spectra_data.keys():
    spectra_data[key][:, 1] = spectra_data[key][:, 1] / spectra_data_ref[reference_key][:, 1]
    spectra_data[key][:, 1] = baseline_als(spectra_data[key][:, 1], 1e6, 0.5)
    #spectra_data[key][:, 1] = savgol_filter(spectra_data[key][:, 1], 131, 1)
    spectra_data[key][:, 1] = spectra_data[key][:, 1] / np.max(spectra_data[key][:, 1])

plot_spectra_std_scatter(spectra_data, cutoff=1000, new_fig=True)
plt.savefig(os.path.join(output_folder, "spectra_std_scatter.png"))
plt.close()





plot_spectra(spectra_data, spectra_params, cutoff=1000, new_fig=True, legend_flag=False)
plt.savefig(os.path.join(output_folder, "spectra_data.png"))
plt.close()

plot_spectra(spectra_data_ref, spectra_params_ref, cutoff=1000, new_fig=True)
plt.savefig(os.path.join(output_folder, "spectra_data_ref.png"))
plt.close()



# Extract peak wavelengths for each spectrum
peak_wavelengths = [data[data[:, 1].argmax(), 0] for data in spectra_data.values()]

# Plot histogram of peak wavelengths
plt.figure()
plt.hist(peak_wavelengths, bins=20, edgecolor='black')
plt.xlabel('Peak Wavelength (nm)')
plt.ylabel('Count')
mean_peak = np.mean(peak_wavelengths)
std_peak = np.std(peak_wavelengths)
plt.title(f'Distribution of Peak Wavelengths\nMean: {mean_peak:.1f} nm, Std: {std_peak:.1f} nm')
plt.savefig(os.path.join(output_folder, "peak_wavelengths_histogram.png"))
plt.close()




#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################

data_path = r"\\AMIPC045962\daily_data\confocal_data\20251024 - Sample 21 Gap Widths 24"

confocal_data = load_with_cache(data_path, confocal_main)
confocal_data_before = filter_confocal(confocal_data, "*", exclude=["after"])
results_dict_before = analyze_confocal(confocal_data_before)

# Extract SNR values from results
snrs = [v['snr_3x3'] for v in results_dict_before.values() if 'snr_3x3' in v]

# Plot histogram with auto bins
plt.figure()
plt.hist(snrs, bins=5, edgecolor='black')
plt.xlabel('SNR', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title(f'SNR Distribution\nMean: {np.mean(snrs):.2f}, Std: {np.std(snrs):.2f}', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "snr_histogram.png"))
plt.close()

# Extract labels and SNR values for violin plot
import re
snr_data = [(re.search(r'[A-D][1-6]', k).group(), v['snr_3x3']) 
            for k, v in results_dict_before.items() if 'snr_3x3' in v and re.search(r'[A-D][1-6]', k)]
labels, snr_values = zip(*sorted(snr_data))

# Violin plot with better styling
plt.figure(figsize=(10, 8))
violin_parts = plt.violinplot([snr_values], positions=[0], widths=0.7, showmeans=True, showmedians=True)

# Style violin body
for pc in violin_parts['bodies']:
    pc.set_facecolor('dodgerblue')
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')
    pc.set_linewidth(1)

# Style violin elements
for element in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
    if element in violin_parts:
        violin_parts[element].set_color('black')
        violin_parts[element].set_linewidth(1)

# Add scatter points with jitter and labels
x_jitter = np.random.normal(0, 0.04, len(snr_values))
plt.scatter(x_jitter, snr_values, alpha=0.8, s=80, color='dodgerblue', edgecolors='black', linewidths=1.5)

# Add labels beside each jittered point
for x, y, label in zip(x_jitter, snr_values, labels):
    plt.text(x + 0.01, y, label, fontsize=7, va='center', ha='left', alpha=0.8)

plt.ylabel('SNR', fontsize=18, fontweight='bold')
plt.title('SNR Distribution', fontsize=22, fontweight='bold')
plt.xticks([])
plt.tick_params(axis='y', which='major', labelsize=16)
plt.xlim(-0.5, 0.5)

# Add statistics box
mean_val = np.mean(snr_values)
std_val = np.std(snr_values)
y_min, y_max = min(snr_values), max(snr_values)
y_range = y_max - y_min

# Set y-axis limits with extra space for text box
plt.ylim(y_min - y_range * 0.05, y_max + y_range * 0.15)

plt.text(0, y_max + y_range * 0.05, f'μ={mean_val:.2f}\nσ={std_val:.2f}', 
         ha='center', va='bottom', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "snr_violin.png"))
plt.show()
