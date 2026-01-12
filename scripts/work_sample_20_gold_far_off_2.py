from nanostructure_analysis import *
import matplotlib.pyplot as plt
import os

output_folder = "plots/sample_20_gold_far_off_2"
os.makedirs(output_folder, exist_ok=True)

# Z tests paths
z_tests_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251104 - Sample 20 - Z tests"
z_tests_ref_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251104 - Sample 20 - Z tests"

# Newer data paths
newer_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251202 - Sample 20 Gold Far Off"
newer_ref_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251202 - Sample 20 Gold Far Off"
newer_bias_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251202 - Sample 20 Gold Far Off"

# WF No telescope path
wf_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251128 - Sample 20 WF No telescope"

# Load
z_data_all, z_params_all = spectra_main(z_tests_path)
z_ref_all, z_ref_params_all = spectra_main(z_tests_ref_path)
newer_data_all, newer_params_all = spectra_main(newer_path)
newer_ref_all, newer_ref_params_all = spectra_main(newer_ref_path)
newer_bias_all, newer_bias_params_all = spectra_main(newer_bias_path)
wf_data_all, wf_params_all = spectra_main(wf_path)

# Filter
z_data, z_params = filter_spectra(z_data_all, z_params_all, "*B6*", exclude=["*bias*", "*baseline*", "*scan*"], average=False)
z_ref, z_ref_params = filter_spectra(z_ref_all, z_ref_params_all, "*5um*", exclude=["*bias*", "*baseline*", "*scan*"], average=False)
newer_data, newer_params = filter_spectra(newer_data_all, newer_params_all, "*B6*", exclude=["*bias*", "*baseline*", "*scan*"], average=False)
newer_ref, newer_ref_params = filter_spectra(newer_ref_all, newer_ref_params_all, "*30um*", exclude=["*bias*", "*baseline*", "*scan*"], average=False)
newer_bias, newer_bias_params = filter_spectra(newer_bias_all, newer_bias_params_all, "*gold*", exclude=["*bias*", "*baseline*", "*scan*"], average=True)
# Normalize
z_normalized = normalize_spectra(z_data, z_ref, normalize_max=False)
newer_normalized = normalize_spectra(newer_data, newer_ref, bias_dict=newer_bias, normalize_max=False)

# Create labels: _z10 -> 100 nm, _z-40 -> -400 nm, etc.
import re
z_labels = []
for key in z_normalized.keys():
    match = re.search(r'_z(-?\d+)', key)
    if match:
        z_value = int(match.group(1))
        nm_value = z_value * 10
        z_labels.append(f"{nm_value} nm")
    else:
        z_labels.append(key)

# Plot
plot_spectra(z_normalized, z_params, cutoff=1000, new_fig=True, legend_labels=z_labels)
plt.savefig(os.path.join(output_folder, "B6_z_tests.png"))
plt.close()

# Filter WF 270nm structure
wf_data, wf_params = filter_spectra(wf_data_all, wf_params_all, "*270nm*", exclude=["*bias*", "*baseline*", "*scan*"], average=False)

# Normalize with far off 30um (newer_ref)
wf_normalized = normalize_spectra(wf_data, newer_ref, normalize_max=False)

# Create labels: extract um subscripts
wf_labels = []
for key in wf_normalized.keys():
    match = re.search(r'(-?\d+)um', key)
    if match:
        um_value = match.group(1)
        wf_labels.append(f"{um_value} um")
    else:
        wf_labels.append(key)

# Plot
plot_spectra(wf_normalized, wf_params, cutoff=1000, new_fig=True, legend_labels=wf_labels)
plt.savefig(os.path.join(output_folder, "270nm_wf.png"))
plt.close()

# Normalize references to max
z_ref_norm = {key: np.column_stack((data[:, 0], data[:, 1] / np.max(data[:, 1]))) for key, data in z_ref.items()}
newer_ref_norm = {key: np.column_stack((data[:, 0], data[:, 1] / np.max(data[:, 1]))) for key, data in newer_ref.items()}
newer_bias_norm = {key: np.column_stack((data[:, 0], data[:, 1] / np.max(data[:, 1]))) for key, data in newer_bias.items()}

# Plot all references
plot_spectra(z_ref_norm, z_ref_params, cutoff=1000, new_fig=True, legend_labels=["5um (Z tests)"])
plot_spectra(newer_ref_norm, newer_ref_params, cutoff=1000, new_fig=False, legend_labels=["30um (Gold Far Off)"])
plot_spectra(newer_bias_norm, newer_bias_params, cutoff=1000, new_fig=False, legend_labels=["Gold bias"])
plt.savefig(os.path.join(output_folder, "all_references.png"))
plt.close()