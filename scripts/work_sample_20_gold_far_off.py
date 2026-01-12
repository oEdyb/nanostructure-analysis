from nanostructure_analysis import *
import matplotlib.pyplot as plt
import os

output_folder = "plots/sample_20_gold_far_off"
os.makedirs(output_folder, exist_ok=True)

# Z tests paths
z_tests_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251113 - Sample 20 new reference holes"
z_tests_ref_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251113 - Sample 20 new reference holes"

# Newer data paths
newer_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251202 - Sample 20 Gold Far Off"
newer_ref_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251202 - Sample 20 Gold Far Off"
newer_bias_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251202 - Sample 20 Gold Far Off"

# Load
z_data_all, z_params_all = spectra_main(z_tests_path)
z_ref_all, z_ref_params_all = spectra_main(z_tests_ref_path)
newer_data_all, newer_params_all = spectra_main(newer_path)
newer_ref_all, newer_ref_params_all = spectra_main(newer_ref_path)
newer_bias_all, newer_bias_params_all = spectra_main(newer_bias_path)

# Filter
z_data, z_params = filter_spectra(z_data_all, z_params_all, "*A6*zlock*", exclude=["*bias*", "*baseline*", "*scan*"], average=False)
z_ref, z_ref_params = filter_spectra(z_ref_all, z_ref_params_all, "*30um*", exclude=["*bias*", "*baseline*", "*scan*"], average=True)
newer_data, newer_params = filter_spectra(newer_data_all, newer_params_all, "*A6*", exclude=["*bias*", "*baseline*", "*scan*"], average=False)
newer_ref, newer_ref_params = filter_spectra(newer_ref_all, newer_ref_params_all, "*30um*", exclude=["*bias*", "*baseline*", "*scan*"], average=True)
newer_bias, newer_bias_params = filter_spectra(newer_bias_all, newer_bias_params_all, "*gold*", exclude=["*bias*", "*baseline*", "*scan*"], average=True)
# Normalize
z_normalized = normalize_spectra(z_data, z_ref, normalize_max=True)
newer_normalized = normalize_spectra(newer_data, newer_ref, normalize_max=True)

# Plot
plot_spectra(z_normalized, z_params, cutoff=1000, new_fig=True, legend_labels=["Confocal"])
plt.gca().lines[-1].set_color('red')
plot_spectra(newer_normalized, newer_params, cutoff=1000, new_fig=False, legend_labels=["Wide-Field"])
plt.gca().lines[-1].set_color('blue')
plt.savefig(os.path.join(output_folder, "D6_wf_vs_confocal.png"))
plt.show()

# Normalize references to max
z_ref_norm = {key: np.column_stack((data[:, 0], data[:, 1] / np.max(data[:, 1]))) for key, data in z_ref.items()}
newer_ref_norm = {key: np.column_stack((data[:, 0], data[:, 1] / np.max(data[:, 1]))) for key, data in newer_ref.items()}
newer_bias_norm = {key: np.column_stack((data[:, 0], data[:, 1] / np.max(data[:, 1]))) for key, data in newer_bias.items()}

# Plot all references
plot_spectra(z_ref_norm, z_ref_params, cutoff=1000, new_fig=True, legend_labels=["30um (Z tests)"])
plot_spectra(newer_ref_norm, newer_ref_params, cutoff=1000, new_fig=False, legend_labels=["30um (Far Off)"])
plot_spectra(newer_bias_norm, newer_bias_params, cutoff=1000, new_fig=False, legend_labels=["Gold bias"])
plt.savefig(os.path.join(output_folder, "all_references.png"))
plt.show()
