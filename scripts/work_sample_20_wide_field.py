from nanostructure_analysis import *
import numpy as np
import matplotlib.pyplot as plt
import os
import re

output_folder = r"plots/sample_20_wide_field"
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




spectra_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251126 - Sample 20 Wide Field"
reference_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251126 - Sample 20 Wide Field"

sensitivity_range = (837, 871)

spectra_data_raw, spectra_params_raw = spectra_main(spectra_path)
spectra_data_ref, spectra_params_ref = spectra_main(reference_path)

spectra_data_raw, spectra_params_raw = filter_spectra(spectra_data_raw, spectra_params_raw, "*[A-D]*", average=False, exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*bad*", "*270nm*", "*5um*", "*vertical*", "*rod*"])
spectra_data_white, spectra_params_white = filter_spectra(spectra_data_ref, spectra_params_ref, "*3um*", average=True, exclude=["*bias*", "*baseline*", "*270nm*",])

spectra_data = normalize_against_reference(spectra_data_raw, spectra_data_white)

# Normalize reference data
spectra_data_ref_normalized = {}
for key in spectra_data_white.keys():
    intensity = spectra_data_white[key][:, 1]
    intensity = intensity / np.max(intensity)
    spectra_data_ref_normalized[key] = np.column_stack((spectra_data_white[key][:, 0], intensity))

plot_spectra(spectra_data, spectra_params_raw, cutoff=1000, new_fig=True)
plt.savefig(os.path.join(output_folder, "spectra_data_normalized.png"))
plt.close()

plot_spectra(spectra_data_ref_normalized, spectra_params_white, cutoff=1000, new_fig=True)
plt.savefig(os.path.join(output_folder, "spectra_data_normalized_reference.png"))
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

plot_lambda_max(spectra_data, spectra_params_raw, new_fig=True)
plt.savefig(os.path.join(output_folder, "lambda_max_scatter.png"))
plt.close()
