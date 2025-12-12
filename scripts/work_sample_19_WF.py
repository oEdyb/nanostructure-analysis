from nanostructure_analysis import *
import numpy as np
import matplotlib.pyplot as plt
import os

output_folder = r"plots/sample_19_WF"
os.makedirs(output_folder, exist_ok=True)

def normalize_against_reference(spectra_dict, reference_dict, lam=1e5, p=0.5):
    reference_key = list(reference_dict.keys())[0]
    reference_intensity = reference_dict[reference_key][:, 1]

    normalized_data = {}
    for key in spectra_dict.keys():
        intensity = spectra_dict[key][:, 1] / reference_intensity
        # intensity = baseline_als(intensity, lam, p)
        intensity = intensity / np.max(intensity)
        normalized_data[key] = np.column_stack((spectra_dict[key][:, 0], intensity))
    return normalized_data

spectra_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251127 - Sample 19 WF"

# Load data
spectra_data_all, spectra_params_all = spectra_main(spectra_path)

# Filter sample and reference data
spectra_data_raw, spectra_params_raw = filter_spectra(
    spectra_data_all,
    spectra_params_all,
    "*[A-D]*",
    average=False,
    exclude=["*bias*", "*baseline*", "*bad*", "*ND*", "*_0*", "*z*"]
)

spectra_data_ND, spectra_params_ND = filter_spectra(
    spectra_data_all,
    spectra_params_all,
    "*[A-D]*ND*",
    average=False,
    exclude=["*bias*", "*baseline*", "*bad*", "*_0*", "*z*"]
)

spectra_data_z, spectra_params_z = filter_spectra(
    spectra_data_all,
    spectra_params_all,
    "*[A-D]*z*",
    average=False,
    exclude=["*bias*", "*baseline*", "*bad*", "*_0*", "*ND*"]
)

spectra_data_white, spectra_params_white = filter_spectra(
    spectra_data_all,
    spectra_params_all,
    "*30um*",
    average=True,
    exclude=["*bias*", "*baseline*"]
)

# Normalize against reference
spectra_data = normalize_against_reference(spectra_data_raw, spectra_data_raw)
spectra_data_ND = normalize_against_reference(spectra_data_ND, spectra_data_white)
spectra_data_z = normalize_against_reference(spectra_data_z, spectra_data_white)

# Normalize reference for plotting
spectra_data_ref_normalized = {}
for key in spectra_data_white.keys():
    intensity = spectra_data_white[key][:, 1]
    intensity = intensity / np.max(intensity)
    spectra_data_ref_normalized[key] = np.column_stack((spectra_data_white[key][:, 0], intensity))

# Plot normalized spectra
plot_spectra(spectra_data, spectra_params_raw, cutoff=1000, new_fig=True)
plt.savefig(os.path.join(output_folder, "spectra_normalized.png"))
plt.close()

# Plot normalized spectra
plot_spectra(spectra_data_ND, spectra_params_ND, cutoff=1000, new_fig=True)
plt.savefig(os.path.join(output_folder, "spectra_normalized_ND.png"))
plt.close()

# Plot normalized spectra
plot_spectra(spectra_data_z, spectra_params_z, cutoff=1000, new_fig=True)
plt.savefig(os.path.join(output_folder, "spectra_normalized_z.png"))
plt.close()

# Plot normalized spectra
plot_spectra(spectra_data_raw, spectra_params_raw, cutoff=1000, new_fig=True)
plt.savefig(os.path.join(output_folder, "spectra_raw.png"))
plt.close()

# Plot reference
plot_spectra(spectra_data_ref_normalized, spectra_params_white, cutoff=1000, new_fig=True)
plt.savefig(os.path.join(output_folder, "reference.png"))
plt.close()
