from nanostructure_analysis import *
import os
import pickle
import numpy as np
import re
import pandas as pd

output_folder = r"plots/sample_21_gap_widths"
os.makedirs(output_folder, exist_ok=True)

path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250922 - Sample 21"

spectra_data, spectra_params = load_spectra_cached(path)

spectra_data_structure, spectra_params_structure = filter_spectra(spectra_data, spectra_params, "[A-D]*", average=True, exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*D5*", "*A6*"])
spectra_data_ref, spectra_params_ref = filter_spectra(spectra_data, spectra_params, "ref2*", average=True, exclude=["*bias*", "*baseline*"])

# Divide by reference spectrum and normalize by maximum value below 950 nm
normalized_data = {}
ref_key = list(spectra_data_ref.keys())[0]

# Set target wavelength for finding maximum and normalizing
target_wavelength = 770  # nm - wavelength where we find the maximum value

for key in spectra_data_structure.keys():
    spectra = spectra_data_structure[key][:, 1] / spectra_data_ref[ref_key][:, 1]
    spectra = savgol_filter(spectra, 131, 1)

    # Find maximum value at target wavelength and normalize by it
    wavelengths = spectra_data_structure[key][:, 0]
    
    # Find the index closest to target wavelength
    idx_target = np.argmin(np.abs(wavelengths - target_wavelength))
    
    # Get the maximum value in a small window around target wavelength
    # Use a window of ±5 data points around the target
    window_size = 50
    start_idx = max(0, idx_target - window_size)
    end_idx = min(len(spectra), idx_target + window_size + 1)
    
    # Find maximum in the window and normalize by it
    max_value = np.max(spectra[start_idx:end_idx])
    spectra = spectra / max_value

    spectra = np.column_stack((wavelengths, spectra))
    normalized_data[key] = spectra


plot_spectra(normalized_data, spectra_params_structure, cutoff=950)
plt.savefig(os.path.join(output_folder, f"spectra_structure_normalized.png"))
plt.show()
plt.close()

plot_spectra_savgol(normalized_data, spectra_params_structure, window_sizes=[81, 111, 131, 151], orders=[1])
plt.savefig(os.path.join(output_folder, f"spectra_structure_normalized_savgol.png"))
plt.show()
plt.close()

plot_spectra(spectra_data_structure, spectra_params_structure, cutoff=950)
plt.savefig(os.path.join(output_folder, "spectra_structure.png"))
plt.close()
plot_spectra(spectra_data_ref, spectra_params_ref, cutoff=950)
plt.savefig(os.path.join(output_folder, "spectra_ref.png"))
plt.close()
