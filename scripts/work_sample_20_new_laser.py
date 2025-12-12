from nanostructure_analysis import *
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import re
import pandas as pd
from scipy.signal import savgol_filter

output_folder = "plots/sample_20_new_laser"
os.makedirs(output_folder, exist_ok=True)


def load_spectra_cached(spectra_path):
    """Load spectra data with caching to avoid long reload times"""
    # Create cache directory
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache filename from path
    cache_name = spectra_path.replace("/", "_").replace("\\", "_").replace(":", "").replace(" ", "_")
    cache_file = os.path.join(cache_dir, f"spectra_{cache_name}.pkl")
    
    # Try loading from cache first
    if os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            print("Cache corrupted, reloading...")
    
    # Load fresh data if no cache
    print(f"Loading spectra from: {spectra_path}")
    spectra_data, spectra_params = spectra_main(spectra_path)
    
    # Save to cache
    print(f"Saving to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump((spectra_data, spectra_params), f)
    
    return spectra_data, spectra_params

path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250925 - Sample 20 new laser"

spectra_data, spectra_params = load_spectra_cached(path)

spectra_data_structure_box1, spectra_params_structure_box1 = filter_spectra(spectra_data, spectra_params, "[A-D]*", average=False, exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*box2*"])
spectra_data_ref, spectra_params_ref = filter_spectra(spectra_data, spectra_params, "*5um_zlocked*", average=False, exclude=["*bias*", "*baseline*"])
spectra_data_fiducial1, spectra_params_fiducial1 = filter_spectra(spectra_data, spectra_params, "*fiducial1*", average=True, exclude=["*bias*", "*baseline*"])
spectra_data_fiducial2, spectra_params_fiducial2 = filter_spectra(spectra_data, spectra_params, "*fiducial2*", average=True, exclude=["*bias*", "*baseline*"])
spectra_data_fiducial3, spectra_params_fiducial3 = filter_spectra(spectra_data, spectra_params, "*fiducial3*", average=True, exclude=["*bias*", "*baseline*"])
# Normalize by maximum value instead of specific wavelength

# Process box1 data - normalize each spectrum by its own maximum
normalized_data_box1 = {}
ref_key = list(spectra_data_ref.keys())[0]

for key in spectra_data_structure_box1.keys():
    spectra = spectra_data_structure_box1[key][:, 1] / spectra_data_ref[ref_key][:, 1]
    #spectra = savgol_filter(spectra, 131, 1)

    # Find maximum value below 940 nm and normalize by it
    wavelengths = spectra_data_structure_box1[key][:, 0]
    mask = wavelengths < 940
    if np.any(mask):
        max_idx = np.argmax(spectra[mask])
        spectra = spectra / spectra[mask][max_idx]
    else:
        # Fallback to overall max if no data below 940 nm
        max_idx = np.argmax(spectra)
        spectra = spectra / spectra[max_idx]

    spectra = np.column_stack((wavelengths, spectra))
    normalized_data_box1[key] = spectra

# Process box2 data - normalize each spectrum by its own maximum
normalized_data_box2 = {}


# Combine both datasets with proper naming
normalized_data_combined = {}
combined_params = {}
for key, value in normalized_data_box1.items():
    normalized_data_combined[f"{key}_box1"] = value
    combined_params[f"{key}_box1"] = spectra_params_structure_box1[key]

# Create custom legend labels
legend_mapping_box1 = {
    'A5_20000ms_1MHz': '40x210 nm',
    'B5_20000ms_1MHz': '40x224 nm', 
    'C5_20000ms_1MHz': '40x245 nm',
    'D5_20000ms_1MHz': '40x280 nm'
}

legend_mapping_box2 = {
    'box2_A5_20000ms_1MHz': '200 nm',
    'box2_B5_20000ms_1MHz': '230 nm',
    'box2_C5_20000ms_1MHz': '260 nm', 
    'box2_D5_20000ms_1MHz': '290 nm'
}

# Create custom data with new keys for box1
custom_data_box1 = {}
custom_params_box1 = {}
for old_key, data in normalized_data_box1.items():
    new_key = legend_mapping_box1.get(old_key, old_key)
    custom_data_box1[new_key] = data
    custom_params_box1[new_key] = spectra_params_structure_box1[old_key]



# Plot box1 only
plot_spectra(custom_data_box1, custom_params_box1, cutoff=940)
plt.savefig(os.path.join(output_folder, "spectra_structure_normalized_box1.png"))
plt.close()

# Plot both together
plot_spectra(normalized_data_combined, combined_params, cutoff=940)
plt.savefig(os.path.join(output_folder, "spectra_structure_normalized_combined.png"))
plt.close()

plot_spectra_savgol(normalized_data_combined, combined_params, window_sizes=[81, 111, 131, 151], orders=[1])
plt.savefig(os.path.join(output_folder, "spectra_structure_normalized_savgol_combined.png"))
plt.close()

plot_spectra(spectra_data_structure_box1, spectra_params_structure_box1, cutoff=940)
plt.savefig(os.path.join(output_folder, "spectra_structure_box1.png"))
plt.close()

plot_spectra(spectra_data_ref, spectra_params_ref, cutoff=940)
plt.savefig(os.path.join(output_folder, "spectra_ref.png"))
plt.close()

plot_spectra(spectra_data_fiducial1, spectra_params_fiducial1, new_fig=True, cutoff=940)
plot_spectra(spectra_data_fiducial2, spectra_params_fiducial2, new_fig=False, cutoff=940)
plot_spectra(spectra_data_fiducial3, spectra_params_fiducial3, new_fig=False, cutoff=940)
plt.savefig(os.path.join(output_folder, "spectra_fiducial.png"))
plt.close()

# Plot normalized by first spectrum (not by reference)
# Get the first spectrum as reference
first_key = list(spectra_data_structure_box1.keys())[0]
first_spectrum = spectra_data_structure_box1[first_key][:, 1]

# Process box1 data - normalize by first spectrum instead of reference
normalized_by_first_box1 = {}
for key in spectra_data_structure_box1.keys():
    spectra = spectra_data_structure_box1[key][:, 1] / first_spectrum
    spectra = savgol_filter(spectra, 131, 1)

    # Find maximum value below 940 nm and normalize by it
    wavelengths = spectra_data_structure_box1[key][:, 0]
    mask = wavelengths < 940
    if np.any(mask):
        max_idx = np.argmax(spectra[mask])
        spectra = spectra / spectra[mask][max_idx]
    else:
        # Fallback to overall max if no data below 940 nm
        max_idx = np.argmax(spectra)
        spectra = spectra / spectra[max_idx]

    spectra = np.column_stack((wavelengths, spectra))
    normalized_by_first_box1[key] = spectra

# Create custom data with new keys
custom_by_first_box1 = {}
custom_by_first_params_box1 = {}
for old_key, data in normalized_by_first_box1.items():
    new_key = legend_mapping_box1.get(old_key, old_key)
    custom_by_first_box1[new_key] = data
    custom_by_first_params_box1[new_key] = spectra_params_structure_box1[old_key]

# Plot normalized by first spectrum
plot_spectra(custom_by_first_box1, custom_by_first_params_box1, cutoff=940)
plt.savefig(os.path.join(output_folder, "spectra_normalized_by_first_spectrum.png"))
plt.close()




