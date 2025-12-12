from nanostructure_analysis import *
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import re
import pandas as pd
from scipy.signal import savgol_filter

output_folder = "plots/sample_19_different_hole_sizes"
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

spectra_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250811 - sample6"

# Load and filter spectra data
spectra, params = load_spectra_cached(spectra_path)
box1_spec, _ = filter_spectra(spectra, params, "*box1*", average=True, exclude=["*C5_2*", "*C4_2*", "*C3_1*"])
box4_spec, _ = filter_spectra(spectra, params, "*box4*", average=True)
ref_5um, _ = filter_spectra(spectra, params, "*5um*_100ms*")
bkg_all, _ = filter_spectra(spectra, params, "*bkg*")
bkg_5000ms, _ = filter_spectra(spectra, params, "*bkg*_5000ms*")
bkg_100ms, _ = filter_spectra(spectra, params, "*bkg*_100ms*")

# Normalize spectra with specific backgrounds and references
norm_box1 = normalize_spectra_bkg(box1_spec, bkg_all['bkg_5000ms'], ref_5um['5um_100ms'], bkg_all['bkg_100ms'], 
                                savgol_before_bkg=False, savgol_after_div=False, savgol_after_div_window=131, savgol_after_div_order=1)
                                
path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250916 - Sample 19"

spectra_data, spectra_params = load_spectra_cached(path)

spectra_data_structure, spectra_params_structure = filter_spectra(spectra_data, spectra_params, "[A-D]*", average=True, exclude=["*bias*", "*baseline*", "*100ms*"])
spectra_data_ref, spectra_params_ref = filter_spectra(spectra_data, spectra_params, "5um*new*", average=True, exclude=["*bias*", "*baseline*"])


# Normalize at 850 nm and divide by reference spectrum

# Divide by reference spectrum and normalize at 850 nm
normalized_data = {}
normalized_data_raw = {}  # Keep raw data for transparent plotting
ref_key = list(spectra_data_ref.keys())[0]
normalize_wavelength = 850  # nm

for key in spectra_data_structure.keys():
    # Raw data (without smoothing)
    spectra_raw = spectra_data_structure[key][:, 1] / spectra_data_ref[ref_key][:, 1]
    wavelengths = spectra_data_structure[key][:, 0]
    idx_850 = np.argmin(np.abs(wavelengths - normalize_wavelength))
    spectra_raw = spectra_raw / spectra_raw[idx_850]
    normalized_data_raw[key] = np.column_stack((wavelengths, spectra_raw))

    # Smoothed data
    spectra = savgol_filter(spectra_raw, 131, 1)
    normalized_data[key] = np.column_stack((wavelengths, spectra))

# Add norm_box1 spectrum to the same plot as sample 19 spectra
if norm_box1:
    first_box1_key = list(norm_box1.keys())[0]
    box1_spectrum = norm_box1[first_box1_key]

    # Re-normalize at 850 nm to match sample 19 normalization
    wavelengths = box1_spectrum[:, 0]
    intensities = box1_spectrum[:, 1]
    idx_850 = np.argmin(np.abs(wavelengths - normalize_wavelength))
    intensities_raw = intensities / intensities[idx_850]

    # Raw data
    normalized_data_raw['Box1 Reference'] = np.column_stack((wavelengths, intensities_raw))

    # Smoothed data
    intensities_smooth = savgol_filter(intensities_raw, 131, 1)
    normalized_data['Box1 Reference'] = np.column_stack((wavelengths, intensities_smooth))

    # Add parameters for the norm_box1 sample - copy from first structure spectrum
    first_structure_key = list(spectra_data_structure.keys())[0]
    spectra_params_structure['Box1 Reference'] = spectra_params_structure[first_structure_key].copy()
    spectra_params_structure['Box1 Reference']['Sample'] = 'Box1 Reference'

# Plot both raw (transparent) and smoothed data
plt.figure(figsize=(16, 8))

# Sort keys based on [A-D][1-6] pattern if it exists
import re
def get_sort_key(key):
    match = re.search(r'[A-D][1-6]', key)
    if match:
        letter_part = match.group()[0]
        number_part = int(match.group()[1])
        return (ord(letter_part), number_part)
    return (999, 999)

sorted_keys = sorted(normalized_data.keys(), key=get_sort_key)
colors = plt.cm.tab10(range(10)) if len(sorted_keys) <= 10 else plt.cm.tab20(range(20))

for i, key in enumerate(sorted_keys):
    # Raw data with transparency (behind)
    wavelength = normalized_data_raw[key][:, 0]
    intensity_raw = normalized_data_raw[key][:, 1]
    mask = wavelength <= 940
    plt.plot(wavelength[mask], intensity_raw[mask],
            alpha=0.3, color=colors[i % len(colors)], linewidth=1)

    # Smoothed data (on top)
    wavelength = normalized_data[key][:, 0]
    intensity_smooth = normalized_data[key][:, 1]
    mask = wavelength <= 940
    plt.plot(wavelength[mask], intensity_smooth[mask],
            label=f"{key}", color=colors[i % len(colors)], linewidth=2)

plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
plt.ylabel('arbitrary units', fontsize=18, fontweight='bold')
plt.title('Spectra Data (Raw + Smoothed)', fontsize=22, fontweight='bold')
plt.grid(True, alpha=0.7, linestyle='--')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, "spectra_structure_normalized_raw_and_smooth.png"))
plt.show()
plt.close()

# Plot with different savgol parameters using the raw data
plot_spectra_savgol(normalized_data_raw, spectra_params_structure, window_sizes=[81, 111, 131, 151], orders=[1])
plt.savefig(os.path.join(output_folder, "spectra_structure_normalized_savgol.png"))
plt.show()
plt.close()

plot_spectra(spectra_data_structure, spectra_params_structure, cutoff=940)
plt.savefig(os.path.join(output_folder, "spectra_structure.png"))
plt.close()
plot_spectra(spectra_data_ref, spectra_params_ref, cutoff=940)
plt.savefig(os.path.join(output_folder, "spectra_ref.png"))
plt.close()

# Create new plot with normalized data using plot_spectra function - no smoothing, just normalized by reference
# Remove Box1 Reference from the plot
normalized_data_raw_no_ref = {k: v for k, v in normalized_data_raw.items() if k != 'Box1 Reference'}
plot_spectra(normalized_data_raw_no_ref, spectra_params_structure, cutoff=940)
plt.savefig(os.path.join(output_folder, "spectra_structure_normalized_reference_only.png"))
plt.close()






