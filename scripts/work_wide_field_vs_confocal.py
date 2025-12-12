from nanostructure_analysis import *
import numpy as np
import matplotlib.pyplot as plt
import os
import re

output_folder = r"plots/sample_20_wide_field_vs_confocal"
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




spectra_path_confocal = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251120 - Sample 20 White Laser Scan - High Signal"
spectra_path_wide_field = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251126 - Sample 20 Wide Field"

sensitivity_range = (837, 871)

# Load confocal data
print("Loading confocal spectra...")
confocal_data, confocal_params = spectra_main(spectra_path_confocal)

# Load wide field data
print("\nLoading wide field spectra...")
wide_field_data, wide_field_params = spectra_main(spectra_path_wide_field)

# Filter for different spot sizes
print("\n" + "="*50)
print("Filtering for different spot sizes...")
print("="*50)

# Exclusion patterns
exclude_patterns = ["*horizontal*", "*vertical*", "*baseline*", "*bias*"]

# 3um spot size
print("\n3um spot size:")
confocal_3um_data, confocal_3um_params = filter_spectra(confocal_data, confocal_params, "*3um*", exclude=exclude_patterns)
wide_field_3um_data, wide_field_3um_params = filter_spectra(wide_field_data, wide_field_params, "*3um*", exclude=exclude_patterns)

# 5um spot size
print("\n5um spot size:")
confocal_5um_data, confocal_5um_params = filter_spectra(confocal_data, confocal_params, "*5um*", exclude=exclude_patterns)
wide_field_5um_data, wide_field_5um_params = filter_spectra(wide_field_data, wide_field_params, "*5um*", exclude=exclude_patterns)

# 30um spot size
print("\n30um spot size:")
confocal_30um_data, confocal_30um_params = filter_spectra(confocal_data, confocal_params, "*30um*", exclude=exclude_patterns)
wide_field_30um_data, wide_field_30um_params = filter_spectra(wide_field_data, wide_field_params, "*30um*", exclude=exclude_patterns)

# Plot comparisons for each spot size
print("\n" + "="*50)
print("Creating plots...")
print("="*50)

# Get D1 data
print("\nD1 comparison:")
confocal_d1_data, confocal_d1_params = filter_spectra(confocal_data, confocal_params, "*D1*", exclude=exclude_patterns)
wide_field_d1_data, wide_field_d1_params = filter_spectra(wide_field_data, wide_field_params, "*D1*", exclude=exclude_patterns)

# Combined plot with all data - normalized by max
plt.figure(figsize=(18, 10))
colors_confocal = ['blue', 'green', 'red', 'purple']
colors_wide_field = ['cyan', 'lime', 'orange', 'magenta']
labels = ['3um', '5um', '30um', 'D1']
datasets_confocal = [confocal_3um_data, confocal_5um_data, confocal_30um_data, confocal_d1_data]
params_confocal = [confocal_3um_params, confocal_5um_params, confocal_30um_params, confocal_d1_params]
datasets_wide_field = [wide_field_3um_data, wide_field_5um_data, wide_field_30um_data, wide_field_d1_data]
params_wide_field = [wide_field_3um_params, wide_field_5um_params, wide_field_30um_params, wide_field_d1_params]

for i, (conf_data, conf_params, wf_data, wf_params, label, color_conf, color_wf) in enumerate(
    zip(datasets_confocal, params_confocal, datasets_wide_field, params_wide_field, labels, colors_confocal, colors_wide_field)
):
    if conf_data:
        # Take only the first spectrum from the dataset
        key = list(conf_data.keys())[0]
        wavelength = conf_data[key][:, 0]
        intensity = conf_data[key][:, 1]
        # Normalize by max
        intensity = intensity / np.max(intensity)
        mask = wavelength <= 950
        plt.plot(wavelength[mask], intensity[mask], color=color_conf, linestyle='-',
                label=f'Confocal {label}', alpha=0.7, linewidth=2)

    if wf_data:
        # Take only the first spectrum from the dataset
        key = list(wf_data.keys())[0]
        wavelength = wf_data[key][:, 0]
        intensity = wf_data[key][:, 1]
        # Normalize by max
        intensity = intensity / np.max(intensity)
        mask = wavelength <= 950
        plt.plot(wavelength[mask], intensity[mask], color=color_wf, linestyle='--',
                label=f'Wide Field {label}', alpha=0.7, linewidth=2)

plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
plt.ylabel('Normalized Intensity', fontsize=18, fontweight='bold')
plt.title('Confocal vs Wide Field - All Measurements', fontsize=22, fontweight='bold')
plt.grid(True, alpha=0.7, linestyle='--')
plt.legend(fontsize=14, loc='best')
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'all_comparison_normalized.png'), dpi=300, bbox_inches='tight')
print("Saved: all_comparison_normalized.png")

print("\n" + "="*50)
print("All plots saved successfully!")
print(f"Output folder: {output_folder}")
print("="*50)
