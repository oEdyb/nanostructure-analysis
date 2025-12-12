from spectra_functions import *
from confocal_functions import *
from sem_functions import *
from spectra_plotting_functions import *
import numpy as np
import matplotlib.pyplot as plt
import os
import re

output_folder = r"plots/sample_20_11.10"
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




spectra_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251110 - Sample 20"
reference_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251110 - Sample 20"
sem_csv_path = r"Data/SEM/SEM_measurements_20251029_sample_21_gap_widths.csv"
confocal_path = r"\\AMIPC045962\daily_data\confocal_data\20251024 - Sample 21 Gap Widths 24"

sensitivity_range = (837, 871)

spectra_data_raw, spectra_params_raw = spectra_main(spectra_path)
spectra_data_ref, spectra_params_ref = spectra_main(reference_path)

spectra_data_raw, spectra_params_raw = filter_spectra(spectra_data_raw, spectra_params_raw, "[A-D]*", average=True, exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*bad*"])
spectra_data_ref, spectra_params_ref = filter_spectra(spectra_data_ref, spectra_params_ref, "*5um*", average=False, exclude=["*bias*", "*baseline*"])

spectra_data = normalize_against_reference(spectra_data_raw, spectra_data_ref)

sem_measurements = load_sem_measurements(sem_csv_path)
sem_group_stats = summarize_sem_groups(sem_measurements)

confocal_data = load_with_cache(confocal_path, confocal_main)
confocal_data_before = filter_confocal(confocal_data, "*", exclude=["after"])
confocal_results = analyze_confocal(confocal_data_before)

plot_spectra(spectra_data, spectra_params_raw, cutoff=1000, new_fig=True)
plt.savefig(os.path.join(output_folder, "spectra_data_normalized.png"))
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
