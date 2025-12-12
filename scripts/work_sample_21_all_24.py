from nanostructure_analysis import *

output_folder = r"plots/sample_21_gap_widths_all_24"
os.makedirs(output_folder, exist_ok=True)

def normalize_against_reference(spectra_dict, reference_dict, lam=1e7, p=0.5):
    reference_key = list(reference_dict.keys())[0]
    reference_intensity = reference_dict[reference_key][:, 1]

    normalized_data = {}
    for key in spectra_dict.keys():
        intensity = spectra_dict[key][:, 1] / reference_intensity
        intensity = baseline_als(intensity, lam, p)
        # intensity = intensity / intensity.max()
        normalized_data[key] = np.column_stack((spectra_dict[key][:, 0], intensity))
    return normalized_data




spectra_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251024 - Sample 21 Gap Widths 24"
reference_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251013 - Sample 5 24 of the same"
sem_csv_path = r"Data/SEM/SEM_measurements_20251029_sample_21_gap_widths.csv"
confocal_path = r"\\AMIPC045962\daily_data\confocal_data\20251024 - Sample 21 Gap Widths 24"

sensitivity_range = (837, 871)

spectra_data_raw, spectra_params_raw = spectra_main(spectra_path)
spectra_data_ref, spectra_params_ref = spectra_main(reference_path)

spectra_data_raw, spectra_params_raw = filter_spectra(spectra_data_raw, spectra_params_raw, "[A-D]*", average=True, exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*B4*"])
spectra_data_ref, spectra_params_ref = filter_spectra(spectra_data_ref, spectra_params_ref, "*20um*", average=True, exclude=["*bias*", "*baseline*"])

spectra_data = normalize_against_reference(spectra_data_raw, spectra_data_ref)

sem_measurements = read_sem_measurements(sem_csv_path)
sem_filtered = filter_sem(sem_measurements, "*[A-D]*")
matched_data = match_sem_spectra(sem_filtered, spectra_data)



plot_spectra(spectra_data, spectra_params_raw, cutoff=1000, new_fig=True,
            group_colors=True, sem_data=sem_filtered)
plt.savefig(os.path.join(output_folder, "spectra_data_normalized.png"))
plt.close()

plot_spectra(spectra_data_ref, spectra_params_ref, cutoff=1000, new_fig=True, legend_labels=["Large Aperture (20um)"])
plt.savefig(os.path.join(output_folder, "spectra_data_normalized_Ref.png"))
plt.close()

plot_spectra_derivative(spectra_data, spectra_params_raw, cutoff=950, new_fig=True, linestyle='-', group_colors=True, sem_data=sem_filtered)
plt.savefig(os.path.join(output_folder, f"spectra_derivative.png"))
plt.close()

plot_derivative_wavelength_histogram(spectra_data, spectra_params_raw, wavelength_target=852, new_fig=True)
plt.savefig(os.path.join(output_folder, "derivative_histogram_852nm.png"))
plt.close()

plot_integrated_transmission_scatter(spectra_data_raw, cutoff=950, new_fig=True)
plt.savefig(os.path.join(output_folder, "integrated_transmission_histogram.png"))
plt.show()
plt.close()

    
plot_spectra_std_scatter(spectra_data, cutoff=1000, new_fig=True, normalize_by_mean=True)
plt.savefig(os.path.join(output_folder, "spectra_std_scatter.png"))
plt.close()

plot_group_sensitivity_violin(spectra_data, wavelength_range=sensitivity_range, new_fig=True)
plt.savefig(os.path.join(output_folder, "group_sensitivity_violin.png"))
plt.close()

plot_peak_wavelength_vs_sem(matched_data, 'gap_width', new_fig=True)
plt.savefig(os.path.join(output_folder, "peak_wavelength_vs_gap_width.png"))
plt.close()

plot_derivative_vs_sem(matched_data, 'gap_width', wavelength_range=sensitivity_range, new_fig=True)
plt.savefig(os.path.join(output_folder, "derivative_vs_gap_width.png"))
plt.close()

# Plot individual groups
for group in ['A', 'B', 'C', 'D']:
    # Filter spectra for this group
    spectra_group, params_group = filter_spectra(spectra_data, spectra_params_raw, f"{group}*")

    # Plot with group colors to get the correct color
    plot_spectra(spectra_group, params_group, cutoff=1000, new_fig=True,
                group_colors=True, sem_data=sem_filtered)
    plt.savefig(os.path.join(output_folder, f"spectra_group_{group}.png"))
    plt.close()
