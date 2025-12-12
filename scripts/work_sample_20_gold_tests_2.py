from nanostructure_analysis import *
import numpy as np
import matplotlib.pyplot as plt
import os

output_folder = r"plots/sample_20_gold_tests_2"
os.makedirs(output_folder, exist_ok=True)

spectra_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251202 - Sample 20 Gold Far Off"
spectra_path_30um = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251201 - Sample 20 - New Analysis"

# Load data
spectra_data_all, spectra_params_all = spectra_main(spectra_path)
spectra_data_ref, spectra_params_ref = spectra_main(spectra_path_30um)

# Filter sample and reference data
spectra_data_raw, spectra_params_raw = filter_spectra(
    spectra_data_all,
    spectra_params_all,
    "*[A-D][1-6]*",
    average=False,
    exclude=["*bias*", "*baseline*", "*um*", "*_0*", "*_1*", "*moving*"]
)

spectra_data_baseline, spectra_params_baseline = filter_spectra(
    spectra_data_all,
    spectra_params_all,
    "*gold*",
    average=True,
    exclude=["*bias*", "*baseline*", "*1um*", "*other*", "*pos7*"]
)

spectra_data_ref, spectra_params_ref = filter_spectra(
    spectra_data_all,
    spectra_params_all,
    "*30um*",
    average=True,
    exclude=["*bias*", "*baseline*", "*1um*"]
)

# Normalize
spectra_data_baseline_removed = normalize_spectra_baseline(spectra_data_raw, spectra_data_baseline, spectra_data_ref, lam=1e7)

spectra_data_normalized = normalize_spectra(spectra_data_raw, spectra_data_ref)

spectra_data_baseline_normalized = normalize_spectra(spectra_data_baseline, spectra_data_ref, normalize_max=False)

# Plot normalized spectra
plot_spectra(spectra_data_baseline_removed, spectra_params_raw, cutoff=1000, new_fig=True)
plt.savefig(os.path.join(output_folder, "spectra_normalized.png"))
plt.close()

# Plot normalized spectra
plot_spectra(spectra_data_normalized, spectra_params_raw, cutoff=1000, new_fig=True)
plt.savefig(os.path.join(output_folder, "spectra_normalized_without_gold.png"))
plt.close()

# Plot normalized spectra
plot_spectra(spectra_data_baseline, spectra_params_baseline, cutoff=1000, new_fig=True)
plt.savefig(os.path.join(output_folder, "spectra_baseline.png"))
plt.close()


# Plot raw spectra
plot_spectra(spectra_data_raw, spectra_params_raw, cutoff=1000, new_fig=True)
plot_spectra(spectra_data_baseline, spectra_params_baseline, cutoff=1000, new_fig=False, linestyle="--")
plt.savefig(os.path.join(output_folder, "spectra_raw.png"))
plt.close()

# Plot peak wavelength for baseline-subtracted normalized data
plot_lambda_max(spectra_data_baseline_removed, spectra_params_raw, cutoff=1000, new_fig=True)
plt.savefig(os.path.join(output_folder, "lambda_max_baseline_subtracted.png"))
plt.close()

# Plot peak wavelength for normalized data (without baseline subtraction)
plot_lambda_max(spectra_data_normalized, spectra_params_raw, cutoff=1000, new_fig=True)
plt.savefig(os.path.join(output_folder, "lambda_max_normalized.png"))
plt.close()

# Plot normalized spectra
plot_spectra(spectra_data_baseline_normalized, spectra_params_baseline, cutoff=1000, new_fig=True, legend_labels=["Gold"])
plt.savefig(os.path.join(output_folder, "spectra_baseline_norm.png"))
plt.close()
