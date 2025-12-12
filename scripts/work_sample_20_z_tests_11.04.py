from nanostructure_analysis import *
from scipy.signal import savgol_filter
import os
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import numpy as np

def baseline_als(y, lam=1e6, p=0.5, niter=10):
    # "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens
    # For smoothing (not baseline removal), use high lambda (1e6) and p=0.5
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


output_folder = "plots/sample_20_z_tests_11.04"
os.makedirs(output_folder, exist_ok=True)
spectra_flag = True
confocal_flag = False

if spectra_flag:
    path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251104 - Sample 20 - Z tests"
    path_ref = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251013 - Sample 5 24 of the same"
    spectra_data_all, spectra_params_all = spectra_main(path)
    spectra_data_ref, spectra_params_ref = spectra_main(path_ref)

    spectra_data, spectra_params= filter_spectra(spectra_data_all, spectra_params_all, "*z40*", average=True, exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*" ])
    spectra_data_ref, spectra_params_ref = filter_spectra(spectra_data_ref, spectra_params_ref, "*20um*", average=False, exclude=["*bias*", "*baseline*"])
    reference_key = list(spectra_data_ref.keys())[0]
    for key in spectra_data.keys():
        spectra_data[key][:, 1] = spectra_data[key][:, 1] 
        #spectra_data[key][:, 1] = baseline_als(spectra_data[key][:, 1], 1e5, 0.5)
        #spectra_data[key][:, 1] = savgol_filter(spectra_data[key][:, 1], 131, 1)
        spectra_data[key][:, 1] = spectra_data[key][:, 1] 
        
    # spectra_data_structure_10um, spectra_params_structure_10um = filter_spectra(spectra_data, spectra_params, "10um*", average=True, exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*D5*", "*A6*"])
    # spectra_data_structure_15um, spectra_params_structure_15um = filter_spectra(spectra_data, spectra_params, "15um*", average=True, exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*D5*", "*A6*"])
    # spectra_data_structure_20um, spectra_params_structure_20um = filter_spectra(spectra_data, spectra_params, "20um*", average=True, exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*D5*", "*A6*"])

    plot_spectra(spectra_data, spectra_params, cutoff=1000, new_fig=True)
    plt.savefig(os.path.join(output_folder, "spectra_data.png"))
    plt.close()

    plot_spectra(spectra_data_ref, spectra_params_ref, cutoff=1000, new_fig=True)
    plt.savefig(os.path.join(output_folder, "spectra_data_structure_5um.png"))
    plt.close()

    # Extract peak wavelengths for each spectrum
    peak_wavelengths = [data[data[:, 1].argmax(), 0] for data in spectra_data.values()]

    # Plot histogram of peak wavelengths
    plt.figure()
    plt.hist(peak_wavelengths, bins=20, edgecolor='black')
    plt.xlabel('Peak Wavelength (nm)')
    plt.ylabel('Count')
    mean_peak = np.mean(peak_wavelengths)
    std_peak = np.std(peak_wavelengths)
    plt.title(f'Distribution of Peak Wavelengths\nMean: {mean_peak:.1f} nm, Std: {std_peak:.1f} nm')
    plt.savefig(os.path.join(output_folder, "peak_wavelengths_histogram.png"))
    plt.close()

    # Calculate sensitivity (derivative) over wavelength range
    wavelength_range = [837, 871]
    wavelength_sens_data = []  # List to store (wavelength, sensitivity) pairs

    for data in spectra_data.values():
        # Find indices corresponding to the wavelength range
        mask = (data[:, 0] >= wavelength_range[0]) & (data[:, 0] <= wavelength_range[1])
        wavelengths_in_range = data[mask, 0]

        intensities_in_range = data[mask, 1]
        
        # Calculate average wavelength spacing for this spectrum
        # Then divide dI by this average to get dI/dλ
        if len(wavelengths_in_range) > 1:
            avg_dlambda = np.mean(np.diff(wavelengths_in_range))
            sensitivity = np.gradient(intensities_in_range) / avg_dlambda
            # Store wavelength-sensitivity pairs
            for wl, sens in zip(wavelengths_in_range, sensitivity):
                wavelength_sens_data.append((wl, sens))

    # Convert to arrays for plotting
    wavelengths_arr = np.array([x[0] for x in wavelength_sens_data])
    sensitivities_arr = np.array([x[1] for x in wavelength_sens_data])
    sensitivities_arr = sensitivities_arr * 1e3 # Convert to mV/nm
    # Calculate mean sensitivity at each unique wavelength for the trend line
    unique_wavelengths = np.unique(wavelengths_arr)
    mean_sensitivities = [np.mean(sensitivities_arr[wavelengths_arr == wl]) for wl in unique_wavelengths]

    # Plot sensitivity vs wavelength with all points and mean line
    plt.figure(figsize=(10, 6))
    plt.scatter(wavelengths_arr, sensitivities_arr, alpha=0.3, s=10, label='Individual points')
    plt.plot(unique_wavelengths, mean_sensitivities, 'r-', linewidth=2, label='Mean')
    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.ylabel('Sensitivity (mV/nm)', fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)
    mean_sens = np.mean(sensitivities_arr)
    std_sens = np.std(sensitivities_arr)
    plt.title(f'Sensitivity vs Wavelength ({wavelength_range[0]}-{wavelength_range[1]} nm)\nMean: {mean_sens:.4f}, Std: {std_sens:.4f}', fontsize=22, fontweight='bold')
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "sensitivity_vs_wavelength.png"))
    plt.close()

    # Extract intensity values at specific wavelength (852 nm)
    target_wavelength = 852
    intensity_values_at_wl = []

    for data in spectra_data.values():
        # Find the closest wavelength to target
        idx = np.argmin(np.abs(data[:, 0] - target_wavelength))
        intensity_values_at_wl.append(data[idx, 1])

    # Plot histogram of intensity values at target wavelength
    plt.figure(figsize=(10, 6))
    plt.hist(intensity_values_at_wl, bins=7, edgecolor='black')
    plt.xlabel('Intensity', fontsize=18, fontweight='bold')
    plt.ylabel('Count', fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)
    mean_intensity = np.mean(intensity_values_at_wl)
    std_intensity = np.std(intensity_values_at_wl)
    plt.title(f'Distribution of Intensity at {target_wavelength} nm\nMean: {mean_intensity:.4f}, Std: {std_intensity:.4f}', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"intensity_histogram_{target_wavelength}nm.png"))
    plt.close()

    plt.show()
