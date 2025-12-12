from apd_functions import * 
from spectra_functions import *
from ALL_plotting_functions_OLD import *
import matplotlib.pyplot as plt
import os
import pickle
from confocal_functions import *
import numpy as np
import pandas as pd

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

def load_simulation_data(csv_path):
    """Load simulation spectra data from CSV file"""
    # Read CSV with semicolon separator
    data = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and ';' in line:
                # Split by semicolon and convert to float
                parts = line.split(';')
                if len(parts) == 2:
                    try:
                        wavelength = float(parts[0].replace(',', '.'))  # Handle European decimal format
                        intensity = float(parts[1].replace(',', '.'))   # Handle European decimal format
                        data.append([wavelength, intensity])
                    except ValueError:
                        continue  # Skip invalid lines
    
    # Convert to numpy arrays
    data = np.array(data)
    wavelengths = data[:, 0]
    intensities = data[:, 1]
    
    # Sort by wavelength for proper plotting
    sort_indices = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_indices]
    intensities = intensities[sort_indices]
    
    return wavelengths, intensities


if __name__ == "__main__":
    # Output directory setup
    output_folder = "plots/Au_100nm"
    os.makedirs(output_folder, exist_ok=True)

    simulation_spectra_path = r"Data\Spectra\michaell_tapered_gap_better_res.csv"

    # Data paths
    box1_apd_path = r"Data\APD\2025.06.11 - Sample 6 Power Threshold"
    box4_apd_path = r"Data\APD\2025.06.03 - Power Threshold Sample 7 box 4"
    box1_confocal_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.06.11 - Sample 6 Power Threshold"
    box4_confocal_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.06.03 - Power Threshold Sample 7 box 4"
    spectra_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250811 - sample6"

    # Load APD data for both boxes
    box1_apd, box1_monitor, box1_params = apd_load_main(box1_apd_path)
    box1_apd, box1_monitor, box1_params = filter_apd(box1_apd, box1_monitor, box1_params, "*[A-D][1-6]*")
    
    box4_apd, box4_monitor, box4_params = apd_load_main(box4_apd_path)
    box4_apd, box4_monitor, box4_params = filter_apd(box4_apd, box4_monitor, box4_params, "*[A-D][1-6]*", exclude=["C4"])

    # Load confocal data for both boxes
    box1_confocal_data = load_with_cache(box1_confocal_path, confocal_main)
    box4_confocal_data = load_with_cache(box4_confocal_path, confocal_main)

    # Filter confocal images
    box1_before = filter_confocal(box1_confocal_data, "*", exclude=["after"])
    box1_after = filter_confocal(box1_confocal_data, "*after*")
    box4_before = filter_confocal(box4_confocal_data, "*", exclude=["after"])
    box4_after = filter_confocal(box4_confocal_data, "*after*")
    
    # Analyze confocal data (now includes APD trace statistics and max values)
    box1_results_before = analyze_confocal(box1_before)
    box1_results_after = analyze_confocal(box1_after)
    box4_results_before = analyze_confocal(box4_before)
    box4_results_after = analyze_confocal(box4_after)
    
    print(f"Box1 images before: {list(box1_before[0].keys())}")
    print(f"Box1 images after: {list(box1_after[0].keys())}")
    print(f"Box4 images before: {list(box4_before[0].keys())}")
    print(f"Box4 images after: {list(box4_after[0].keys())}")

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
                                  savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    norm_box4 = normalize_spectra_bkg(box4_spec, bkg_all['bkg_5000ms'], ref_5um['5um_100ms'], bkg_all['bkg_100ms'], 
                                  savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    
    # Fix monitor data for missing power measurements
    # D6 box1: set to 100mW equivalent if signal too low
    for key in box1_monitor.keys():
        if 'D6' in key and np.mean(box1_monitor[key]) < 0.05:
            box1_monitor[key] = np.full_like(box1_monitor[key], 100 / 50)  # 100mW with 50 power_factor

    # C6 box4: set to 100mW equivalent if signal too low  
    for key in box4_monitor.keys():
        if 'C6' in key and np.mean(box4_monitor[key]) < 0.05:
            box4_monitor[key] = np.full_like(box4_monitor[key], 100 / 50)  # 100mW with 50 power_factor
            
    # Fix power parameter for C6 in box4
    for key in box4_params.keys():
        if 'C6' in key:
            box4_params[key]['power'] = 100.0

        # Fix monitor data for missing power measurements
    # D6 box1: set to 100mW equivalent if signal too low
    for key in box1_monitor.keys():
        if 'D6' in key:
            box1_params[key]['power'] = 100.0

    # Plot APD traces
    plot_apd(box1_apd, box1_monitor, box1_params, new_fig=True)
    plt.savefig(os.path.join(output_folder, "apd_box1.png"))
    plt.close()

    plot_apd(box4_apd, box4_monitor, box4_params, new_fig=True)
    plt.savefig(os.path.join(output_folder, "apd_box4.png"))
    plt.close()

    plot_apd(box1_apd, box1_monitor, box1_params, new_fig=True, time=30)
    plt.savefig(os.path.join(output_folder, "apd_box1_30s.png"))
    plt.close()

    plot_apd_transparent(box1_apd, box1_monitor, box1_params)
    plt.savefig(os.path.join(output_folder, "apd_box1_transparent.png"))
    plt.close()

    # Plot raw spectra
    plot_spectra(box1_spec, params)
    plt.savefig(os.path.join(output_folder, "spectra_box1_raw.png"))
    plt.close()

    plot_spectra(bkg_5000ms, params)
    plt.savefig(os.path.join(output_folder, "spectra_bkg_5000ms.png"))
    plt.close()

    plot_spectra(bkg_100ms, params)
    plt.savefig(os.path.join(output_folder, "spectra_bkg_100ms.png"))
    plt.close()

    plot_spectra(ref_5um, params)
    plt.savefig(os.path.join(output_folder, "spectra_ref_5um.png"))
    plt.close()

    # Plot normalized spectra
    plot_spectra(norm_box1, params)
    plt.savefig(os.path.join(output_folder, "spectra_box1_normalized.png"))
    plt.close()

    plot_spectra(norm_box4, params)
    plt.savefig(os.path.join(output_folder, "spectra_box4_normalized.png"))
    plt.close()
    
    # Plot normalized spectra with simulation data overlay
    sim_wavelengths, sim_intensities = load_simulation_data(simulation_spectra_path)
    
    # Normalize simulation data to 1 at point closest to 740 nm
    closest_740_idx = np.argmin(np.abs(sim_wavelengths - 740))
    norm_factor = sim_intensities[closest_740_idx]
    sim_intensities_normalized = sim_intensities / norm_factor
    
    # Combine box1 and box4 normalized spectra for plotting
    combined_norm = {**norm_box1, **norm_box4}
    
    # Plot experimental spectra using existing function
    plot_spectra(combined_norm, params)
    
    # Get current x-axis limits before adding simulation
    xlim = plt.xlim()
    
    # Add simulation data on top
    plt.plot(sim_wavelengths, sim_intensities_normalized, 'k--', linewidth=4, 
             label='Simulation', alpha=0.9, zorder=10)
    
    # Restore original x-axis limits
    plt.xlim(xlim)
    
    plt.legend()
    plt.savefig(os.path.join(output_folder, "spectra_normalized_with_simulation.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot box1 normalized spectra with simulation data overlay
    plot_spectra(norm_box1, params)
    
    # Get current x-axis limits before adding simulation
    xlim = plt.xlim()
    
    # Add simulation data on top
    plt.plot(sim_wavelengths, sim_intensities_normalized, 'k--', linewidth=4, 
             label='Simulation', alpha=0.9, zorder=10)
    
    # Restore original x-axis limits
    plt.xlim(xlim)
    
    plt.legend()
    plt.savefig(os.path.join(output_folder, "spectra_box1_normalized_with_simulation.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot only C1 and D1 keys with simulation
    c1_d1_data = {}
    for key in combined_norm:
        if 'C1' in key or 'D1' in key:
            c1_d1_data[key] = combined_norm[key]
    
    plot_spectra(c1_d1_data, params)
    
    # Get current x-axis limits before adding simulation
    xlim = plt.xlim()
    
    # Add simulation data on top
    plt.plot(sim_wavelengths, sim_intensities_normalized, 'k--', linewidth=4, 
             label='Simulation', alpha=0.9, zorder=10)
    
    # Restore original x-axis limits
    plt.xlim(xlim)
    
    plt.legend()
    plt.savefig(os.path.join(output_folder, "spectra_C1_D1_with_simulation.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Savgol filter analysis
    plot_spectra_savgol(box1_spec, params, window_sizes=[21], orders=[1, 2, 3, 4, 5])
    plt.savefig(os.path.join(output_folder, "spectra_box1_savgol_analysis.png"))
    plt.close()

    plot_spectra_savgol(bkg_5000ms, params, window_sizes=[21], orders=[1, 2, 3, 4, 5])
    plt.savefig(os.path.join(output_folder, "spectra_bkg_savgol_analysis.png"))
    plt.close()

    # Plot confocal SNR analysis
    plot_confocal_snr(box1_before, box1_results_before, box1_params, label="Before")
    plot_confocal_snr(box1_after, box1_results_after, box1_params, new_fig=False, marker='s', label="After")
    plt.savefig(os.path.join(output_folder, "confocal_snr_box1.png"))
    plt.close()

    plot_confocal_snr(box4_before, box4_results_before, box4_params, label="Before")
    plot_confocal_snr(box4_after, box4_results_after, box4_params, new_fig=False, marker='s', label="After")
    plt.savefig(os.path.join(output_folder, "confocal_snr_box4.png"))
    plt.close()

    # Plot confocal max value scatter analysis
    plot_confocal_scatters(box1_before, box1_after, box1_results_before, box1_results_after, box1_params, label="Box1")
    plot_confocal_scatters(box4_before, box4_after, box4_results_before, box4_results_after, box4_params, new_fig=False, marker='s', label="Box4")
    plt.savefig(os.path.join(output_folder, "confocal_max_value_scatter.png"))
    plt.close()

    # Plot SNR vs power for before and after measurements
    plot_snr_vs_power(box1_before, box1_results_before, box1_params, label="Box1 Before")
    plot_snr_vs_power(box1_after, box1_results_after, box1_params, new_fig=False, marker='s', label="Box1 After")
    plt.savefig(os.path.join(output_folder, "snr_vs_power_box1.png"))
    plt.close()

    plot_snr_vs_power(box4_before, box4_results_before, box4_params, label="Box4 Before")
    plot_snr_vs_power(box4_after, box4_results_after, box4_params, new_fig=False, marker='s', label="Box4 After")
    plt.savefig(os.path.join(output_folder, "snr_vs_power_box4.png"))
    plt.close()

    # Calculate total PSF size and add to results (convert pixels to nm: 2000nm/20px = 100nm/px)
    for results in [box1_results_before, box1_results_after, box4_results_before, box4_results_after]:
        for key in results:
            results[key]['psf_total_size'] = (results[key]['sigma_x'] * results[key]['sigma_y']) ** 0.5 * 100

    # Plot PSF total size vs power
    plot_confocal_parameter_scatter(box1_before, box1_results_before, box1_params, parameter='psf_total_size', label="Box1 Before")
    plot_confocal_parameter_scatter(box1_after, box1_results_after, box1_params, parameter='psf_total_size', new_fig=False, marker='s', label="Box1 After")
    plt.savefig(os.path.join(output_folder, "psf_total_size_vs_power_box1.png"))
    plt.close()

    plot_confocal_parameter_scatter(box4_before, box4_results_before, box4_params, parameter='psf_total_size', label="Box4 Before")
    plot_confocal_parameter_scatter(box4_after, box4_results_after, box4_params, parameter='psf_total_size', new_fig=False, marker='s', label="Box4 After")
    plt.savefig(os.path.join(output_folder, "psf_total_size_vs_power_box4.png"))
    plt.close()

    # SEM analysis plots - separate by box, exclude C1/C2 from box1
    sem_csv_box1_path = r"Data\SEM\SEM_measurements_20250910_sample_6_after_irradiation_and_spectrum.csv"
    sem_csv_box4_path = r"Data\SEM\SEM_measurements_20250613_sample_7_box_4_after_power_threshold.csv"
    combined_params = {**box1_params, **box4_params}
    
    # Box1 plots (excludes C1, C2) - using sample 6 data
    plot_sem_vs_power(sem_csv_box1_path, box1_params, parameter='gap_width', box_filter='Box1')
    plt.savefig(os.path.join(output_folder, "sem_gap_width_vs_power_box1.png"))
    plt.close()

    plot_sem_vs_power(sem_csv_box1_path, box1_params, parameter='interhole_distance', box_filter='Box1')
    plt.savefig(os.path.join(output_folder, "sem_interhole_distance_vs_power_box1.png"))
    plt.close()

    plot_sem_vs_power(sem_csv_box1_path, box1_params, parameter='radius_top', box_filter='Box1')
    plt.savefig(os.path.join(output_folder, "sem_radius_top_vs_power_box1.png"))
    plt.close()

    # Box4 plots - using sample 7 data (specify default_box since this CSV only has box4 data)
    plot_sem_vs_power(sem_csv_box4_path, box4_params, parameter='gap_width', box_filter='Box4', default_box='4')
    plt.savefig(os.path.join(output_folder, "sem_gap_width_vs_power_box4.png"))
    plt.close()

    plot_sem_vs_power(sem_csv_box4_path, box4_params, parameter='interhole_distance', box_filter='Box4', default_box='4')
    plt.savefig(os.path.join(output_folder, "sem_interhole_distance_vs_power_box4.png"))
    plt.close()

    plot_sem_vs_power(sem_csv_box4_path, box4_params, parameter='radius_top', box_filter='Box4', default_box='4')
    plt.savefig(os.path.join(output_folder, "sem_radius_top_vs_power_box4.png"))
    plt.close()


    