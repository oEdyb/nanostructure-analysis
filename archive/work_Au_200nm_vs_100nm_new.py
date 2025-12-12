# ==============================================================================
# Au 200nm vs 100nm Analysis Script
# ==============================================================================
# This script compares gold nanostructures of 200nm vs 100nm thickness
# Processes APD, confocal, and spectra data for comprehensive analysis

from apd_functions import * 
from spectra_functions import *
from ALL_plotting_functions_OLD import *
from confocal_functions import *
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import re
import pandas as pd

# Global variable to store peak wavelength found from 100nm box1 data
peak_wavelength_100nm_global = None

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

def find_max_in_range(wavelength, intensity, min_wl=700, max_wl=800):
    """Find maximum value in specified wavelength range"""
    mask = (wavelength >= min_wl) & (wavelength <= max_wl)
    if not np.any(mask):
        return 1.0  # Return 1 if no data in range
    return np.max(intensity[mask])

def find_peak_wavelength(wavelength, intensity, min_wl=700, max_wl=800):
    """Find wavelength of peak in specified range"""
    mask = (wavelength >= min_wl) & (wavelength <= max_wl)
    if not np.any(mask):
        # Fallback to overall peak
        peak_idx = np.argmax(intensity)
        return wavelength[peak_idx]
    
    masked_wavelengths = wavelength[mask]
    masked_intensities = intensity[mask]
    peak_idx = np.argmax(masked_intensities)
    return masked_wavelengths[peak_idx]

def normalize_at_peak_wavelength(data_dict, peak_wl):
    """Normalize spectra so that intensity at peak wavelength equals 1"""
    normalized_data = {}
    for key, data in data_dict.items():
        wavelengths = data[:, 0]
        intensities = data[:, 1]
        
        # Find closest wavelength to peak
        closest_idx = np.argmin(np.abs(wavelengths - peak_wl))
        norm_value = intensities[closest_idx]
        
        if norm_value != 0:
            normalized_intensities = intensities / norm_value
        else:
            normalized_intensities = intensities
            print(f"Warning: Zero intensity at peak wavelength for {key}")
            
        normalized_data[key] = np.column_stack((wavelengths, normalized_intensities))
    return normalized_data

def extract_sem_measurements(csv_path, box_filter, gold_type):
    """
    Extract average SEM measurements for a specific box
    
    Args:
        csv_path: Path to the SEM measurements CSV file
        box_filter: String filter for box (e.g., "box1", "box4")
        gold_type: String describing the sample (e.g., "100nm", "200nm")
    
    Returns:
        Dictionary with average measurements
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Filter out tilted measurements and select specific box, only D1-D6 measurements
        df_filtered = df[
            (~df['label'].str.contains('tilted', na=False)) &  # Exclude tilted
            (df['label'].str.contains(box_filter, na=False)) &  # Include specific box
            (df['label'].str.contains(r'D[1-6]', regex=True, na=False))  # Include only D1-D6 pattern
        ]
        
        if len(df_filtered) == 0:
            print(f"Warning: No measurements found for {box_filter} in {csv_path}")
            return None
        
        # Calculate averages
        avg_measurements = {
            'interhole_distance': df_filtered['interhole_distance'].mean(),
            'gap_width': df_filtered['gap_width'].mean(),
            'avg_radius': (df_filtered['radius_top'] + df_filtered['radius_bottom']).mean() / 2,
            'n_measurements': len(df_filtered)
        }
        
        print(f"{gold_type} {box_filter}: {avg_measurements['n_measurements']} measurements")
        print(f"  Avg interhole distance: {avg_measurements['interhole_distance']:.1f} nm")
        print(f"  Avg gap width: {avg_measurements['gap_width']:.1f} nm") 
        print(f"  Avg radius: {avg_measurements['avg_radius']:.1f} nm")
        
        return avg_measurements
        
    except Exception as e:
        print(f"Error reading SEM data from {csv_path}: {e}")
        return None

def normalize_spectra_with_100nm_max(data_dict, bkg_data, ref_data, ref_bkg_data, 
                                   norm_100nm_data, savgol_before_bkg=False, 
                                   savgol_after_div=True, savgol_after_div_window=31, 
                                   savgol_after_div_order=2, cut_off=950):
    """
    Custom normalization that uses max value in 700-800 nm range from 100nm gold data
    """
    # First, find the normalization factor from 100nm data
    norm_factor = None
    if norm_100nm_data:
        # Average all 100nm spectra and find max in 700-800 nm range
        wavelengths = []
        intensities = []
        for key in norm_100nm_data:
            wl = norm_100nm_data[key][:, 0]
            intensity = norm_100nm_data[key][:, 1]
            wavelengths.append(wl)
            intensities.append(intensity)
        
        # Use first spectrum's wavelength as reference
        avg_wavelength = wavelengths[0]
        avg_intensity = np.mean(intensities, axis=0)
        norm_factor = find_max_in_range(avg_wavelength, avg_intensity, 700, 800)
        print(f"Using normalization factor from 100nm gold (700-800 nm max): {norm_factor:.3f}")
    
    if norm_factor is None or norm_factor == 0:
        norm_factor = 1.0
        print("Warning: Could not determine normalization factor, using 1.0")
    
    # Now normalize all data using this factor
    normalized_data = {}
    for key_data in data_dict.keys():
        # Create copies to avoid modifying original data
        data = data_dict[key_data].copy()
        bkg_copy = bkg_data.copy()
        ref_copy = ref_data.copy()
        ref_bkg_copy = ref_bkg_data.copy()
        
        if savgol_before_bkg:
            from scipy.signal import savgol_filter
            data[:, 1] = savgol_filter(data[:, 1], 21, 4)
            bkg_copy[:, 1] = savgol_filter(bkg_copy[:, 1], 21, 4)
            ref_copy[:, 1] = savgol_filter(ref_copy[:, 1], 21, 4)
            ref_bkg_copy[:, 1] = savgol_filter(ref_bkg_copy[:, 1], 21, 4)
        
        # Perform normalization calculation
        normalized_data_y = (data[:, 1] - bkg_copy[:, 1]) / (ref_copy[:, 1] - ref_bkg_copy[:, 1])
        normalized_data_x = data[:, 0]
        
        # Apply savgol filter after normalization if requested
        if savgol_after_div:
            from scipy.signal import savgol_filter
            normalized_data_y = savgol_filter(normalized_data_y, savgol_after_div_window, savgol_after_div_order)
        
        # Apply cutoff wavelength filter to remove data below cutoff
        cutoff_mask = normalized_data_x <= cut_off
        normalized_data_x = normalized_data_x[cutoff_mask]
        normalized_data_y = normalized_data_y[cutoff_mask]
        
        # Normalize using the 100nm max factor
        normalized_data_y = normalized_data_y / norm_factor
            
        normalized_data[key_data] = np.column_stack((normalized_data_x, normalized_data_y))
    return normalized_data

def extract_snr_values(results_dict):
    """Extract SNR values from confocal analysis results"""
    snr_values = []
    for key in results_dict:
        if 'snr_3x3' in results_dict[key]:
            snr_values.append(results_dict[key]['snr_3x3'])
    return snr_values

def extract_noise_values(results_dict):
    """Extract noise values (std_3x3) from confocal analysis results"""
    noise_values = []
    for key in results_dict:
        if 'std_3x3' in results_dict[key]:
            noise_values.append(results_dict[key]['std_3x3'])
    return noise_values

if __name__ == "__main__":
    
    # ==============================================================================
    # SETUP AND CONFIGURATION
    # ==============================================================================
    
    # Create output directory for plots
    output_folder = "plots/Au_200nm_vs_100nm_new"
    os.makedirs(output_folder, exist_ok=True)

    # ==============================================================================
    # DATA PATHS CONFIGURATION
    # ==============================================================================
    
    # 100nm samples (Sample 6 and Sample 7)
    sample6_apd_path = r"Data\APD\2025.06.11 - Sample 6 Power Threshold"
    sample7_apd_path = r"Data\APD\2025.06.03 - Power Threshold Sample 7 box 4"
    sample6_confocal_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.06.11 - Sample 6 Power Threshold"
    sample7_confocal_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.06.03 - Power Threshold Sample 7 box 4"
    spectra_100nm_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250811 - sample6"

    # 200nm samples (Sample 13 - normal and high power threshold measurements)
    sample13_apd_path = r"Data\APD\2025.08.21 - Sample 13 Power Threshold"
    sample13_pt_apd_path = r"Data\APD\2025.09.02 - Sample 13 PT high power"
    sample13_confocal_path = r"Data\Confocal\2025.08.21 - Sample 13 Power Threshold box1"
    sample13_pt_confocal_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.09.02 - Sample 13 before after break"
    spectra_200nm_after_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250821 - sample13 - after"
    spectra_200nm_before_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250812 - sample13"

    # ==============================================================================
    # LOAD AND PROCESS 100nm APD DATA
    # ==============================================================================
    print("Loading 100nm APD data...")
    
    # Load Sample 6 APD data (100nm thickness)
    sample6_apd, sample6_monitor, sample6_params = apd_load_main(sample6_apd_path)
    sample6_apd, sample6_monitor, sample6_params = filter_apd(
        sample6_apd, sample6_monitor, sample6_params, "*[A-D][1-6]*"
    )
    
    # Load Sample 7 APD data (100nm thickness) - exclude problematic C4 measurement
    sample7_apd, sample7_monitor, sample7_params = apd_load_main(sample7_apd_path)
    sample7_apd, sample7_monitor, sample7_params = filter_apd(
        sample7_apd, sample7_monitor, sample7_params, "*[A-D][1-6]*", exclude=["C4"]
    )

    # Fix missing power parameters for high power measurements (100mW)
    for key in sample6_params.keys():
        if 'D6' in key:
            sample6_params[key]['power'] = 100.0
    
    for key in sample7_params.keys():
        if 'C6' in key:
            sample7_params[key]['power'] = 100.0

    # ==============================================================================
    # LOAD AND PROCESS 200nm APD DATA  
    # ==============================================================================
    print("Loading 200nm APD data...")
    
    # Load Sample 13 normal power threshold data
    sample13_apd, sample13_monitor, sample13_params = apd_load_main(sample13_apd_path)
    
    # Load Sample 13 high power threshold data 
    sample13_pt_apd, sample13_pt_monitor, sample13_pt_params = apd_load_main(sample13_pt_apd_path)
    
    # Combine normal and high power threshold datasets for comprehensive analysis
    combined_13_apd = {**sample13_apd, **sample13_pt_apd}
    combined_13_monitor = {**sample13_monitor, **sample13_pt_monitor}
    combined_13_params = {**sample13_params, **sample13_pt_params}
    
    # Separate data by box regions for focused analysis
    sample13_box1_apd, sample13_box1_monitor, sample13_box1_params = filter_apd(
        combined_13_apd, combined_13_monitor, combined_13_params, "*box1*"
    )
    sample13_box4_apd, sample13_box4_monitor, sample13_box4_params = filter_apd(
        combined_13_apd, combined_13_monitor, combined_13_params, "*box4*[!_D4_*]*"
    )

    # ==============================================================================
    # LOAD AND PROCESS CONFOCAL DATA
    # ==============================================================================
    print("Loading confocal data...")
    
    # Load 100nm confocal data with caching for faster processing
    sample6_confocal_data = load_with_cache(sample6_confocal_path, confocal_main)
    sample7_confocal_data = load_with_cache(sample7_confocal_path, confocal_main)
    
    # Separate before/after irradiation images for 100nm samples
    sample6_before = filter_confocal(sample6_confocal_data, "*", exclude=["after"])
    sample6_after = filter_confocal(sample6_confocal_data, "*after*")
    sample7_before = filter_confocal(sample7_confocal_data, "*", exclude=["after"])
    sample7_after = filter_confocal(sample7_confocal_data, "*after*")
    
    # Load 200nm confocal data
    sample13_confocal_data = load_with_cache(sample13_confocal_path, confocal_main)
    sample13_pt_confocal_data = load_with_cache(sample13_pt_confocal_path, confocal_main)
    
    # Combine normal and high power threshold confocal datasets
    combined_13_confocal = (
        {**sample13_confocal_data[0], **sample13_pt_confocal_data[0]},  # image_dict
        {**sample13_confocal_data[1], **sample13_pt_confocal_data[1]},  # apd_dict  
        {**sample13_confocal_data[2], **sample13_pt_confocal_data[2]},  # monitor_dict
        {**sample13_confocal_data[3], **sample13_pt_confocal_data[3]},  # xy_dict
        {**sample13_confocal_data[4], **sample13_pt_confocal_data[4]}   # z_dict
    )
    
    # Separate before/after irradiation images for 200nm samples (exclude problematic C2)
    sample13_box1_before = filter_confocal(combined_13_confocal, "*box1*", exclude=["after", "C2"])
    sample13_box1_after = filter_confocal(combined_13_confocal, "*box1*after*", exclude=["C2"])
    sample13_box4_before = filter_confocal(combined_13_confocal, "*box4*", exclude=["after", "C2"])
    sample13_box4_after = filter_confocal(combined_13_confocal, "*box4*after*", exclude=["C2"])

    # ==============================================================================
    # ANALYZE CONFOCAL DATA
    # ==============================================================================
    print("Analyzing confocal data...")
    
    # Analyze 100nm samples - extract SNR, PSF parameters, and intensity statistics
    sample6_results_before = analyze_confocal(sample6_before)
    sample6_results_after = analyze_confocal(sample6_after)
    sample7_results_before = analyze_confocal(sample7_before)
    sample7_results_after = analyze_confocal(sample7_after)
    
    # Analyze 200nm samples - separate by box regions
    sample13_box1_results_before = analyze_confocal(sample13_box1_before)
    sample13_box1_results_after = analyze_confocal(sample13_box1_after)
    sample13_box4_results_before = analyze_confocal(sample13_box4_before)
    sample13_box4_results_after = analyze_confocal(sample13_box4_after)

    # ==============================================================================
    # LOAD AND PROCESS SPECTRA DATA
    # ==============================================================================
    print("Loading and processing spectra data...")
    
    # Load 100nm sample spectra data with caching
    spectra_100nm, params_100nm = load_spectra_cached(spectra_100nm_path)
    
    # Filter and average box1 and box4 spectra for 100nm sample (exclude problematic measurements)
    box1_100nm_spec, _ = filter_spectra(
        spectra_100nm, params_100nm, "*box1*", 
        average=True, exclude=["*C5_2*", "*C4_2*", "*C3_1*"]
    )
    box4_100nm_spec, _ = filter_spectra(
        spectra_100nm, params_100nm, "*box4*", 
        average=True
    )
    
    # Extract reference and background spectra for normalization
    ref_5um_100nm, _ = filter_spectra(spectra_100nm, params_100nm, "*5um*_100ms*")
    bkg_100nm, _ = filter_spectra(spectra_100nm, params_100nm, "*bkg*")
    
    # Normalize 100nm spectra using background subtraction and reference division
    norm_100nm_box1 = normalize_spectra_bkg(
        box1_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], 
        savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1
    )
    norm_100nm_box4 = normalize_spectra_bkg(
        box4_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], 
        savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1
    )
    
    # Load 200nm sample spectra data (before and after measurements) with caching
    spectra_200nm_after, params_200nm_after = load_spectra_cached(spectra_200nm_after_path)
    spectra_200nm_before, params_200nm_before = load_spectra_cached(spectra_200nm_before_path)
    
    # Filter and average box1 and box4 spectra for 200nm samples
    box1_200nm_after_spec, _ = filter_spectra(spectra_200nm_after, params_200nm_after, "*box1*", average=True)
    box1_200nm_before_spec, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*box1*", average=True)
    box4_200nm_before_spec, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*box4*", average=True)
    
    # Extract reference and background spectra for 200nm normalization
    ref_5um_200nm_after, _ = filter_spectra(spectra_200nm_after, params_200nm_after, "*5um*_z_locked*")
    bkg_200nm_after, _ = filter_spectra(spectra_200nm_after, params_200nm_after, "*bkg*")
    ref_5um_200nm_before, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*5um*")
    bkg_200nm_before, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*bkg*")
    
    # Normalize 200nm spectra using appropriate backgrounds and references (only before data)
    norm_200nm_box1_before = normalize_spectra_bkg(
        box1_200nm_before_spec, bkg_200nm_before['bkg_single_track'], 
        ref_5um_200nm_before['5um_ref_single_track_50ms'], bkg_200nm_before['bkg_single_track_50ms'], 
        savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1
    )
    norm_200nm_box4_before = normalize_spectra_bkg(
        box4_200nm_before_spec, bkg_200nm_before['bkg_single_track'], 
        ref_5um_200nm_before['5um_ref_single_track_50ms'], bkg_200nm_before['bkg_single_track_50ms'], 
        savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1
    )

    # ==============================================================================
    # COMPREHENSIVE PEAK NORMALIZATION (matching original analysis approach)
    # ==============================================================================
    print("Finding peak wavelength and performing comprehensive normalization...")
    
    # First create temporary normalization to find peak from 100nm data
    norm_100nm_temp = normalize_spectra_bkg(box1_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], 
                                       savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    
    # Average the box1 data and find the peak wavelength in 700-800 nm range
    if norm_100nm_temp:
        wavelength_100nm_box1 = list(norm_100nm_temp.values())[0][:, 0]
        intensities_100nm_box1 = [norm_100nm_temp[key][:, 1] for key in norm_100nm_temp.keys()]
        averaged_intensity_100nm_box1 = np.mean(intensities_100nm_box1, axis=0)
        
        # Find peak wavelength in 700-800 nm range using the helper function
        peak_wavelength = find_peak_wavelength(wavelength_100nm_box1, averaged_intensity_100nm_box1, 700, 800)
        print(f"100nm Box1 peak wavelength found at: {peak_wavelength:.1f} nm")
        
    else:
        print("Warning: No box1 data found for 100nm sample")
        peak_wavelength = 750.0  # Fallback value
    
    # Store global peak wavelength for use across functions
    peak_wavelength_100nm_global = peak_wavelength
    
    # Now perform the comprehensive normalization using the same approach as original script
    
    # Apply comprehensive normalization to all datasets using consistent method
    def normalize_spectra_no_peak(data_dict, bkg_data, ref_data, ref_bkg_data, savgol_before_bkg=False, 
                                 savgol_after_div=True, savgol_after_div_window=31, savgol_after_div_order=2, cut_off=950):
        from scipy.signal import savgol_filter
        normalized_data = {}
        for key_data in data_dict.keys():
            # Create copies to avoid modifying original data
            data = data_dict[key_data].copy()
            bkg_copy = bkg_data.copy()
            ref_copy = ref_data.copy()
            ref_bkg_copy = ref_bkg_data.copy()
            
            if savgol_before_bkg:
                data[:, 1] = savgol_filter(data[:, 1], 21, 4)
                bkg_copy[:, 1] = savgol_filter(bkg_copy[:, 1], 21, 4)
                ref_copy[:, 1] = savgol_filter(ref_copy[:, 1], 21, 4)
                ref_bkg_copy[:, 1] = savgol_filter(ref_bkg_copy[:, 1], 21, 4)
            
            # Perform normalization calculation
            normalized_data_y = (data[:, 1] - bkg_copy[:, 1]) / (ref_copy[:, 1] - ref_bkg_copy[:, 1])
            normalized_data_x = data[:, 0]
            
            # Apply savgol filter after normalization if requested
            if savgol_after_div:
                normalized_data_y = savgol_filter(normalized_data_y, savgol_after_div_window, savgol_after_div_order)
            
            # Apply cutoff wavelength filter to remove data below cutoff
            cutoff_mask = normalized_data_x <= cut_off
            normalized_data_x = normalized_data_x[cutoff_mask]
            normalized_data_y = normalized_data_y[cutoff_mask]
            
            # NO peak normalization here - we'll do it manually
            normalized_data[key_data] = np.column_stack((normalized_data_x, normalized_data_y))
        return normalized_data
    
    # Get basic normalized spectra (without peak normalization)
    basic_norm_100nm_box1 = normalize_spectra_no_peak(box1_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], 
                                                      savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    basic_norm_100nm_box4 = normalize_spectra_no_peak(box4_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], 
                                                      savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    basic_norm_200nm_box1 = normalize_spectra_no_peak(box1_200nm_before_spec, bkg_200nm_before['bkg_single_track'], 
                                                      ref_5um_200nm_before['5um_ref_single_track_50ms'], bkg_200nm_before['bkg_single_track_50ms'], 
                                                      savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    basic_norm_200nm_box4 = normalize_spectra_no_peak(box4_200nm_before_spec, bkg_200nm_before['bkg_single_track'], 
                                                      ref_5um_200nm_before['5um_ref_single_track_50ms'], bkg_200nm_before['bkg_single_track_50ms'], 
                                                      savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    
    # Now normalize all datasets at the same peak wavelength
    norm_100nm_box1_at_peak = normalize_at_peak_wavelength(basic_norm_100nm_box1, peak_wavelength)
    norm_100nm_box4_at_peak = normalize_at_peak_wavelength(basic_norm_100nm_box4, peak_wavelength)
    norm_200nm_box1_before_at_peak = normalize_at_peak_wavelength(basic_norm_200nm_box1, peak_wavelength)
    norm_200nm_box4_before_at_peak = normalize_at_peak_wavelength(basic_norm_200nm_box4, peak_wavelength)
    
    # Calculate PSF total size for confocal results
    print("Calculating PSF parameters...")
    for results in [sample6_results_before, sample6_results_after, sample7_results_before, sample7_results_after,
                    sample13_box1_results_before, sample13_box1_results_after, sample13_box4_results_before, sample13_box4_results_after]:
        for key in results:
            results[key]['psf_total_size'] = (results[key]['sigma_x'] * results[key]['sigma_y']) ** 0.5 * 100
    
    # Create averaged spectra for comparison (matching original script approach)
    print("Creating averaged spectra...")
    # Average 100nm normalized spectra
    wavelength_100nm = list(norm_100nm_box1_at_peak.values())[0][:, 0]
    intensities_100nm = [norm_100nm_box1_at_peak[key][:, 1] for key in norm_100nm_box1_at_peak.keys()]
    averaged_intensity_100nm = np.mean(intensities_100nm, axis=0)
    averaged_100nm = np.column_stack([wavelength_100nm, averaged_intensity_100nm])
    
    # Average 200nm normalized spectra  
    wavelength_200nm = list(norm_200nm_box1_before_at_peak.values())[0][:, 0]
    intensities_200nm = [norm_200nm_box1_before_at_peak[key][:, 1] for key in norm_200nm_box1_before_at_peak.keys()]
    averaged_intensity_200nm = np.mean(intensities_200nm, axis=0)
    averaged_200nm = np.column_stack([wavelength_200nm, averaged_intensity_200nm])
    
    # ==============================================================================
    # DATA SUMMARY AND COMPLETION
    # ==============================================================================
    
    print("Data processing complete!")
    print(f"Peak wavelength determined: {peak_wavelength:.1f} nm")
    print(f"100nm Box1 spectra count: {len(norm_100nm_box1_at_peak)}")
    print(f"100nm Box4 spectra count: {len(norm_100nm_box4_at_peak)}")
    print(f"200nm Box1 spectra count: {len(norm_200nm_box1_before_at_peak)}")
    print(f"200nm Box4 spectra count: {len(norm_200nm_box4_before_at_peak)}")
    
    # Store all processed data in easily accessible variables for plotting scripts to use
    # Normalized spectral data (at peak wavelength)
    data_100nm_box1_normalized = norm_100nm_box1_at_peak
    data_100nm_box4_normalized = norm_100nm_box4_at_peak
    data_200nm_box1_normalized = norm_200nm_box1_before_at_peak
    data_200nm_box4_normalized = norm_200nm_box4_before_at_peak
    
    # Averaged spectra for comparison
    averaged_100nm_box1 = averaged_100nm
    averaged_200nm_box1 = averaged_200nm
    
    # APD datasets
    apd_100nm_box1 = (sample6_apd, sample6_monitor, sample6_params)
    apd_100nm_box4 = (sample7_apd, sample7_monitor, sample7_params)
    apd_200nm_box1 = (sample13_box1_apd, sample13_box1_monitor, sample13_box1_params)
    apd_200nm_box4 = (sample13_box4_apd, sample13_box4_monitor, sample13_box4_params)
    
    # Confocal datasets with analysis results
    confocal_100nm_box1 = {
        'before': (sample6_before, sample6_results_before),
        'after': (sample6_after, sample6_results_after)
    }
    confocal_100nm_box4 = {
        'before': (sample7_before, sample7_results_before),
        'after': (sample7_after, sample7_results_after)
    }
    confocal_200nm_box1 = {
        'before': (sample13_box1_before, sample13_box1_results_before),
        'after': (sample13_box1_after, sample13_box1_results_after)
    }
    confocal_200nm_box4 = {
        'before': (sample13_box4_before, sample13_box4_results_before),
        'after': (sample13_box4_after, sample13_box4_results_after)
    }
    
    print("All data variables ready for plotting scripts to use!")
    print("Use 'peak_wavelength_100nm_global' for the normalization wavelength")
    print("Data access examples:")
    print("  - data_100nm_box1_normalized: normalized spectra for 100nm box1")
    print("  - confocal_100nm_box1['before']: (confocal_data, analysis_results)")
    print("  - apd_100nm_box1: (apd_data, monitor_data, params)")
    
    # Save processed data for future use
    processed_data = {
        'peak_wavelength': peak_wavelength,
        'spectra': {
            '100nm_box1': data_100nm_box1_normalized,
            '100nm_box4': data_100nm_box4_normalized,
            '200nm_box1': data_200nm_box1_normalized,
            '200nm_box4': data_200nm_box4_normalized
        },
        'averaged_spectra': {
            '100nm': averaged_100nm_box1,
            '200nm': averaged_200nm_box1
        }
    }
    
    # Save to cache for quick access
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "processed_spectral_data.pkl"), 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Processed data saved to: {cache_dir}/processed_spectral_data.pkl")

    