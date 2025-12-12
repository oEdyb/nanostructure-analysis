from nanostructure_analysis import *
# ==============================================================================
# Plot All 4 Spectra: No Max Normalization with Raw Data Background
# ==============================================================================
# This script plots all 4 spectra (100nm Box1/4, 200nm Box1/4) with:
# - Basic normalization only (background subtraction + reference division)
# - Transparent raw data in background
# - Smoothed data as main traces

import matplotlib.pyplot as plt
import numpy as np
import os

# Import required functions from the processing modules
from Au_200nm_vs_100nm_new import load_spectra_cached

def plot_all_4_spectra_no_max_norm():
    """
    Plot all 4 spectra with basic normalization only and raw data background
    """
    print("Creating all 4 spectra plot with no max normalization...")
    
    # Define data paths
    spectra_100nm_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250811 - sample6"
    spectra_200nm_before_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250812 - sample13"
    
    # Load spectra data with caching
    print("Loading 100nm spectra...")
    spectra_100nm, params_100nm = load_spectra_cached(spectra_100nm_path)
    
    print("Loading 200nm spectra...")
    spectra_200nm_before, params_200nm_before = load_spectra_cached(spectra_200nm_before_path)
    
    # Filter and process 100nm data
    print("Processing 100nm data...")
    box1_100nm_spec, _ = filter_spectra(
        spectra_100nm, params_100nm, "*box1*", 
        average=True, exclude=["*C5_2*", "*C4_2*", "*C3_1*"]
    )
    box4_100nm_spec, _ = filter_spectra(
        spectra_100nm, params_100nm, "*box4*", 
        average=True
    )
    
    # Get reference and background for 100nm
    ref_5um_100nm, _ = filter_spectra(spectra_100nm, params_100nm, "*5um*_100ms*")
    bkg_100nm, _ = filter_spectra(spectra_100nm, params_100nm, "*bkg*")
    
    # Filter and process 200nm data
    print("Processing 200nm data...")
    box1_200nm_spec, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*box1*", average=True)
    box4_200nm_spec, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*box4*", average=True)
    
    # Get reference and background for 200nm
    ref_5um_200nm_before, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*5um*")
    bkg_200nm_before, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*bkg*")
    
    # Basic normalization function - no peak normalization
    def basic_normalize_no_peak(data_dict, bkg_data, ref_data, ref_bkg_data, apply_smoothing=True):
        """Basic normalization: (signal - bkg) / (ref - ref_bkg)"""
        from scipy.signal import savgol_filter
        
        normalized_data = {}
        for key_data in data_dict.keys():
            data = data_dict[key_data].copy()
            bkg_copy = bkg_data.copy()
            ref_copy = ref_data.copy()
            ref_bkg_copy = ref_bkg_data.copy()
            
            # Perform normalization calculation
            normalized_data_y = (data[:, 1] - bkg_copy[:, 1]) / (ref_copy[:, 1] - ref_bkg_copy[:, 1])
            normalized_data_x = data[:, 0]
            
            # Apply smoothing only if requested
            if apply_smoothing:
                normalized_data_y = savgol_filter(normalized_data_y, 131, 1)
            
            # Apply cutoff wavelength filter
            cutoff_mask = normalized_data_x <= 950
            normalized_data_x = normalized_data_x[cutoff_mask]
            normalized_data_y = normalized_data_y[cutoff_mask]
            
            normalized_data[key_data] = np.column_stack((normalized_data_x, normalized_data_y))
        return normalized_data
    
    # Process all datasets - both smoothed and raw versions
    print("Normalizing all datasets...")
    
    # 100nm datasets
    norm_100nm_box1_smooth = basic_normalize_no_peak(
        box1_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], True
    )
    norm_100nm_box1_raw = basic_normalize_no_peak(
        box1_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], False
    )
    
    norm_100nm_box4_smooth = basic_normalize_no_peak(
        box4_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], True
    )
    norm_100nm_box4_raw = basic_normalize_no_peak(
        box4_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], False
    )
    
    # 200nm datasets
    norm_200nm_box1_smooth = basic_normalize_no_peak(
        box1_200nm_spec, bkg_200nm_before['bkg_single_track'], 
        ref_5um_200nm_before['5um_ref_single_track_50ms'], bkg_200nm_before['bkg_single_track_50ms'], True
    )
    norm_200nm_box1_raw = basic_normalize_no_peak(
        box1_200nm_spec, bkg_200nm_before['bkg_single_track'], 
        ref_5um_200nm_before['5um_ref_single_track_50ms'], bkg_200nm_before['bkg_single_track_50ms'], False
    )
    
    norm_200nm_box4_smooth = basic_normalize_no_peak(
        box4_200nm_spec, bkg_200nm_before['bkg_single_track'], 
        ref_5um_200nm_before['5um_ref_single_track_50ms'], bkg_200nm_before['bkg_single_track_50ms'], True
    )
    norm_200nm_box4_raw = basic_normalize_no_peak(
        box4_200nm_spec, bkg_200nm_before['bkg_single_track'], 
        ref_5um_200nm_before['5um_ref_single_track_50ms'], bkg_200nm_before['bkg_single_track_50ms'], False
    )
    
    # Average all datasets
    print("Averaging spectra...")
    
    def average_spectra(norm_data):
        """Average multiple spectra into a single trace"""
        if not norm_data:
            return None
        wavelengths = list(norm_data.values())[0][:, 0]
        intensities = [norm_data[key][:, 1] for key in norm_data.keys()]
        averaged_intensity = np.mean(intensities, axis=0)
        return np.column_stack([wavelengths, averaged_intensity])
    
    # Smooth averages
    avg_100nm_box1_smooth = average_spectra(norm_100nm_box1_smooth)
    avg_100nm_box4_smooth = average_spectra(norm_100nm_box4_smooth)
    avg_200nm_box1_smooth = average_spectra(norm_200nm_box1_smooth)
    avg_200nm_box4_smooth = average_spectra(norm_200nm_box4_smooth)
    
    # Raw averages
    avg_100nm_box1_raw = average_spectra(norm_100nm_box1_raw)
    avg_100nm_box4_raw = average_spectra(norm_100nm_box4_raw)
    avg_200nm_box1_raw = average_spectra(norm_200nm_box1_raw)
    avg_200nm_box4_raw = average_spectra(norm_200nm_box4_raw)
    
    # Create the comprehensive plot
    print("Creating plot...")
    plt.figure(figsize=(16, 8))
    
    # Colors: blue family for 100nm, red family for 200nm
    # Line styles: solid for box1, dashed for box4
    
    # Plot transparent background traces (raw data) first
    if avg_100nm_box1_raw is not None:
        plt.plot(avg_100nm_box1_raw[:, 0], avg_100nm_box1_raw[:, 1], 
                 color='dodgerblue', linestyle='-', linewidth=1, alpha=0.3)
    
    if avg_100nm_box4_raw is not None:
        plt.plot(avg_100nm_box4_raw[:, 0], avg_100nm_box4_raw[:, 1], 
                 color='dodgerblue', linestyle='--', linewidth=1, alpha=0.3)
    
    if avg_200nm_box1_raw is not None:
        plt.plot(avg_200nm_box1_raw[:, 0], avg_200nm_box1_raw[:, 1], 
                 color='crimson', linestyle='-', linewidth=1, alpha=0.3)
    
    if avg_200nm_box4_raw is not None:
        plt.plot(avg_200nm_box4_raw[:, 0], avg_200nm_box4_raw[:, 1], 
                 color='crimson', linestyle='--', linewidth=1, alpha=0.3)
    
    # Plot main smoothed traces on top
    if avg_100nm_box1_smooth is not None:
        plt.plot(avg_100nm_box1_smooth[:, 0], avg_100nm_box1_smooth[:, 1], 
                 color='dodgerblue', linestyle='-', linewidth=3, 
                 label='100nm Box1 (d$_{ih}$ = 190 nm)')
    
    if avg_100nm_box4_smooth is not None:
        plt.plot(avg_100nm_box4_smooth[:, 0], avg_100nm_box4_smooth[:, 1], 
                 color='dodgerblue', linestyle='--', linewidth=3,
                 label='100nm Box4 (d$_{ih}$ = 220 nm)')
    
    if avg_200nm_box1_smooth is not None:
        plt.plot(avg_200nm_box1_smooth[:, 0], avg_200nm_box1_smooth[:, 1], 
                 color='crimson', linestyle='-', linewidth=3,
                 label='200nm Box1 (d$_{ih}$ = 190 nm)')
    
    if avg_200nm_box4_smooth is not None:
        plt.plot(avg_200nm_box4_smooth[:, 0], avg_200nm_box4_smooth[:, 1], 
                 color='crimson', linestyle='--', linewidth=3,
                 label='200nm Box4 (d$_{ih}$ = 220 nm)')
    
    # Styling
    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.ylabel('Normalized Intensity (Absolute)', fontsize=18, fontweight='bold')
    plt.title('100 vs 200nm: Box1 vs Box4 [No Max Norm]', 
              fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.legend(fontsize=14, loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    
    return plt

if __name__ == "__main__":
    # Create output directory
    output_folder = "plots/Au_200nm_vs_100nm_new"
    os.makedirs(output_folder, exist_ok=True)
    
    # Create the plot
    plt_obj = plot_all_4_spectra_no_max_norm()
    
    if plt_obj:
        # Save the plot
        plt_obj.savefig(os.path.join(output_folder, "all_4_spectra_no_max_norm_with_raw.png"), 
                       dpi=300, bbox_inches='tight')
        plt_obj.close()
        
        print(f"Plot saved to: {output_folder}/all_4_spectra_no_max_norm_with_raw.png")
    else:
        print("Failed to create plot - check data availability")
