from nanostructure_analysis import *
# ==============================================================================
# Plot 200nm Gold: Box1 vs Box4 - Absolute Values (Before Irradiation Only)
# ==============================================================================
# This script plots the 200nm gold nanostructures comparing Box1 vs Box4
# using absolute intensity values without peak normalization

import matplotlib.pyplot as plt
import numpy as np
import os

# Import required functions from the processing modules
from Au_200nm_vs_100nm_new import load_spectra_cached

def plot_200nm_absolute_comparison():
    """
    Plot 200nm Box1 vs Box4 spectra using absolute intensity values
    Shows raw intensity differences without normalization
    """
    print("Creating 200nm absolute comparison plot...")
    
    # Define data path for 200nm before measurements
    spectra_200nm_before_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250812 - sample13"
    
    # Load 200nm spectra data 
    spectra_200nm_before, params_200nm_before = load_spectra_cached(spectra_200nm_before_path)
    
    # Filter Box1 and Box4 data
    box1_200nm_spec, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*box1*", average=True)
    box4_200nm_spec, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*box4*", average=True)
    
    # Get reference and background data
    ref_5um_200nm_before, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*5um*")
    bkg_200nm_before, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*bkg*")
    
    # Basic normalization function (background subtraction and reference division only)
    def basic_normalize_absolute(data_dict, bkg_data, ref_data, ref_bkg_data, apply_smoothing=True):
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
    
    # Get absolute normalized spectra - both smoothed and unsmoothed versions
    # Smoothed version (for main plot lines)
    abs_norm_200nm_box1_smooth = basic_normalize_absolute(
        box1_200nm_spec, 
        bkg_200nm_before['bkg_single_track'], 
        ref_5um_200nm_before['5um_ref_single_track_50ms'], 
        bkg_200nm_before['bkg_single_track_50ms'],
        apply_smoothing=True
    )
    
    abs_norm_200nm_box4_smooth = basic_normalize_absolute(
        box4_200nm_spec, 
        bkg_200nm_before['bkg_single_track'], 
        ref_5um_200nm_before['5um_ref_single_track_50ms'], 
        bkg_200nm_before['bkg_single_track_50ms'],
        apply_smoothing=True
    )
    
    # Unsmoothed version (for transparent background)
    abs_norm_200nm_box1_raw = basic_normalize_absolute(
        box1_200nm_spec, 
        bkg_200nm_before['bkg_single_track'], 
        ref_5um_200nm_before['5um_ref_single_track_50ms'], 
        bkg_200nm_before['bkg_single_track_50ms'],
        apply_smoothing=False
    )
    
    abs_norm_200nm_box4_raw = basic_normalize_absolute(
        box4_200nm_spec, 
        bkg_200nm_before['bkg_single_track'], 
        ref_5um_200nm_before['5um_ref_single_track_50ms'], 
        bkg_200nm_before['bkg_single_track_50ms'],
        apply_smoothing=False
    )
    
    # Average the spectra for each box - both smooth and raw versions
    # Smoothed averages
    if abs_norm_200nm_box1_smooth:
        wavelength_box1_smooth = list(abs_norm_200nm_box1_smooth.values())[0][:, 0]
        intensities_box1_smooth = [abs_norm_200nm_box1_smooth[key][:, 1] for key in abs_norm_200nm_box1_smooth.keys()]
        averaged_intensity_box1_smooth = np.mean(intensities_box1_smooth, axis=0)
        averaged_box1_smooth = np.column_stack([wavelength_box1_smooth, averaged_intensity_box1_smooth])
    else:
        print("Warning: No Box1 smooth data found")
        return None
    
    if abs_norm_200nm_box4_smooth:
        wavelength_box4_smooth = list(abs_norm_200nm_box4_smooth.values())[0][:, 0]
        intensities_box4_smooth = [abs_norm_200nm_box4_smooth[key][:, 1] for key in abs_norm_200nm_box4_smooth.keys()]
        averaged_intensity_box4_smooth = np.mean(intensities_box4_smooth, axis=0)
        averaged_box4_smooth = np.column_stack([wavelength_box4_smooth, averaged_intensity_box4_smooth])
    else:
        print("Warning: No Box4 smooth data found")
        return None
    
    # Raw (unsmoothed) averages
    if abs_norm_200nm_box1_raw:
        wavelength_box1_raw = list(abs_norm_200nm_box1_raw.values())[0][:, 0]
        intensities_box1_raw = [abs_norm_200nm_box1_raw[key][:, 1] for key in abs_norm_200nm_box1_raw.keys()]
        averaged_intensity_box1_raw = np.mean(intensities_box1_raw, axis=0)
        averaged_box1_raw = np.column_stack([wavelength_box1_raw, averaged_intensity_box1_raw])
    else:
        print("Warning: No Box1 raw data found")
        return None
    
    if abs_norm_200nm_box4_raw:
        wavelength_box4_raw = list(abs_norm_200nm_box4_raw.values())[0][:, 0]
        intensities_box4_raw = [abs_norm_200nm_box4_raw[key][:, 1] for key in abs_norm_200nm_box4_raw.keys()]
        averaged_intensity_box4_raw = np.mean(intensities_box4_raw, axis=0)
        averaged_box4_raw = np.column_stack([wavelength_box4_raw, averaged_intensity_box4_raw])
    else:
        print("Warning: No Box4 raw data found")
        return None
    
    # Create the plot
    plt.figure(figsize=(16, 8))
    
    # Plot transparent background traces (raw data)
    plt.plot(averaged_box1_raw[:, 0], averaged_box1_raw[:, 1], 
             color='dodgerblue', linestyle='-', linewidth=1, alpha=0.3,
             label='200nm Box1 (raw)')
    
    plt.plot(averaged_box4_raw[:, 0], averaged_box4_raw[:, 1], 
             color='crimson', linestyle='-', linewidth=1, alpha=0.3,
             label='200nm Box4 (raw)')
    
    # Plot main smoothed traces on top
    plt.plot(averaged_box1_smooth[:, 0], averaged_box1_smooth[:, 1], 
             color='dodgerblue', linestyle='-', linewidth=3, 
             label='200nm Box1 (d$_{ih}$ = 190 nm)')
    
    plt.plot(averaged_box4_smooth[:, 0], averaged_box4_smooth[:, 1], 
             color='crimson', linestyle='-', linewidth=3, 
             label='200nm Box4 (d$_{ih}$ = 220 nm)')
    
    # Styling
    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.ylabel('Normalized Intensity (Absolute)', fontsize=18, fontweight='bold')
    plt.title('200nm Gold: Box1 vs Box4 [No Max Norm]', 
              fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.legend(fontsize=12, loc='upper right')  # Smaller font for more legend entries
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    
    return plt

if __name__ == "__main__":
    # Create output directory
    output_folder = "plots/Au_200nm_vs_100nm_new"
    os.makedirs(output_folder, exist_ok=True)
    
    # Create the plot
    plt_obj = plot_200nm_absolute_comparison()
    
    if plt_obj:
        # Save the plot with new filename
        plt_obj.savefig(os.path.join(output_folder, "200nm_box1_vs_box4_absolute_values_with_raw.png"), 
                       dpi=300, bbox_inches='tight')
        plt_obj.close()
        
        print(f"Plot saved to: {output_folder}/200nm_box1_vs_box4_absolute_values_with_raw.png")
    else:
        print("Failed to create plot - check data availability")
