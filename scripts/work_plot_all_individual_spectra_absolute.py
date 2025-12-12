from nanostructure_analysis import *
# ==============================================================================
# Plot All Individual Spectra: Absolute Values, No Averaging, No Max Normalization
# ==============================================================================
# This script plots every individual spectrum measurement with:
# - Basic normalization only (background subtraction + reference division)
# - No averaging - all individual traces shown
# - No peak/max normalization - absolute intensity values
# - Separate plots for 100nm and 200nm datasets

import matplotlib.pyplot as plt
import numpy as np
import os

# Import required functions from the processing modules
from Au_200nm_vs_100nm_new import load_spectra_cached

def basic_normalize_absolute_individual(data_dict, bkg_data, ref_data, ref_bkg_data, apply_smoothing=True):
    """
    Basic normalization for individual spectra: (signal - bkg) / (ref - ref_bkg)
    
    Args:
        data_dict: Dictionary of individual spectra data
        bkg_data: Background spectrum
        ref_data: Reference spectrum  
        ref_bkg_data: Reference background spectrum
        apply_smoothing: Whether to apply Savitzky-Golay smoothing
    
    Returns:
        Dictionary of normalized individual spectra
    """
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
        
        # Apply smoothing if requested
        if apply_smoothing:
            normalized_data_y = savgol_filter(normalized_data_y, 131, 1)
        
        # Apply cutoff wavelength filter
        cutoff_mask = normalized_data_x <= 950
        normalized_data_x = normalized_data_x[cutoff_mask]
        normalized_data_y = normalized_data_y[cutoff_mask]
        
        normalized_data[key_data] = np.column_stack((normalized_data_x, normalized_data_y))
    
    return normalized_data

def plot_all_individual_spectra_100nm():
    """
    Plot all individual 100nm spectra with absolute values
    """
    print("Creating 100nm individual spectra plot...")
    
    # Define data paths
    spectra_100nm_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250811 - sample6"
    
    # Load spectra data with caching
    print("Loading 100nm spectra...")
    spectra_100nm, params_100nm = load_spectra_cached(spectra_100nm_path)
    
    # Filter data (no averaging this time)
    print("Processing 100nm data...")
    box1_100nm_spec, _ = filter_spectra(
        spectra_100nm, params_100nm, "*box1*", 
        average=False, exclude=["*C5_2*", "*C4_2*", "*C3_1*"]  # No averaging
    )
    box4_100nm_spec, _ = filter_spectra(
        spectra_100nm, params_100nm, "*box4*", 
        average=False  # No averaging
    )
    
    # Get reference and background for 100nm
    ref_5um_100nm, _ = filter_spectra(spectra_100nm, params_100nm, "*5um*_100ms*")
    bkg_100nm, _ = filter_spectra(spectra_100nm, params_100nm, "*bkg*")
    
    # Normalize all individual spectra
    print("Normalizing 100nm individual spectra...")
    norm_100nm_box1 = basic_normalize_absolute_individual(
        box1_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], True
    )
    norm_100nm_box4 = basic_normalize_absolute_individual(
        box4_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], True
    )
    
    # Create the plot
    print("Creating 100nm plot...")
    plt.figure(figsize=(16, 10))
    
    # Plot all Box1 individual spectra
    for i, (key, spectrum) in enumerate(norm_100nm_box1.items()):
        plt.plot(spectrum[:, 0], spectrum[:, 1], 
                 color='dodgerblue', alpha=0.6, linewidth=1.5,
                 label='100nm Box1' if i == 0 else "")  # Only label first one
    
    # Plot all Box4 individual spectra  
    for i, (key, spectrum) in enumerate(norm_100nm_box4.items()):
        plt.plot(spectrum[:, 0], spectrum[:, 1], 
                 color='crimson', alpha=0.6, linewidth=1.5,
                 label='100nm Box4' if i == 0 else "")  # Only label first one
    
    # Styling - match Au_200nm_vs_100nm_new.py parameters
    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.ylabel('Normalized Intensity (Absolute)', fontsize=18, fontweight='bold')
    plt.title('100nm Gold: All Individual Spectra - No Max Norm', 
              fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.legend(fontsize=14, loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    # Set y-axis limits to 0-0.15 as requested
    plt.ylim(0, 0.15)
    
    # Add text showing number of spectra
    num_box1 = len(norm_100nm_box1)
    num_box4 = len(norm_100nm_box4)
    plt.text(0.02, 0.98, f'Box1: {num_box1} spectra\nBox4: {num_box4} spectra', 
             transform=plt.gca().transAxes, fontsize=14, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return plt

def plot_all_individual_spectra_200nm():
    """
    Plot all individual 200nm spectra with absolute values
    """
    print("Creating 200nm individual spectra plot...")
    
    # Define data paths
    spectra_200nm_before_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250812 - sample13"
    
    # Load spectra data with caching
    print("Loading 200nm spectra...")
    spectra_200nm_before, params_200nm_before = load_spectra_cached(spectra_200nm_before_path)
    
    # Filter data (no averaging this time)
    print("Processing 200nm data...")
    box1_200nm_spec, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*box1*", average=False)
    box4_200nm_spec, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*box4*", average=False)
    
    # Get reference and background for 200nm
    ref_5um_200nm_before, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*5um*")
    bkg_200nm_before, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*bkg*")
    
    # Normalize all individual spectra
    print("Normalizing 200nm individual spectra...")
    norm_200nm_box1 = basic_normalize_absolute_individual(
        box1_200nm_spec, bkg_200nm_before['bkg_single_track'], 
        ref_5um_200nm_before['5um_ref_single_track_50ms'], bkg_200nm_before['bkg_single_track_50ms'], True
    )
    norm_200nm_box4 = basic_normalize_absolute_individual(
        box4_200nm_spec, bkg_200nm_before['bkg_single_track'], 
        ref_5um_200nm_before['5um_ref_single_track_50ms'], bkg_200nm_before['bkg_single_track_50ms'], True
    )
    
    # Create the plot
    print("Creating 200nm plot...")
    plt.figure(figsize=(16, 10))
    
    # Plot all Box1 individual spectra
    for i, (key, spectrum) in enumerate(norm_200nm_box1.items()):
        plt.plot(spectrum[:, 0], spectrum[:, 1], 
                 color='dodgerblue', alpha=0.6, linewidth=1.5,
                 label='200nm Box1' if i == 0 else "")  # Only label first one
    
    # Plot all Box4 individual spectra  
    for i, (key, spectrum) in enumerate(norm_200nm_box4.items()):
        plt.plot(spectrum[:, 0], spectrum[:, 1], 
                 color='crimson', alpha=0.6, linewidth=1.5,
                 label='200nm Box4' if i == 0 else "")  # Only label first one
    
    # Styling - match Au_200nm_vs_100nm_new.py parameters
    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.ylabel('Normalized Intensity (Absolute)', fontsize=18, fontweight='bold')
    plt.title('200nm Gold: All Individual Spectra - No Max Norm', 
              fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.legend(fontsize=14, loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    # Set y-axis limits to 0-0.15 as requested
    plt.ylim(0, 0.15)
    
    # Add text showing number of spectra
    num_box1 = len(norm_200nm_box1)
    num_box4 = len(norm_200nm_box4)
    plt.text(0.02, 0.98, f'Box1: {num_box1} spectra\nBox4: {num_box4} spectra', 
             transform=plt.gca().transAxes, fontsize=14, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return plt

def plot_all_individual_spectra_combined():
    """
    Plot all individual spectra from both 100nm and 200nm datasets on one plot
    """
    print("Creating combined individual spectra plot...")
    
    # Define data paths
    spectra_100nm_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250811 - sample6"
    spectra_200nm_before_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250812 - sample13"
    
    # Load spectra data with caching
    print("Loading 100nm spectra...")
    spectra_100nm, params_100nm = load_spectra_cached(spectra_100nm_path)
    print("Loading 200nm spectra...")
    spectra_200nm_before, params_200nm_before = load_spectra_cached(spectra_200nm_before_path)
    
    # Filter 100nm data (no averaging)
    print("Processing 100nm data...")
    box1_100nm_spec, _ = filter_spectra(
        spectra_100nm, params_100nm, "*box1*", 
        average=False, exclude=["*C5_2*", "*C4_2*", "*C3_1*"]
    )
    box4_100nm_spec, _ = filter_spectra(
        spectra_100nm, params_100nm, "*box4*", 
        average=False
    )
    
    # Get reference and background for 100nm
    ref_5um_100nm, _ = filter_spectra(spectra_100nm, params_100nm, "*5um*_100ms*")
    bkg_100nm, _ = filter_spectra(spectra_100nm, params_100nm, "*bkg*")
    
    # Filter 200nm data (no averaging)
    print("Processing 200nm data...")
    box1_200nm_spec, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*box1*", average=False)
    box4_200nm_spec, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*box4*", average=False)
    
    # Get reference and background for 200nm
    ref_5um_200nm_before, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*5um*")
    bkg_200nm_before, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*bkg*")
    
    # Normalize all individual spectra
    print("Normalizing all individual spectra...")
    
    # 100nm datasets
    norm_100nm_box1 = basic_normalize_absolute_individual(
        box1_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], True
    )
    norm_100nm_box4 = basic_normalize_absolute_individual(
        box4_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], True
    )
    
    # 200nm datasets
    norm_200nm_box1 = basic_normalize_absolute_individual(
        box1_200nm_spec, bkg_200nm_before['bkg_single_track'], 
        ref_5um_200nm_before['5um_ref_single_track_50ms'], bkg_200nm_before['bkg_single_track_50ms'], True
    )
    norm_200nm_box4 = basic_normalize_absolute_individual(
        box4_200nm_spec, bkg_200nm_before['bkg_single_track'], 
        ref_5um_200nm_before['5um_ref_single_track_50ms'], bkg_200nm_before['bkg_single_track_50ms'], True
    )
    
    # Create the plot
    print("Creating combined plot...")
    plt.figure(figsize=(18, 12))
    
    # Plot all 100nm Box1 individual spectra
    for i, (key, spectrum) in enumerate(norm_100nm_box1.items()):
        plt.plot(spectrum[:, 0], spectrum[:, 1], 
                 color='lightblue', alpha=0.4, linewidth=1,
                 label='100nm Box1' if i == 0 else "")
    
    # Plot all 100nm Box4 individual spectra  
    for i, (key, spectrum) in enumerate(norm_100nm_box4.items()):
        plt.plot(spectrum[:, 0], spectrum[:, 1], 
                 color='lightblue', alpha=0.4, linewidth=1, linestyle='--',
                 label='100nm Box4' if i == 0 else "")
    
    # Plot all 200nm Box1 individual spectra
    for i, (key, spectrum) in enumerate(norm_200nm_box1.items()):
        plt.plot(spectrum[:, 0], spectrum[:, 1], 
                 color='lightcoral', alpha=0.4, linewidth=1,
                 label='200nm Box1' if i == 0 else "")
    
    # Plot all 200nm Box4 individual spectra  
    for i, (key, spectrum) in enumerate(norm_200nm_box4.items()):
        plt.plot(spectrum[:, 0], spectrum[:, 1], 
                 color='lightcoral', alpha=0.4, linewidth=1, linestyle='--',
                 label='200nm Box4' if i == 0 else "")
    
    # Styling - match Au_200nm_vs_100nm_new.py parameters
    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.ylabel('Normalized Intensity (Absolute)', fontsize=18, fontweight='bold')
    plt.title('All Individual Spectra: 100nm vs 200nm - No Max Norm', 
              fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.legend(fontsize=14, loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    # Set y-axis limits to 0-0.15 as requested
    plt.ylim(0, 0.15)
    
    # Add text showing number of spectra
    num_100nm_box1 = len(norm_100nm_box1)
    num_100nm_box4 = len(norm_100nm_box4)
    num_200nm_box1 = len(norm_200nm_box1)
    num_200nm_box4 = len(norm_200nm_box4)
    total_spectra = num_100nm_box1 + num_100nm_box4 + num_200nm_box1 + num_200nm_box4
    
    plt.text(0.02, 0.98, f'Total: {total_spectra} individual spectra\n' +
                         f'100nm Box1: {num_100nm_box1}\n' +
                         f'100nm Box4: {num_100nm_box4}\n' +
                         f'200nm Box1: {num_200nm_box1}\n' +
                         f'200nm Box4: {num_200nm_box4}', 
             transform=plt.gca().transAxes, fontsize=14, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return plt

if __name__ == "__main__":
    # Create output directory
    output_folder = "plots/Au_200nm_vs_100nm_new"
    os.makedirs(output_folder, exist_ok=True)
    
    print("Starting individual spectra plotting...")
    
    # Create individual plots
    
    # 100nm individual spectra
    plt_100nm = plot_all_individual_spectra_100nm()
    if plt_100nm:
        plt_100nm.savefig(os.path.join(output_folder, "all_individual_spectra_100nm_absolute.png"), 
                         dpi=300, bbox_inches='tight')
        plt_100nm.close()
        print(f"100nm individual spectra plot saved")
    
    # 200nm individual spectra
    plt_200nm = plot_all_individual_spectra_200nm()
    if plt_200nm:
        plt_200nm.savefig(os.path.join(output_folder, "all_individual_spectra_200nm_absolute.png"), 
                         dpi=300, bbox_inches='tight')
        plt_200nm.close()
        print(f"200nm individual spectra plot saved")
    
    # Combined individual spectra
    plt_combined = plot_all_individual_spectra_combined()
    if plt_combined:
        plt_combined.savefig(os.path.join(output_folder, "all_individual_spectra_combined_absolute.png"), 
                            dpi=300, bbox_inches='tight')
        plt_combined.close()
        print(f"Combined individual spectra plot saved")
    
    print("All individual spectra plots completed!")
