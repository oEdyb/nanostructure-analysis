from apd_functions import * 
from spectra_functions import *
from ALL_plotting_functions_OLD import *
import matplotlib.pyplot as plt
import os
import pickle
from confocal_functions import *
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

def get_sem_title_addition(gold_type, box_type):
    """
    Get SEM measurement string to add to plot titles
    
    Args:
        gold_type: "100nm" or "200nm"
        box_type: "box1" or "box4"
    
    Returns:
        String with SEM measurements or empty string if not available
    """
    # Define CSV paths for each sample
    csv_paths = {
        "100nm_box1": "Data/SEM/SEM_measurements_20250602_sample_6_box_1_and_4.csv",
        "100nm_box4": "Data/SEM/SEM_measurements_20250602_sample_6_box_1_and_4.csv", 
        "200nm_box1": "Data/SEM/SEM_measurements_20250304_DNHs_pristine_200nm_sample_13.csv",
        "200nm_box4": "Data/SEM/SEM_measurements_20250304_DNHs_pristine_200nm_sample_13.csv"
    }
    
    key = f"{gold_type}_{box_type}"
    if key not in csv_paths:
        return ""
    
    # Extract measurements
    measurements = extract_sem_measurements(csv_paths[key], box_type, gold_type)
    
    if measurements is None:
        return ""
    
    # Format the addition string
    addition = f"\nSEM: d_ih={measurements['interhole_distance']:.0f}nm, gap={measurements['gap_width']:.1f}nm, R̄={measurements['avg_radius']:.0f}nm"
    
    return addition

def extract_sem_value(sem_info_string, parameter):
    """
    Extract specific parameter value from SEM info string
    
    Args:
        sem_info_string: String containing SEM measurements
        parameter: Parameter to extract (e.g., 'd_ih', 'gap', 'R̄')
    
    Returns:
        Extracted value as string or "N/A" if not found
    """
    if not sem_info_string:
        return "N/A"
    
    # Use regex to extract the value after the parameter
    import re
    if parameter == 'd_ih':
        match = re.search(r'd_ih=(\d+)nm', sem_info_string)
    elif parameter == 'gap':
        match = re.search(r'gap=([\d.]+)nm', sem_info_string)
    elif parameter == 'R̄':
        match = re.search(r'R̄=(\d+)nm', sem_info_string)
    else:
        return "N/A"
    
    if match:
        return match.group(1)
    else:
        return "N/A"

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

def plot_snr_violin_comparison():
    """Create violin plot comparing SNR distributions across different sample/box combinations"""
    # Collect SNR data
    sample6_snr_before = extract_snr_values(sample6_results_before)
    sample7_snr_before = extract_snr_values(sample7_results_before)
    sample13_box1_snr_before = extract_snr_values(sample13_box1_results_before)
    sample13_box4_snr_before = extract_snr_values(sample13_box4_results_before)
    
    # Prepare data for violin plot
    data = [sample6_snr_before, sample7_snr_before, sample13_box1_snr_before, sample13_box4_snr_before]
    
    # Multi-line labels with interhole distances
    labels = ['100nm\nBox1\nd$_{ih}$ = 190 nm', 
              '100nm\nBox4\nd$_{ih}$ = 220 nm', 
              '200nm\nBox1\nd$_{ih}$ = 190 nm', 
              '200nm\nBox4\nd$_{ih}$ = 220 nm']
    
    # Create violin plot with styling matching plotting_functions.py
    plt.figure(figsize=(16, 8))  # Match the figsize from plotting_functions
    violin_parts = plt.violinplot(data, positions=range(len(labels)), showmeans=True, showmedians=True)
    
    # More visible colors - using distinct, saturated colors
    colors = ['dodgerblue', 'crimson', 'forestgreen', 'orange']
    for i, (pc, color) in enumerate(zip(violin_parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.8)  # Higher alpha for better visibility
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Style the violin plot elements to match plotting_functions.py
    for element in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
        if element in violin_parts:
            violin_parts[element].set_color('black')
            violin_parts[element].set_linewidth(2)
    
    # Add scatter points for individual measurements with better visibility
    for i, (snr_vals, color) in enumerate(zip(data, colors)):
        if len(snr_vals) > 0:
            # Add some jitter for visibility
            x_jitter = np.random.normal(i, 0.05, len(snr_vals))
            plt.scatter(x_jitter, snr_vals, alpha=0.8, s=80, color=color, 
                       edgecolors='black', linewidths=1.5)
    
    # Customize plot with styling matching plotting_functions.py
    plt.xticks(range(len(labels)), labels, fontsize=18, fontweight='bold')  # Match font sizes
    plt.ylabel('SNR (Before Irradiation)', fontsize=18, fontweight='bold')
    plt.title('SNR Comparison', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')  # Match grid style
    plt.tick_params(axis='both', which='major', labelsize=16)  # Match tick size
    plt.ylim(2, 16)  # Set y-axis limit to 16
    
    # Add statistics text with better formatting
    for i, (snr_vals, label) in enumerate(zip(data, labels)):
        if len(snr_vals) > 0:
            mean_val = np.mean(snr_vals)
            std_val = np.std(snr_vals)
            plt.text(i, max(snr_vals) + 0.5, f'μ={mean_val:.1f}\nσ={std_val:.1f}', 
                    ha='center', va='bottom', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return plt

def plot_noise_violin_comparison():
    """Create violin plot comparing noise distributions across different sample/box combinations"""
    # Collect noise data
    sample6_noise_before = extract_noise_values(sample6_results_before)
    sample7_noise_before = extract_noise_values(sample7_results_before)
    sample13_box1_noise_before = extract_noise_values(sample13_box1_results_before)
    sample13_box4_noise_before = extract_noise_values(sample13_box4_results_before)
    
    # Prepare data for violin plot
    data = [sample6_noise_before, sample7_noise_before, sample13_box1_noise_before, sample13_box4_noise_before]
    
    # Multi-line labels with interhole distances
    labels = ['100nm\nBox1\nd$_{ih}$ = 190 nm', 
              '100nm\nBox4\nd$_{ih}$ = 220 nm', 
              '200nm\nBox1\nd$_{ih}$ = 190 nm', 
              '200nm\nBox4\nd$_{ih}$ = 220 nm']
    
    # Create violin plot with styling matching plotting_functions.py
    plt.figure(figsize=(16, 8))  # Match the figsize from plotting_functions
    violin_parts = plt.violinplot(data, positions=range(len(labels)), showmeans=True, showmedians=True)
    
    # More visible colors - using distinct, saturated colors
    colors = ['dodgerblue', 'crimson', 'forestgreen', 'orange']
    for i, (pc, color) in enumerate(zip(violin_parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.8)  # Higher alpha for better visibility
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Style the violin plot elements to match plotting_functions.py
    for element in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
        if element in violin_parts:
            violin_parts[element].set_color('black')
            violin_parts[element].set_linewidth(2)
    
    # Add scatter points for individual measurements with better visibility
    for i, (noise_vals, color) in enumerate(zip(data, colors)):
        if len(noise_vals) > 0:
            # Add some jitter for visibility
            x_jitter = np.random.normal(i, 0.05, len(noise_vals))
            plt.scatter(x_jitter, noise_vals, alpha=0.8, s=80, color=color, 
                       edgecolors='black', linewidths=1.5)
    
    # Customize plot with styling matching plotting_functions.py
    plt.xticks(range(len(labels)), labels, fontsize=18, fontweight='bold')  # Match font sizes
    plt.ylabel('Noise (mV, Before Irradiation)', fontsize=18, fontweight='bold')
    plt.title('Noise Comparison', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')  # Match grid style
    plt.tick_params(axis='both', which='major', labelsize=16)  # Match tick size
    
    # Add statistics text with better formatting for mV values
    for i, (noise_vals, label) in enumerate(zip(data, labels)):
        if len(noise_vals) > 0:
            mean_val = np.mean(noise_vals)
            std_val = np.std(noise_vals)
            plt.text(i, max(noise_vals) + max(noise_vals)*0.05, f'μ={mean_val:.2f}\nσ={std_val:.2f}', 
                    ha='center', va='bottom', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return plt

def plot_snr_violin_comparison_after():
    """Create violin plot comparing SNR distributions after irradiation across different sample/box combinations"""
    # Collect SNR data after irradiation
    sample6_snr_after = extract_snr_values(sample6_results_after)
    sample7_snr_after = extract_snr_values(sample7_results_after)
    sample13_box1_snr_after = extract_snr_values(sample13_box1_results_after)
    sample13_box4_snr_after = extract_snr_values(sample13_box4_results_after)
    
    # Prepare data for violin plot
    data = [sample6_snr_after, sample7_snr_after, sample13_box1_snr_after, sample13_box4_snr_after]
    
    # Multi-line labels with interhole distances
    labels = ['100nm\nBox1\nd$_{ih}$ = 190 nm', 
              '100nm\nBox4\nd$_{ih}$ = 220 nm', 
              '200nm\nBox1\nd$_{ih}$ = 190 nm', 
              '200nm\nBox4\nd$_{ih}$ = 220 nm']
    
    # Create violin plot with styling matching plotting_functions.py
    plt.figure(figsize=(16, 8))  # Match the figsize from plotting_functions
    violin_parts = plt.violinplot(data, positions=range(len(labels)), showmeans=True, showmedians=True)
    
    # More visible colors - using distinct, saturated colors
    colors = ['dodgerblue', 'crimson', 'forestgreen', 'orange']
    for i, (pc, color) in enumerate(zip(violin_parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.8)  # Higher alpha for better visibility
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Style the violin plot elements to match plotting_functions.py
    for element in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
        if element in violin_parts:
            violin_parts[element].set_color('black')
            violin_parts[element].set_linewidth(2)
    
    # Add scatter points for individual measurements with better visibility
    for i, (snr_vals, color) in enumerate(zip(data, colors)):
        if len(snr_vals) > 0:
            # Add some jitter for visibility
            x_jitter = np.random.normal(i, 0.05, len(snr_vals))
            plt.scatter(x_jitter, snr_vals, alpha=0.8, s=80, color=color, 
                       edgecolors='black', linewidths=1.5)
    
    # Customize plot with styling matching plotting_functions.py
    plt.xticks(range(len(labels)), labels, fontsize=18, fontweight='bold')  # Match font sizes
    plt.ylabel('SNR (After Irradiation)', fontsize=18, fontweight='bold')
    plt.title('SNR Comparison After Irradiation', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')  # Match grid style
    plt.tick_params(axis='both', which='major', labelsize=16)  # Match tick size
    plt.ylim(2, 22)  # Set y-axis limit to 16 to match before plot
    
    # Add statistics text with better formatting
    for i, (snr_vals, label) in enumerate(zip(data, labels)):
        if len(snr_vals) > 0:
            mean_val = np.mean(snr_vals)
            std_val = np.std(snr_vals)
            plt.text(i, max(snr_vals) + 0.5, f'μ={mean_val:.1f}\nσ={std_val:.1f}', 
                    ha='center', va='bottom', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return plt

def plot_snr_before_after_comparison():
    """Create scatter plot with SNR after values vs power, with median SNR before as dashed lines"""
    plt.figure(figsize=(16, 8))
    
    # Colors: blue family for 100nm, red family for 200nm, consistent with spectra plot
    colors = ['dodgerblue', 'deepskyblue', 'crimson', 'indianred']
    
    # Dataset information
    datasets = [
        (sample6_before, sample6_after, sample6_results_before, sample6_results_after, sample6_params, "100nm Box1 (d$_{ih}$ = 190 nm)"),
        (sample7_before, sample7_after, sample7_results_before, sample7_results_after, sample7_params, "100nm Box4 (d$_{ih}$ = 220 nm)"),
        (sample13_box1_before, sample13_box1_after, sample13_box1_results_before, sample13_box1_results_after, sample13_box1_params, "200nm Box1 (d$_{ih}$ = 190 nm)"),
        (sample13_box4_before, sample13_box4_after, sample13_box4_results_before, sample13_box4_results_after, sample13_box4_params, "200nm Box4 (d$_{ih}$ = 220 nm)")
    ]
    
    # Get global power range for dashed lines
    all_powers = []
    for confocal_before, confocal_after, results_before, results_after, apd_params, label in datasets:
        # Extract powers from APD params
        pattern = r'[A-D][1-6]'
        for k in apd_params.keys():
            match = re.search(pattern, k)
            if match:
                all_powers.append(apd_params[k]['power'])
    
    power_min = min(all_powers) if all_powers else 0
    power_max = max(all_powers) if all_powers else 100
    
    for i, (confocal_before, confocal_after, results_before, results_after, apd_params, label) in enumerate(datasets):
        color = colors[i]
        
        # Extract image_dicts from tuples
        confocal_dict_before = confocal_before[0]
        confocal_dict_after = confocal_after[0]
        
        # Find common [A-D][1-6] patterns between datasets
        pattern = r'[A-D][1-6]'
        before_locs = {re.search(pattern, k).group(): k for k in confocal_dict_before.keys() if re.search(pattern, k)}
        after_locs = {re.search(pattern, k).group(): k for k in confocal_dict_after.keys() if re.search(pattern, k)}
        
        # Find matching patterns in APD params and extract power values
        apd_locs = {}
        for k in apd_params.keys():
            match = re.search(pattern, k)
            if match:
                apd_locs[match.group()] = apd_params[k]['power']
        
        # Get locations that exist in all datasets
        locations = sorted(set(before_locs.keys()) & set(after_locs.keys()) & set(apd_locs.keys()))
        
        # Extract SNR values for before (for median line)
        snr_before_values = []
        for loc in locations:
            before_key = before_locs[loc]
            if before_key in results_before and 'snr_3x3' in results_before[before_key]:
                snr_before_values.append(results_before[before_key]['snr_3x3'])
        
        # Calculate median SNR before
        if snr_before_values:
            median_snr_before = np.median(snr_before_values)
            
            # Plot median line across power range
            plt.axhline(y=median_snr_before, color=color, linestyle='--', linewidth=3, 
                       alpha=0.8, label=f'{label} Before (median)')
        
        # Extract SNR after values and powers for scatter plot
        snr_after_values = []
        powers_after = []
        for loc in locations:
            after_key = after_locs[loc]
            if after_key in results_after and 'snr_3x3' in results_after[after_key]:
                snr_after_values.append(results_after[after_key]['snr_3x3'])
                powers_after.append(apd_locs[loc])
        
        # Plot scatter points for after values
        if snr_after_values and powers_after:
            plt.scatter(powers_after, snr_after_values, s=100, alpha=0.8, color=color, 
                       edgecolors='black', linewidths=1.5, label=f'{label} After')
    
    # Customize plot with styling matching plotting_functions.py
    plt.xlabel('Power (mW)', fontsize=18, fontweight='bold')
    plt.ylabel('SNR', fontsize=18, fontweight='bold')
    plt.title('SNR Before vs After Comparison', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.tight_layout()
    return plt

def plot_100nm_box1_vs_box4_before():
    """Create comparison plot for 100nm gold: box1 vs box4 (before data only) with proper peak normalization"""
    # First, do basic normalization (background subtraction and reference division)
    box1_100nm_spec, _ = filter_spectra(spectra_100nm, params_100nm, "*box1*", average=True, exclude=["*C5_2*", "*C4_2*", "*C3_1*"])
    box4_100nm_spec, _ = filter_spectra(spectra_100nm, params_100nm, "*box4*", average=True)
    
    # Basic normalization without peak normalization
    from spectra_functions import normalize_spectra
    
    # Temporarily disable peak normalization in normalize_spectra by using a custom version
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
    
    # Get basic normalized spectra
    basic_norm_100nm_box1 = normalize_spectra_no_peak(box1_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], 
                                                      savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    basic_norm_100nm_box4 = normalize_spectra_no_peak(box4_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], 
                                                      savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    
    # Average the box1 data and find the peak wavelength in 700-800 nm range
    if basic_norm_100nm_box1:
        wavelength_100nm_box1 = list(basic_norm_100nm_box1.values())[0][:, 0]
        intensities_100nm_box1 = [basic_norm_100nm_box1[key][:, 1] for key in basic_norm_100nm_box1.keys()]
        averaged_intensity_100nm_box1 = np.mean(intensities_100nm_box1, axis=0)
        
        # Find peak wavelength in 700-800 nm range
        peak_wavelength = find_peak_wavelength(wavelength_100nm_box1, averaged_intensity_100nm_box1, 700, 800)
        print(f"100nm Box1 peak wavelength found at: {peak_wavelength:.1f} nm")
        
        # Now normalize both box1 and box4 at this specific wavelength
        norm_100nm_box1 = normalize_at_peak_wavelength(basic_norm_100nm_box1, peak_wavelength)
        norm_100nm_box4 = normalize_at_peak_wavelength(basic_norm_100nm_box4, peak_wavelength)
        
        # Re-average after peak normalization
        intensities_100nm_box1_norm = [norm_100nm_box1[key][:, 1] for key in norm_100nm_box1.keys()]
        averaged_intensity_100nm_box1_norm = np.mean(intensities_100nm_box1_norm, axis=0)
        averaged_100nm_box1 = np.column_stack([wavelength_100nm_box1, averaged_intensity_100nm_box1_norm])
        
        wavelength_100nm_box4 = list(norm_100nm_box4.values())[0][:, 0]
        intensities_100nm_box4_norm = [norm_100nm_box4[key][:, 1] for key in norm_100nm_box4.keys()]
        averaged_intensity_100nm_box4_norm = np.mean(intensities_100nm_box4_norm, axis=0)
        averaged_100nm_box4 = np.column_stack([wavelength_100nm_box4, averaged_intensity_100nm_box4_norm])
        
        # Store peak wavelength for use in 200nm plot
        global peak_wavelength_100nm_global
        peak_wavelength_100nm_global = peak_wavelength
    else:
        print("Warning: No box1 data found for 100nm sample")
        return None
    
    # Create the comparison plot
    plt.figure(figsize=(16, 8))
    
    # Plot both datasets with distinct colors
    plt.plot(averaged_100nm_box1[:, 0], averaged_100nm_box1[:, 1], 
             color='dodgerblue', linestyle='-', linewidth=3, label='Box1')
    
    plt.plot(averaged_100nm_box4[:, 0], averaged_100nm_box4[:, 1], 
             color='crimson', linestyle='-', linewidth=3, label='Box4')
    
    # Add vertical line at normalization peak
    plt.axvline(x=peak_wavelength, color='gray', linestyle=':', linewidth=2, alpha=0.8, label=f'Peak: {peak_wavelength:.1f} nm')
    
    # Clean, concise styling
    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.ylabel('Normalized Intensity', fontsize=18, fontweight='bold')
    plt.title('100nm Gold: Box1 vs Box4', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    return plt

def plot_200nm_box1_vs_box4_before(peak_wavelength_100nm):
    """Create comparison plot for 200nm gold: box1 vs box4 (before data only) using 100nm peak wavelength"""
    # Filter and normalize box1 and box4 data for 200nm sample (before irradiation)
    box1_200nm_before_spec, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*box1*", average=True)
    box4_200nm_before_spec, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*box4*", average=True)
    
    if not box1_200nm_before_spec:
        print("Warning: No box1 data found for 200nm sample before irradiation")
        return None
    
    if not box4_200nm_before_spec:
        print("Warning: No box4 data found for 200nm sample before irradiation")
        return None
    
    # Basic normalization without peak normalization
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
    
    # Get basic normalized spectra
    basic_norm_200nm_box1 = normalize_spectra_no_peak(box1_200nm_before_spec, bkg_200nm_before['bkg_single_track'], ref_5um_200nm_before['5um_ref_single_track_50ms'], bkg_200nm_before['bkg_single_track_50ms'], 
                                                      savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    basic_norm_200nm_box4 = normalize_spectra_no_peak(box4_200nm_before_spec, bkg_200nm_before['bkg_single_track'], ref_5um_200nm_before['5um_ref_single_track_50ms'], bkg_200nm_before['bkg_single_track_50ms'], 
                                                      savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    
    print(f"Using 100nm Box1 peak wavelength for 200nm normalization: {peak_wavelength_100nm:.1f} nm")
    
    # Normalize both datasets at the same wavelength as 100nm box1
    norm_200nm_box1 = normalize_at_peak_wavelength(basic_norm_200nm_box1, peak_wavelength_100nm)
    norm_200nm_box4 = normalize_at_peak_wavelength(basic_norm_200nm_box4, peak_wavelength_100nm)
    
    # Average the normalized spectra
    wavelength_200nm_box1 = list(norm_200nm_box1.values())[0][:, 0]
    intensities_200nm_box1 = [norm_200nm_box1[key][:, 1] for key in norm_200nm_box1.keys()]
    averaged_intensity_200nm_box1 = np.mean(intensities_200nm_box1, axis=0)
    averaged_200nm_box1 = np.column_stack([wavelength_200nm_box1, averaged_intensity_200nm_box1])
    
    wavelength_200nm_box4 = list(norm_200nm_box4.values())[0][:, 0]
    intensities_200nm_box4 = [norm_200nm_box4[key][:, 1] for key in norm_200nm_box4.keys()]
    averaged_intensity_200nm_box4 = np.mean(intensities_200nm_box4, axis=0)
    averaged_200nm_box4 = np.column_stack([wavelength_200nm_box4, averaged_intensity_200nm_box4])

    # Create the comparison plot
    plt.figure(figsize=(16, 8))
    
    # Plot both datasets with distinct colors
    plt.plot(averaged_200nm_box1[:, 0], averaged_200nm_box1[:, 1], 
             color='dodgerblue', linestyle='-', linewidth=3, label='Box1')
    
    plt.plot(averaged_200nm_box4[:, 0], averaged_200nm_box4[:, 1], 
             color='crimson', linestyle='-', linewidth=3, label='Box4')
    
    # Add vertical line at normalization peak
    plt.axvline(x=peak_wavelength_100nm, color='gray', linestyle=':', linewidth=2, alpha=0.8, label=f'Peak: {peak_wavelength_100nm:.1f} nm')
    
    # Clean, concise styling
    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.ylabel('Normalized Intensity', fontsize=18, fontweight='bold')
    plt.title('200nm Gold: Box1 vs Box4', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    return plt

def plot_all_spectra_comparison_at_peak():
    """Create comparison plot with all 4 averaged normalized spectra datasets using proper peak normalization"""
    # Use the peak wavelength determined from 100nm box1 data
    if peak_wavelength_100nm_global is None:
        print("Error: Peak wavelength not determined yet. Run 100nm analysis first.")
        return None
        
    # Filter and process all datasets with consistent normalization
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
    
    # Process 100nm data
    box1_100nm_spec, _ = filter_spectra(spectra_100nm, params_100nm, "*box1*", average=True, exclude=["*C5_2*", "*C4_2*", "*C3_1*"])
    box4_100nm_spec, _ = filter_spectra(spectra_100nm, params_100nm, "*box4*", average=True)
    
    basic_norm_100nm_box1 = normalize_spectra_no_peak(box1_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], 
                                                      savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    basic_norm_100nm_box4 = normalize_spectra_no_peak(box4_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], 
                                                      savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    
    # Process 200nm data (using BEFORE data for consistency)
    box1_200nm_before_spec, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*box1*", average=True)
    box4_200nm_before_spec, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*box4*", average=True)
    
    basic_norm_200nm_box1 = normalize_spectra_no_peak(box1_200nm_before_spec, bkg_200nm_before['bkg_single_track'], ref_5um_200nm_before['5um_ref_single_track_50ms'], bkg_200nm_before['bkg_single_track_50ms'], 
                                                      savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    basic_norm_200nm_box4 = normalize_spectra_no_peak(box4_200nm_before_spec, bkg_200nm_before['bkg_single_track'], ref_5um_200nm_before['5um_ref_single_track_50ms'], bkg_200nm_before['bkg_single_track_50ms'], 
                                                      savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    
    # Normalize all at the same peak wavelength from 100nm box1
    norm_100nm_box1_peak = normalize_at_peak_wavelength(basic_norm_100nm_box1, peak_wavelength_100nm_global)
    norm_100nm_box4_peak = normalize_at_peak_wavelength(basic_norm_100nm_box4, peak_wavelength_100nm_global)
    norm_200nm_box1_peak = normalize_at_peak_wavelength(basic_norm_200nm_box1, peak_wavelength_100nm_global)
    norm_200nm_box4_peak = normalize_at_peak_wavelength(basic_norm_200nm_box4, peak_wavelength_100nm_global)
    
    # Average the normalized spectra
    wavelength_100nm_box1 = list(norm_100nm_box1_peak.values())[0][:, 0]
    intensities_100nm_box1 = [norm_100nm_box1_peak[key][:, 1] for key in norm_100nm_box1_peak.keys()]
    averaged_intensity_100nm_box1 = np.mean(intensities_100nm_box1, axis=0)
    averaged_100nm_box1 = np.column_stack([wavelength_100nm_box1, averaged_intensity_100nm_box1])
    
    wavelength_100nm_box4 = list(norm_100nm_box4_peak.values())[0][:, 0]
    intensities_100nm_box4 = [norm_100nm_box4_peak[key][:, 1] for key in norm_100nm_box4_peak.keys()]
    averaged_intensity_100nm_box4 = np.mean(intensities_100nm_box4, axis=0)
    averaged_100nm_box4 = np.column_stack([wavelength_100nm_box4, averaged_intensity_100nm_box4])
    
    wavelength_200nm_box1 = list(norm_200nm_box1_peak.values())[0][:, 0]
    intensities_200nm_box1 = [norm_200nm_box1_peak[key][:, 1] for key in norm_200nm_box1_peak.keys()]
    averaged_intensity_200nm_box1 = np.mean(intensities_200nm_box1, axis=0)
    averaged_200nm_box1 = np.column_stack([wavelength_200nm_box1, averaged_intensity_200nm_box1])
    
    wavelength_200nm_box4 = list(norm_200nm_box4_peak.values())[0][:, 0]
    intensities_200nm_box4 = [norm_200nm_box4_peak[key][:, 1] for key in norm_200nm_box4_peak.keys()]
    averaged_intensity_200nm_box4 = np.mean(intensities_200nm_box4, axis=0)
    averaged_200nm_box4 = np.column_stack([wavelength_200nm_box4, averaged_intensity_200nm_box4])
    
    # Create the comparison plot
    plt.figure(figsize=(16, 8))
    
    # Plot all 4 datasets with consistent styling
    # 100nm Box1 - solid blue
    plt.plot(averaged_100nm_box1[:, 0], averaged_100nm_box1[:, 1], 
             color='dodgerblue', linestyle='-', linewidth=3, label='100nm Box1')
    
    # 100nm Box4 - dashed blue
    plt.plot(averaged_100nm_box4[:, 0], averaged_100nm_box4[:, 1], 
             color='dodgerblue', linestyle='--', linewidth=3, label='100nm Box4')
    
    # 200nm Box1 - solid red
    plt.plot(averaged_200nm_box1[:, 0], averaged_200nm_box1[:, 1], 
             color='crimson', linestyle='-', linewidth=3, label='200nm Box1')
    
    # 200nm Box4 - dashed red
    plt.plot(averaged_200nm_box4[:, 0], averaged_200nm_box4[:, 1], 
             color='crimson', linestyle='--', linewidth=3, label='200nm Box4')
    
    # Add vertical line at normalization peak
    plt.axvline(x=peak_wavelength_100nm_global, color='gray', linestyle=':', linewidth=2, alpha=0.8, label=f'Peak: {peak_wavelength_100nm_global:.1f} nm')
    
    # Clean, concise styling
    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.ylabel('Normalized Intensity', fontsize=18, fontweight='bold')
    plt.title('100 vs 200nm gold: Box1 vs Box4', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    return plt


if __name__ == "__main__":
    # Output directory setup
    output_folder = "plots/Au_200nm_vs_100nm"
    os.makedirs(output_folder, exist_ok=True)

    # ========== Data paths ==========
    # 100nm sample data paths
    sample6_apd_path = r"Data\APD\2025.06.11 - Sample 6 Power Threshold"
    sample7_apd_path = r"Data\APD\2025.06.03 - Power Threshold Sample 7 box 4"
    sample6_confocal_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.06.11 - Sample 6 Power Threshold"
    sample7_confocal_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.06.03 - Power Threshold Sample 7 box 4"
    spectra_100nm_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250811 - sample6"

    # 200nm sample data paths
    sample13_apd_path = r"Data\APD\2025.08.21 - Sample 13 Power Threshold"
    sample13_pt_apd_path = r"Data\APD\2025.09.02 - Sample 13 PT high power"
    sample13_confocal_path = r"Data\Confocal\2025.08.21 - Sample 13 Power Threshold box1"
    sample13_pt_confocal_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.09.02 - Sample 13 before after break"
    spectra_200nm_after_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250821 - sample13 - after"
    spectra_200nm_before_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250812 - sample13"

    # ========== Load 100nm APD data ==========
    print("Loading 100nm APD data...")
    sample6_apd, sample6_monitor, sample6_params = apd_load_main(sample6_apd_path)
    sample6_apd, sample6_monitor, sample6_params = filter_apd(sample6_apd, sample6_monitor, sample6_params, "*[A-D][1-6]*")
    
    sample7_apd, sample7_monitor, sample7_params = apd_load_main(sample7_apd_path)
    sample7_apd, sample7_monitor, sample7_params = filter_apd(sample7_apd, sample7_monitor, sample7_params, "*[A-D][1-6]*", exclude=["C4"])

    # Fix power parameters for missing measurements
    for key in sample6_params.keys():
        if 'D6' in key:
            sample6_params[key]['power'] = 100.0
    
    for key in sample7_params.keys():
        if 'C6' in key:
            sample7_params[key]['power'] = 100.0

    # ========== Load 200nm APD data ==========
    print("Loading 200nm APD data...")
    sample13_apd, sample13_monitor, sample13_params = apd_load_main(sample13_apd_path)
    sample13_pt_apd, sample13_pt_monitor, sample13_pt_params = apd_load_main(sample13_pt_apd_path)
    
    # Combine normal and PT APD data for 200nm
    combined_13_apd = {**sample13_apd, **sample13_pt_apd}
    combined_13_monitor = {**sample13_monitor, **sample13_pt_monitor}
    combined_13_params = {**sample13_params, **sample13_pt_params}
    
    # Filter combined datasets for Box1 and Box4
    sample13_box1_apd, sample13_box1_monitor, sample13_box1_params = filter_apd(combined_13_apd, combined_13_monitor, combined_13_params, "*box1*")
    sample13_box4_apd, sample13_box4_monitor, sample13_box4_params = filter_apd(combined_13_apd, combined_13_monitor, combined_13_params, "*box4*[!_D4_*]*")

    # ========== Load confocal data ==========
    print("Loading confocal data...")
    # 100nm confocal
    sample6_confocal_data = load_with_cache(sample6_confocal_path, confocal_main)
    sample7_confocal_data = load_with_cache(sample7_confocal_path, confocal_main)
    
    # Filter 100nm confocal images
    sample6_before = filter_confocal(sample6_confocal_data, "*", exclude=["after"])
    sample6_after = filter_confocal(sample6_confocal_data, "*after*")
    sample7_before = filter_confocal(sample7_confocal_data, "*", exclude=["after"])
    sample7_after = filter_confocal(sample7_confocal_data, "*after*")
    
    # 200nm confocal
    sample13_confocal_data = load_with_cache(sample13_confocal_path, confocal_main)
    sample13_pt_confocal_data = load_with_cache(sample13_pt_confocal_path, confocal_main)
    
    # Combine 200nm confocal datasets
    combined_13_confocal = (
        {**sample13_confocal_data[0], **sample13_pt_confocal_data[0]},  # image_dict
        {**sample13_confocal_data[1], **sample13_pt_confocal_data[1]},  # apd_dict  
        {**sample13_confocal_data[2], **sample13_pt_confocal_data[2]},  # monitor_dict
        {**sample13_confocal_data[3], **sample13_pt_confocal_data[3]},  # xy_dict
        {**sample13_confocal_data[4], **sample13_pt_confocal_data[4]}   # z_dict
    )
    
    # Filter 200nm confocal images
    sample13_box1_before = filter_confocal(combined_13_confocal, "*box1*", exclude=["after", "C2"])
    sample13_box1_after = filter_confocal(combined_13_confocal, "*box1*after*", exclude=["C2"])
    sample13_box4_before = filter_confocal(combined_13_confocal, "*box4*", exclude=["after", "C2"])
    sample13_box4_after = filter_confocal(combined_13_confocal, "*box4*after*", exclude=["C2"])

    # ========== Analyze confocal data ==========
    print("Analyzing confocal data...")
    sample6_results_before = analyze_confocal(sample6_before)
    sample6_results_after = analyze_confocal(sample6_after)
    sample7_results_before = analyze_confocal(sample7_before)
    sample7_results_after = analyze_confocal(sample7_after)
    
    sample13_box1_results_before = analyze_confocal(sample13_box1_before)
    sample13_box1_results_after = analyze_confocal(sample13_box1_after)
    sample13_box4_results_before = analyze_confocal(sample13_box4_before)
    sample13_box4_results_after = analyze_confocal(sample13_box4_after)

    # ========== Load and normalize spectra data ==========
    print("Loading and processing spectra data...")
    # 100nm spectra
    spectra_100nm, params_100nm = load_spectra_cached(spectra_100nm_path)
    box1_100nm_spec, _ = filter_spectra(spectra_100nm, params_100nm, "*box1*", average=True, exclude=["*C5_2*", "*C4_2*", "*C3_1*"])
    ref_5um_100nm, _ = filter_spectra(spectra_100nm, params_100nm, "*5um*_100ms*")
    bkg_100nm, _ = filter_spectra(spectra_100nm, params_100nm, "*bkg*")
    
    # First normalize 100nm spectra to get reference for 700-800 nm max
    norm_100nm_temp = normalize_spectra_bkg(box1_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], 
                                       savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    
    # Now normalize 100nm spectra using the new method (self-referencing)
    norm_100nm = normalize_spectra_with_100nm_max(box1_100nm_spec, bkg_100nm['bkg_5000ms'], ref_5um_100nm['5um_100ms'], bkg_100nm['bkg_100ms'], 
                                                 norm_100nm_temp, savgol_before_bkg=False, savgol_after_div=True, 
                                                 savgol_after_div_window=131, savgol_after_div_order=1)
    
    # 200nm spectra
    spectra_200nm_after, params_200nm_after = load_spectra_cached(spectra_200nm_after_path)
    spectra_200nm_before, params_200nm_before = load_spectra_cached(spectra_200nm_before_path)
    
    # Filter 200nm spectra
    box1_200nm_after_spec, _ = filter_spectra(spectra_200nm_after, params_200nm_after, "*box1*", average=True)
    box1_200nm_before_spec, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*box1*", average=True)
    ref_5um_200nm_after, _ = filter_spectra(spectra_200nm_after, params_200nm_after, "*5um*_z_locked*")
    bkg_200nm_after, _ = filter_spectra(spectra_200nm_after, params_200nm_after, "*bkg*")
    ref_5um_200nm_before, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*5um*")
    bkg_200nm_before, _ = filter_spectra(spectra_200nm_before, params_200nm_before, "*bkg*")
    
    # Normalize 200nm spectra using 100nm max as reference
    norm_200nm_after = normalize_spectra_with_100nm_max(box1_200nm_after_spec, bkg_200nm_after['bkg_10000ms_1'], ref_5um_200nm_after['5um_100ms_z_locked_1'], bkg_200nm_after['bkg_100ms_3'], 
                                                       norm_100nm_temp, savgol_before_bkg=False, savgol_after_div=True, 
                                                       savgol_after_div_window=131, savgol_after_div_order=1)
    norm_200nm_before = normalize_spectra_with_100nm_max(box1_200nm_before_spec, bkg_200nm_before['bkg_single_track'], ref_5um_200nm_before['5um_ref_single_track_50ms'], bkg_200nm_before['bkg_single_track_50ms'], 
                                                        norm_100nm_temp, savgol_before_bkg=False, savgol_after_div=True, 
                                                        savgol_after_div_window=131, savgol_after_div_order=1)

    # ========== Calculate PSF total size ==========
    print("Calculating PSF parameters...")
    for results in [sample6_results_before, sample6_results_after, sample7_results_before, sample7_results_after,
                    sample13_box1_results_before, sample13_box1_results_after, sample13_box4_results_before, sample13_box4_results_after]:
        for key in results:
            results[key]['psf_total_size'] = (results[key]['sigma_x'] * results[key]['sigma_y']) ** 0.5 * 100

    # ========== Create averaged spectra for comparison ==========
    print("Creating averaged spectra...")
    # Average 100nm normalized spectra
    wavelength_100nm = list(norm_100nm.values())[0][:, 0]
    intensities_100nm = [norm_100nm[key][:, 1] for key in norm_100nm.keys()]
    averaged_intensity_100nm = np.mean(intensities_100nm, axis=0)
    averaged_100nm = np.column_stack([wavelength_100nm, averaged_intensity_100nm])
    
    # Average 200nm after normalized spectra
    wavelength_200nm = list(norm_200nm_after.values())[0][:, 0]
    intensities_200nm = [norm_200nm_after[key][:, 1] for key in norm_200nm_after.keys()]
    averaged_intensity_200nm = np.mean(intensities_200nm, axis=0)
    averaged_200nm = np.column_stack([wavelength_200nm, averaged_intensity_200nm])

    # ========== Generate plots ==========
    print("Generating plots...")
    
    # Plot APD comparisons
    plot_apd(sample6_apd, sample6_monitor, sample6_params, new_fig=True)
    plt.savefig(os.path.join(output_folder, "apd_100nm_sample6.png"))
    plt.close()

    plot_apd(sample13_box1_apd, sample13_box1_monitor, sample13_box1_params, new_fig=True)
    plt.savefig(os.path.join(output_folder, "apd_200nm_sample13_box1.png"))
    plt.close()

    # Plot normalized spectra comparison
    plot_spectra({'100nm': averaged_100nm}, {'100nm': list(params_100nm.values())[0]})
    plot_spectra({'200nm': averaged_200nm}, {'200nm': list(params_200nm_after.values())[0]}, new_fig=False, linestyle='--')
    plt.savefig(os.path.join(output_folder, "normalized_spectra_100nm_vs_200nm.png"))
    plt.close()

    # Plot raw spectra
    plot_spectra(box1_100nm_spec, params_100nm)
    plt.savefig(os.path.join(output_folder, "raw_spectra_100nm.png"))
    plt.close()

    plot_spectra(box1_200nm_after_spec, params_200nm_after)
    plt.savefig(os.path.join(output_folder, "raw_spectra_200nm_after.png"))
    plt.close()

    # Plot 5um reference comparison
    plot_spectra(ref_5um_100nm, params_100nm, linestyle='-')
    plot_spectra(ref_5um_200nm_after, params_200nm_after, new_fig=False, linestyle='--')
    plt.savefig(os.path.join(output_folder, "5um_comparison.png"))
    plt.close()

    # Plot confocal image comparisons
    plot_confocal_image_comparison(sample6_before, sample6_after)
    plt.savefig(os.path.join(output_folder, "confocal_100nm_sample6_comparison.png"))
    plt.close()

    plot_confocal_image_comparison(sample13_box1_before, sample13_box1_after)
    plt.savefig(os.path.join(output_folder, "confocal_200nm_sample13_comparison.png"))
    plt.close()

    # Plot confocal scatter comparisons
    plot_confocal_scatters(sample6_before, sample6_after, sample6_results_before, sample6_results_after, sample6_params, label="100nm Box1")
    plot_confocal_scatters(sample7_before, sample7_after, sample7_results_before, sample7_results_after, sample7_params, new_fig=False, marker='s', label="100nm Box4")
    plt.savefig(os.path.join(output_folder, "confocal_max_value_scatter_100nm.png"))
    plt.close()

    plot_confocal_scatters(sample13_box1_before, sample13_box1_after, sample13_box1_results_before, sample13_box1_results_after, sample13_box1_params, label="200nm Box1")
    plot_confocal_scatters(sample13_box4_before, sample13_box4_after, sample13_box4_results_before, sample13_box4_results_after, sample13_box4_params, new_fig=False, marker='s', label="200nm Box4")
    plt.savefig(os.path.join(output_folder, "confocal_max_value_scatter_200nm.png"))
    plt.close()

    # Combined comparison: 100nm vs 200nm
    plot_confocal_scatters(sample6_before, sample6_after, sample6_results_before, sample6_results_after, sample6_params, label="100nm")
    plot_confocal_scatters(sample13_box1_before, sample13_box1_after, sample13_box1_results_before, sample13_box1_results_after, sample13_box1_params, new_fig=False, marker='s', label="200nm")
    plt.savefig(os.path.join(output_folder, "confocal_max_value_scatter_comparison.png"))
    plt.close()

    # Plot PSF size comparisons
    plot_confocal_parameter_scatter(sample6_before, sample6_results_before, sample6_params, parameter='psf_total_size', label="100nm Before")
    plot_confocal_parameter_scatter(sample6_after, sample6_results_after, sample6_params, parameter='psf_total_size', new_fig=False, marker='s', label="100nm After")
    plt.savefig(os.path.join(output_folder, "psf_total_size_vs_power_100nm.png"))
    plt.close()

    plot_confocal_parameter_scatter(sample13_box1_before, sample13_box1_results_before, sample13_box1_params, parameter='psf_total_size', label="200nm Before")
    plot_confocal_parameter_scatter(sample13_box1_after, sample13_box1_results_after, sample13_box1_params, parameter='psf_total_size', new_fig=False, marker='s', label="200nm After")
    plt.savefig(os.path.join(output_folder, "psf_total_size_vs_power_200nm.png"))
    plt.close()

    # Combined PSF size comparison
    plot_confocal_parameter_scatter(sample6_before, sample6_results_before, sample6_params, parameter='psf_total_size', label="100nm")
    plot_confocal_parameter_scatter(sample13_box1_before, sample13_box1_results_before, sample13_box1_params, parameter='psf_total_size', new_fig=False, marker='s', label="200nm")
    plt.savefig(os.path.join(output_folder, "psf_total_size_vs_power_comparison.png"))
    plt.close()

    # Plot SNR comparisons
    plot_snr_vs_power(sample6_before, sample6_results_before, sample6_params, label="100nm Before")
    plot_snr_vs_power(sample6_after, sample6_results_after, sample6_params, new_fig=False, marker='s', label="100nm After")
    plt.savefig(os.path.join(output_folder, "snr_vs_power_100nm.png"))
    plt.close()

    plot_snr_vs_power(sample13_box1_before, sample13_box1_results_before, sample13_box1_params, label="200nm Before")
    plot_snr_vs_power(sample13_box1_after, sample13_box1_results_after, sample13_box1_params, new_fig=False, marker='s', label="200nm After")
    plt.savefig(os.path.join(output_folder, "snr_vs_power_200nm.png"))
    plt.close()

    # Combined SNR comparison
    plot_snr_vs_power(sample6_before, sample6_results_before, sample6_params, label="100nm")
    plot_snr_vs_power(sample13_box1_before, sample13_box1_results_before, sample13_box1_params, new_fig=False, marker='s', label="200nm")
    plt.savefig(os.path.join(output_folder, "snr_vs_power_comparison.png"))
    plt.close()
    
    # SNR violin plot comparison
    plot_snr_violin_comparison()
    plt.savefig(os.path.join(output_folder, "snr_violin_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # SNR violin plot comparison after irradiation
    plot_snr_violin_comparison_after()
    plt.savefig(os.path.join(output_folder, "snr_violin_comparison_after.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Noise violin plot comparison
    plot_noise_violin_comparison()
    plt.savefig(os.path.join(output_folder, "noise_violin_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # SNR before vs after comparison plot
    plot_snr_before_after_comparison()
    plt.savefig(os.path.join(output_folder, "snr_before_after_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # New box1 vs box4 comparison plots (before data only)
    # First plot 100nm to determine peak wavelength
    plot_100nm_box1_vs_box4_before()
    plt.savefig(os.path.join(output_folder, "100nm_box1_vs_box4_before.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Then plot 200nm using the same peak wavelength
    plot_200nm_box1_vs_box4_before(peak_wavelength_100nm_global)
    plt.savefig(os.path.join(output_folder, "200nm_box1_vs_box4_before.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # All spectra comparison plot with proper peak normalization (after peak is determined)
    plot_all_spectra_comparison_at_peak()
    plt.savefig(os.path.join(output_folder, "all_4_spectra_normalized_at_peak.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("Analysis complete! Check the plots/Au_200nm_vs_100nm folder for results.")
