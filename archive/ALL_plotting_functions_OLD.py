import matplotlib.pyplot as plt
import apd_functions
import numpy as np
from scipy.signal import savgol_filter
import re

import spectra_functions
from spectra_functions import compute_sensitivity_stat

def plot_apd(apd_data, monitor_data, apd_params, normalize=True, savgol=True, time=None, new_fig=True, power_factor=50, log_scale=False):
    if new_fig:
        plt.figure(figsize=(16, 8))
    
    # Calculate power for each key and sort by power
    power_data = []
    for key in apd_data.keys():
        average_power = np.mean(monitor_data[key])
        try:
            power_factor = apd_params[key]['Power calibration factor (mW/V)']
        except:
            print(f"Power calibration factor not found for {key}")
            power_factor = power_factor
        power_mw = average_power * float(power_factor)
        power_data.append((power_mw, key))
    
    # Sort by power (ascending order)
    power_data.sort(key=lambda x: x[0])
    

    
    # Plot in power order with colors
    for power_mw, key in power_data:
        data = apd_data[key]
        duration = apd_params[key]['Duration (s)']
        time_axis = np.linspace(0, duration, len(data))
        
        if time is not None:
            mask = time_axis <= time
            data = data[mask]
            time_axis = time_axis[mask]
        
        if normalize:
            normalized_apd = (data-(data[0]))/(data[0])*100
        else:
            normalized_apd = data
        
        if savgol:
            normalized_apd = savgol_filter(normalized_apd, 51, 3)
        
        plt.plot(time_axis, normalized_apd, label=f"{key} - {power_mw:.0f} mW")
    
    # Add reference lines at ±5% if normalized
    if normalize:
        plt.axhline(y=10, color='gray', linestyle='--', alpha=0.8, linewidth=1.5)
        plt.axhline(y=-10, color='gray', linestyle='--', alpha=0.8, linewidth=1.5)
    
    # Set logarithmic scale if requested
    if log_scale:
        plt.yscale('log')
    
    plt.xlabel('Time (s)', fontsize=18, fontweight='bold')
    plt.ylabel('APD Signal', fontsize=18, fontweight='bold')
    if normalize:
        plt.ylabel('Transmission Change (%)', fontsize=18, fontweight='bold')
    plt.title('APD Data', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.legend(loc='best', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()


def plot_monitor(apd_data, monitor_data, apd_params):
    plt.figure(figsize=(16, 8))
    for key in apd_data.keys():
        duration = apd_params[key]['Duration (s)']
        time_axis = np.linspace(0, duration, len(monitor_data[key]))
        plt.plot(time_axis, monitor_data[key], label=key)
    plt.xlabel('Time (s)', fontsize=18, fontweight='bold')
    plt.ylabel('Monitor Signal (V)', fontsize=18, fontweight='bold')
    plt.title('Monitor Data', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()


def plot_spectra(spectra_data, spectra_params, cutoff=950, new_fig=True, linestyle='-'):
    if new_fig:
        plt.figure(figsize=(16, 8))
    
    # Sort keys based on [A-D][1-6] pattern if it exists
    import re
    def get_sort_key(key):
        # Look for pattern like A1, B2, C3, D6 etc in the key
        match = re.search(r'[A-D][1-6]', key)
        if match:
            letter_part = match.group()[0]  # A, B, C, or D
            number_part = int(match.group()[1])  # 1-6
            return (ord(letter_part), number_part)  # Sort by letter first, then number
        return (999, 999)  # Put non-matching keys at the end
    
    sorted_keys = sorted(spectra_data.keys(), key=get_sort_key)
    
    # Define consistent color scheme - use colormap to handle any number of datasets
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(sorted_keys)))
    
    for i, key in enumerate(sorted_keys):
        exposure_time = spectra_params[key]['Exposure time (s)']
        # Dynamic formatting: use appropriate precision based on value magnitude
        if exposure_time >= 1:
            time_str = f"{exposure_time:.0f} s"
        elif exposure_time >= 0.1:
            time_str = f"{exposure_time:.1f} s"
        elif exposure_time >= 0.01:
            time_str = f"{exposure_time:.2f} s" 
        else:
            time_str = f"{exposure_time:.3f} s"
        
        # Filter data below cutoff wavelength
        wavelength = spectra_data[key][:, 0]
        intensity = spectra_data[key][:, 1]
        mask = wavelength <= cutoff
        
        plt.plot(wavelength[mask], intensity[mask], label=f"{key}", 
                linestyle=linestyle, color=colors[i])
        
    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.ylabel('arbitrary units', fontsize=18, fontweight='bold')
    plt.title('Spectra Data', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()

def plot_spectra_savgol(spectra_data, spectra_params, window_sizes=[11, 21, 31, 51], orders=[1, 2, 3], cutoff=950):
    from scipy.signal import savgol_filter
    
    plt.figure(figsize=(16, 8))
    
    # Get first spectrum only
    first_key = list(spectra_data.keys())[0]
    data = spectra_data[first_key]
    exposure_time = spectra_params[first_key]['Exposure time (s)']
    
    # Dynamic formatting for exposure time
    if exposure_time >= 1:
        time_str = f"{exposure_time:.0f} s"
    elif exposure_time >= 0.1:
        time_str = f"{exposure_time:.1f} s"
    elif exposure_time >= 0.01:
        time_str = f"{exposure_time:.2f} s" 
    else:
        time_str = f"{exposure_time:.3f} s"
    
    # Filter data below cutoff wavelength
    wavelength = data[:, 0]
    intensity = data[:, 1]
    mask = wavelength <= cutoff
    
    # Plot raw data
    plt.plot(wavelength[mask], intensity[mask], label=f"Raw - {first_key} - {time_str}", alpha=0.7, linewidth=2)
    
    # Plot savgol filtered versions with different windows and orders
    for window in window_sizes:
        for order in orders:
            # Apply savgol filter with current window size and order
            filtered_y = savgol_filter(intensity, window, order)
            plt.plot(wavelength[mask], filtered_y[mask], label=f"Savgol w={window}, o={order}", linewidth=1, alpha=0.9)
        
    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.ylabel('arbitrary units', fontsize=18, fontweight='bold')
    plt.title('Savgol Filter Comparison', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()


def plot_apd_transparent(apd_data, monitor_data, apd_params, normalize=True, savgol=True, time=None, power_factor=50):
    # Create figure with transparent background
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_alpha(0)  # Transparent figure background
    ax.patch.set_alpha(0)   # Transparent axis background
    
    # Calculate power for each key and sort by power
    power_data = []
    for key in apd_data.keys():
        average_power = np.mean(monitor_data[key])
        try:
            power_factor = apd_params[key]['Power calibration factor (mW/V)']
        except:
            print(f"Power calibration factor not found for {key}")
            power_factor = power_factor
        power_mw = average_power * float(power_factor)
        power_data.append((power_mw, key))
    
    # Sort by power (ascending order)
    power_data.sort(key=lambda x: x[0])
    
    # Plot in power order with colors
    for power_mw, key in power_data:
        data = apd_data[key]
        duration = apd_params[key]['Duration (s)']
        time_axis = np.linspace(0, duration, len(data))
        
        if time is not None:
            mask = time_axis <= time
            data = data[mask]
            time_axis = time_axis[mask]
        
        if normalize:
            normalized_apd = (data-(data[0]))/(data[0])*100
        else:
            normalized_apd = data
        
        if savgol:
            normalized_apd = savgol_filter(normalized_apd, 51, 3)
        
        plt.plot(time_axis, normalized_apd, label=f"{key} - {power_mw:.0f} mW")
    
    # Remove all visual elements for clean PPT insertion
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('')  # Remove y-axis label
    ax.set_title('')   # Remove title
    ax.grid(False)     # Remove grid
    ax.legend().remove() if ax.get_legend() else None  # Remove legend
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.spines['top'].set_visible(False)     # Remove top border
    ax.spines['right'].set_visible(False)   # Remove right border
    ax.spines['bottom'].set_visible(False)  # Remove bottom border
    ax.spines['left'].set_visible(False)    # Remove left border
    plt.tight_layout()


def plot_spectra_transparent(spectra_data, spectra_params, cutoff=950, new_fig=True, linestyle='-'):
    # Create figure with transparent background
    if new_fig:
        fig, ax = plt.subplots(figsize=(16, 8))
        fig.patch.set_alpha(0)  # Transparent figure background
        ax.patch.set_alpha(0)   # Transparent axis background

    
    # Sort keys based on [A-D][1-6] pattern if it exists
    def get_sort_key(key):
        # Look for pattern like A1, B2, C3, D6 etc in the key
        match = re.search(r'[A-D][1-6]', key)
        if match:
            letter_part = match.group()[0]  # A, B, C, or D
            number_part = int(match.group()[1])  # 1-6
            return (ord(letter_part), number_part)  # Sort by letter first, then number
        return (999, 999)  # Put non-matching keys at the end
    
    sorted_keys = sorted(spectra_data.keys(), key=get_sort_key)
    
    # Define consistent color scheme - expands to handle many datasets
    colors = plt.cm.tab10(range(10)) if len(sorted_keys) <= 10 else plt.cm.tab20(range(20))
    
    for i, key in enumerate(sorted_keys):
        exposure_time = spectra_params[key]['Exposure time (s)']
        # Dynamic formatting: use appropriate precision based on value magnitude
        if exposure_time >= 1:
            time_str = f"{exposure_time:.0f} s"
        elif exposure_time >= 0.1:
            time_str = f"{exposure_time:.1f} s"
        elif exposure_time >= 0.01:
            time_str = f"{exposure_time:.2f} s" 
        else:
            time_str = f"{exposure_time:.3f} s"
        
        # Filter data below cutoff wavelength
        wavelength = spectra_data[key][:, 0]
        intensity = spectra_data[key][:, 1]
        mask = wavelength <= cutoff
        
        plt.plot(wavelength[mask], intensity[mask], label=f"{key} - {time_str}", 
                linestyle=linestyle, color=colors[i % len(colors)])
    # Remove all visual elements for clean PPT insertion
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('')  # Remove y-axis label
    ax.set_title('')   # Remove title
    ax.grid(False)     # Remove grid
    ax.legend().remove() if ax.get_legend() else None  # Remove legend
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.spines['top'].set_visible(False)     # Remove top border
    ax.spines['right'].set_visible(False)   # Remove right border
    ax.spines['bottom'].set_visible(False)  # Remove bottom border
    ax.spines['left'].set_visible(False)    # Remove left border
    plt.tight_layout()


def plot_confocal(confocal_data):
    # Extract image_dict from tuple
    confocal_dict = confocal_data[0]
    for key in confocal_dict.keys():
        plt.figure(figsize=(8, 6))
        data = confocal_dict[key]
        plt.imshow(data)
        plt.colorbar()
        plt.title(key, fontsize=22, fontweight='bold')
        plt.xlabel('X (px)', fontsize=18, fontweight='bold')
        plt.ylabel('Y (px)', fontsize=18, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()
        plt.show()
        plt.close()


def plot_confocal_image_comparison(confocal_data_before, confocal_data_after):
    # Extract image_dicts from tuples
    confocal_dict_before = confocal_data_before[0]
    confocal_dict_after = confocal_data_after[0]
    
    # Find common [A-D][1-6] patterns between datasets
    pattern = r'[A-D][1-6]'
    before_locs = {re.search(pattern, k).group(): k for k in confocal_dict_before.keys() if re.search(pattern, k)}
    after_locs = {re.search(pattern, k).group(): k for k in confocal_dict_after.keys() if re.search(pattern, k)}
    
    # Plot only matching locations
    for loc in sorted(set(before_locs.keys()) & set(after_locs.keys())):
        plt.figure(figsize=(12, 5))
        
        # Get data for both images to determine shared colorbar range
        before_data = confocal_dict_before[before_locs[loc]]
        after_data = confocal_dict_after[after_locs[loc]]
        
        # Calculate shared min/max for consistent colorbar
        vmin = min(before_data.min(), after_data.min())
        vmax = max(before_data.max(), after_data.max())
        
        # Before image
        plt.subplot(1, 2, 1)
        plt.imshow(before_data, vmin=vmin, vmax=vmax)
        plt.title(f'Before - {loc}', fontsize=22, fontweight='bold')
        plt.xlabel('X (px)', fontsize=18, fontweight='bold')
        plt.ylabel('Y (px)', fontsize=18, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()  

        # After image
        plt.subplot(1, 2, 2)
        plt.imshow(after_data, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(f'After - {loc}', fontsize=22, fontweight='bold')
        plt.xlabel('X (px)', fontsize=18, fontweight='bold')
        plt.ylabel('Y (px)', fontsize=18, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()


def plot_confocal_comparison(confocal_data_before, confocal_data_after, popt_before=None, popt_after=None):
    # Extract image_dicts from tuples
    confocal_dict_before = confocal_data_before[0]
    confocal_dict_after = confocal_data_after[0]
    
    # Find common [A-D][1-6] patterns between datasets
    pattern = r'[A-D][1-6]'
    before_locs = {re.search(pattern, k).group(): k for k in confocal_dict_before.keys() if re.search(pattern, k)}
    after_locs = {re.search(pattern, k).group(): k for k in confocal_dict_after.keys() if re.search(pattern, k)}
    
    # Plot only matching locations
    for loc in sorted(set(before_locs.keys()) & set(after_locs.keys())):
        plt.figure(figsize=(10, 8))
        
        # Get data
        before_data = confocal_dict_before[before_locs[loc]]
        after_data = confocal_dict_after[after_locs[loc]]
        
        # Calculate shared min/max for consistent colorbar
        vmin = min(before_data.min(), after_data.min())
        vmax = max(before_data.max(), after_data.max())
        
        # Extract center positions from provided popt parameters
        # popt format: [amplitude, xo, yo, sigma_x, sigma_y, offset]
        if popt_before is not None:
            center_y_before = int(popt_before[before_locs[loc]][2])
        else:
            center_y_before = before_data.shape[0]//2
            
        if popt_after is not None:
            center_y_after = int(popt_after[after_locs[loc]][2])
        else:
            center_y_after = after_data.shape[0]//2
        
        # Before image (top left)
        plt.subplot(2, 2, 1)
        plt.imshow(before_data, vmin=vmin, vmax=vmax)
        plt.axhline(center_y_before, color='red', linestyle='--')
        plt.title(f'Before - {loc}', fontsize=18, fontweight='bold')
        plt.xlabel('X (px)', fontsize=14, fontweight='bold')
        plt.ylabel('Y (px)', fontsize=14, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # After image (top right)
        plt.subplot(2, 2, 2)
        plt.imshow(after_data, vmin=vmin, vmax=vmax)
        plt.axhline(center_y_after, color='red', linestyle='--')
        plt.colorbar()
        plt.title(f'After - {loc}', fontsize=18, fontweight='bold')
        plt.xlabel('X (px)', fontsize=14, fontweight='bold')
        plt.ylabel('Y (px)', fontsize=14, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=12)
        # X cross sections (bottom, spanning both columns)
        plt.subplot(2, 1, 2)
        plt.plot(before_data[center_y_before, :], label='Before', linewidth=2)
        plt.plot(after_data[center_y_after, :], label='After', linewidth=2)
        plt.xlabel('X (px)', fontsize=14, fontweight='bold')
        plt.ylabel('Intensity', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.title('X Cross Section', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()

def plot_confocal_scatters(confocal_data_before, confocal_data_after, results_before, results_after, apd_params, new_fig=True, marker='o', label=None):
    # Extract image_dicts from tuples
    confocal_dict_before = confocal_data_before[0]
    confocal_dict_after = confocal_data_after[0]

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
    
    # Get locations that exist in all three datasets
    locations = sorted(set(before_locs.keys()) & set(after_locs.keys()) & set(apd_locs.keys()))
    
    # Extract max values for matching locations and calculate percent change
    before_max = [results_before[before_locs[loc]]['max_value'] for loc in locations]
    after_max = [results_after[after_locs[loc]]['max_value'] for loc in locations]
    # Calculate percent change: (after - before) / before * 100
    percent_change = [(after - before) / before * 100 for before, after in zip(before_max, after_max)]
    powers = [apd_locs[loc] for loc in locations]
    
    # Create new figure only if requested
    if new_fig:
        plt.figure(figsize=(10, 6))

    # Add horizontal line at y=0 for reference (no change) - only on new figure
    if new_fig:
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='No change')
    
    plt.scatter(powers, percent_change, s=100, alpha=0.7, marker=marker, edgecolors='black', linewidths=1, label=label)
    
    # # Add labels for each point
    # for i, loc in enumerate(locations):
    #     plt.annotate(loc, (powers[i], percent_change[i]), fontsize=10, ha='center', va='bottom')
    

    
    # Set labels and formatting - only on new figure
    if new_fig:
        plt.xlabel('Power (mW)', fontsize=14, fontweight='bold')
        plt.ylabel('Max Value Change (%)', fontsize=14, fontweight='bold')
        plt.title('Max Value Change vs Power', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
    plt.legend(fontsize=12)


def plot_confocal_snr(confocal_data, results_dict, apd_params, new_fig=True, marker='o', label=None):
    # Extract image_dict from tuple
    confocal_dict = confocal_data[0]

    # Find common [A-D][1-6] patterns
    pattern = r'[A-D][1-6]'
    confocal_locs = {re.search(pattern, k).group(): k for k in confocal_dict.keys() if re.search(pattern, k)}
    
    # Find matching patterns in APD params and extract power values
    apd_locs = {}
    for k in apd_params.keys():
        match = re.search(pattern, k)
        if match:
            apd_locs[match.group()] = apd_params[k]['power']
    
    # Get locations that exist in both datasets
    locations = sorted(set(confocal_locs.keys()) & set(apd_locs.keys()))
    
    # Extract SNR values for matching locations
    snr_values = []
    powers = []
    for loc in locations:
        confocal_key = confocal_locs[loc]
        if confocal_key in results_dict and 'snr_3x3' in results_dict[confocal_key]:
            snr_values.append(results_dict[confocal_key]['snr_3x3'])
            powers.append(apd_locs[loc])
    
    # Create new figure only if requested
    if new_fig:
        plt.figure(figsize=(10, 6))
    
    plt.scatter(powers, snr_values, s=100, alpha=0.7, marker=marker, edgecolors='black', linewidths=1, label=label)
    
    # Set labels and formatting - only on new figure
    if new_fig:
        plt.xlabel('Power (mW)', fontsize=14, fontweight='bold')
        plt.ylabel('SNR (3x3 center)', fontsize=14, fontweight='bold')
        plt.title('SNR vs Power', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
    plt.legend(fontsize=12)


def plot_snr_before_after(confocal_before, confocal_after, results_before, results_after, new_fig=True, marker='o', label=None):
    # Extract image_dicts from tuples
    confocal_dict_before = confocal_before[0]
    confocal_dict_after = confocal_after[0]

    # Find common [A-D][1-6] patterns between datasets
    pattern = r'[A-D][1-6]'
    before_locs = {re.search(pattern, k).group(): k for k in confocal_dict_before.keys() if re.search(pattern, k)}
    after_locs = {re.search(pattern, k).group(): k for k in confocal_dict_after.keys() if re.search(pattern, k)}
    
    # Get locations that exist in both datasets
    locations = sorted(set(before_locs.keys()) & set(after_locs.keys()))
    
    # Extract SNR values for matching locations
    snr_before = []
    snr_after = []
    for loc in locations:
        before_key = before_locs[loc]
        after_key = after_locs[loc]
        
        if before_key in results_before and 'snr_3x3' in results_before[before_key]:
            if after_key in results_after and 'snr_3x3' in results_after[after_key]:
                snr_before.append(results_before[before_key]['snr_3x3'])
                snr_after.append(results_after[after_key]['snr_3x3'])
    
    # Create new figure only if requested
    if new_fig:
        plt.figure(figsize=(8, 8))
    
    plt.scatter(snr_before, snr_after, s=100, alpha=0.7, marker=marker, edgecolors='black', linewidths=1, label=label)
    
    # Set labels and formatting - only on new figure
    if new_fig:
        # Calculate max SNR for equal axis limits
        all_snr = snr_before + snr_after
        max_snr = max(all_snr) if all_snr else 1
        max_lim = max_snr * 1.1  # Add 10% padding
        
        # Add diagonal line for reference (no change)
        plt.plot([0, max_lim], [0, max_lim], 'k--', alpha=0.5, label='No change')
        
        plt.xlabel('SNR Before', fontsize=14, fontweight='bold')
        plt.ylabel('SNR After', fontsize=14, fontweight='bold')
        plt.title('SNR Before vs After', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # Set equal axes starting from 0
        plt.xlim(0, max_lim)
        plt.ylim(0, max_lim)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
    plt.legend(fontsize=12)


def plot_snr_vs_power(confocal_data, results_dict, apd_params, new_fig=True, marker='o', label=None):
    # Extract image_dict from tuple
    confocal_dict = confocal_data[0]

    # Find common [A-D][1-6] patterns
    pattern = r'[A-D][1-6]'
    confocal_locs = {re.search(pattern, k).group(): k for k in confocal_dict.keys() if re.search(pattern, k)}
    
    # Find matching patterns in APD params and extract power values
    apd_locs = {}
    for k in apd_params.keys():
        match = re.search(pattern, k)
        if match:
            apd_locs[match.group()] = apd_params[k]['power']
    
    # Get locations that exist in both datasets
    locations = sorted(set(confocal_locs.keys()) & set(apd_locs.keys()))
    
    # Extract SNR values for matching locations
    snr_values = []
    powers = []
    for loc in locations:
        confocal_key = confocal_locs[loc]
        if confocal_key in results_dict and 'snr_3x3' in results_dict[confocal_key]:
            snr_values.append(results_dict[confocal_key]['snr_3x3'])
            powers.append(apd_locs[loc])
    
    # Create new figure only if requested
    if new_fig:
        plt.figure(figsize=(10, 6))
    
    scatter = plt.scatter(powers, snr_values, s=100, alpha=0.7, marker=marker, edgecolors='black', linewidths=1, label=label)
    
    # Fit trend line excluding outliers from fit only (z-score > 2)
    if len(powers) > 2:
        from scipy import stats
        z_scores = np.abs(stats.zscore(snr_values))
        fit_mask = z_scores < 2  # Use only non-outliers for fitting
        if np.sum(fit_mask) > 1:
            x_fit = np.array(powers)[fit_mask]
            y_fit = np.array(snr_values)[fit_mask]
            coeffs = np.polyfit(x_fit, y_fit, 1)
            x_line = np.linspace(min(powers), max(powers), 100)
            y_line = np.polyval(coeffs, x_line)
            plt.plot(x_line, y_line, '--', color=scatter.get_facecolors()[0], alpha=0.8, linewidth=2)
    
    # Set labels and formatting - only on new figure
    if new_fig:
        plt.xlabel('Power (mW)', fontsize=14, fontweight='bold')
        plt.ylabel('SNR (3x3 center)', fontsize=14, fontweight='bold')
        plt.title('SNR vs Power', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
    plt.legend(fontsize=12)


def plot_confocal_parameter_scatter(confocal_data, results_dict, apd_params, parameter='snr_3x3', new_fig=True, marker='o', label=None):
    """Generic scatter plot for any confocal parameter vs power"""
    # Extract image_dict from tuple
    confocal_dict = confocal_data[0]

    # Find common [A-D][1-6] patterns
    pattern = r'[A-D][1-6]'
    confocal_locs = {re.search(pattern, k).group(): k for k in confocal_dict.keys() if re.search(pattern, k)}
    
    # Find matching patterns in APD params and extract power values
    apd_locs = {}
    for k in apd_params.keys():
        match = re.search(pattern, k)
        if match:
            apd_locs[match.group()] = apd_params[k]['power']
    
    # Get locations that exist in both datasets
    locations = sorted(set(confocal_locs.keys()) & set(apd_locs.keys()))
    
    # Extract parameter values for matching locations
    param_values = []
    powers = []
    for loc in locations:
        confocal_key = confocal_locs[loc]
        if confocal_key in results_dict and parameter in results_dict[confocal_key]:
            param_values.append(results_dict[confocal_key][parameter])
            powers.append(apd_locs[loc])
    
    # Create new figure only if requested
    if new_fig:
        plt.figure(figsize=(10, 6))
    
    plt.scatter(powers, param_values, s=100, alpha=0.7, marker=marker, edgecolors='black', linewidths=1, label=label)
    
    # Set labels and formatting - only on new figure
    if new_fig:
        plt.xlabel('Power (mW)', fontsize=14, fontweight='bold')
        ylabel = parameter.replace('_', ' ').title()
        if 'psf' in parameter.lower() and 'size' in parameter.lower():
            ylabel += ' (nm)'
        plt.ylabel(ylabel, fontsize=14, fontweight='bold')
        plt.title(f'{parameter.replace("_", " ").title()} vs Power', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
    plt.legend(fontsize=12)


def plot_sem_vs_power(csv_path, apd_params, parameter='gap_width', box_filter=None, new_fig=True, default_box=None):
    """Plot SEM parameters vs power from CSV and APD params"""
    
    # Read CSV manually to avoid pandas issues
    import csv
    
    # Create a mapping from CSV entries to power values
    # Use the same approach as plot_apd: iterate through apd_params keys
    powers, values, boxes = [], [], []
    
    # Build CSV lookup dictionary first
    csv_data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        pattern_with_box = r'box([14])_([A-D][1-6])'
        pattern_location_only = r'([A-D][1-6])'
        
        for row in reader:
            if float(row[parameter]) <= 0:  # Skip zero values
                continue
            
            # SUPER IMPORTANT: Skip tilted entries
            if '_tilted_' in row['label']:
                continue
                
            # Try pattern with box number first
            match = re.search(pattern_with_box, row['label'])
            if match:
                box, loc = match.groups()
            else:
                # Try pattern with location only
                match = re.search(pattern_location_only, row['label'])
                if match and default_box:
                    loc = match.group(1)
                    box = default_box
                else:
                    continue
            
            # Skip C1 and C2 for box1
            if box == '1' and loc in ['C1', 'C2']:
                continue
            # Filter by box if specified
            if box_filter and f'Box{box}' != box_filter:
                continue
                
            # Store the data with location as key
            csv_data[loc] = {
                'value': float(row[parameter]),
                'box': box
            }
    
    # Now iterate through APD params like plot_apd does
    for key in apd_params.keys():
        if key in csv_data and 'power' in apd_params[key]:
            power = apd_params[key]['power']
            try:
                power_val = float(power)
                if power_val > 0.01:
                    powers.append(power_val)
                    values.append(csv_data[key]['value'])
                    boxes.append(f"Box{csv_data[key]['box']}")
            except (ValueError, TypeError):
                continue
    
    if new_fig:
        plt.figure(figsize=(10, 6))
    
    # Plot by box with different markers
    for box in ['Box1', 'Box4']:
        box_powers = [p for p, b in zip(powers, boxes) if b == box]
        box_values = [v for v, b in zip(values, boxes) if b == box]
        if box_powers:
            plt.scatter(box_powers, box_values, label=box, s=100, alpha=0.7, 
                       marker='o' if box == 'Box1' else 's', edgecolors='black', linewidths=1)
    
    if new_fig:
        box_title = f' - {box_filter}' if box_filter else ''
        plt.xlabel('Power (mW)', fontsize=14, fontweight='bold')
        plt.ylabel(parameter.replace('_', ' ').title() + ' (nm)', fontsize=14, fontweight='bold')
        plt.title(f'{parameter.replace("_", " ").title()} vs Power{box_title}', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
    plt.legend(fontsize=12)

def plot_spectra_sensitivty(spectra_data):
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

def plot_spectra_at_wavelength(spectra_data, target_wavelength=852):
    # Extract intensity values at specific wavelength (852 nm)
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


def plot_sensitivity_points_by_group(
    group_value_store,
    output_path,
    group_stats=None,
    xtick_formatter=None,
    secondary_group_values=None,
    secondary_ylabel='SNR'
):
    """Violin distribution per group for sensitivity (and optional secondary metric)."""
    if not group_value_store:
        return

    groups = sorted(group_value_store)
    datasets = [('Sensitivity (mV/nm)', group_value_store)]
    if secondary_group_values:
        datasets.append((secondary_ylabel, secondary_group_values))

    stats = group_stats or {}
    colors = ['#2C7BB6', '#ABD9E9', '#FDAE61', '#D7191C']
    color_map = {grp: colors[i % len(colors)] for i, grp in enumerate(groups)}

    fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 5), squeeze=False)
    axes = axes.ravel()

    for ax, (ylabel, value_store) in zip(axes, datasets):
        values_per_group = [np.asarray(value_store.get(group, []), dtype=float) for group in groups]
        violins = ax.violinplot(
            values_per_group,
            positions=np.arange(len(groups)),
            widths=0.6,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )
        for body, group in zip(violins['bodies'], groups):
            color = color_map[group]
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.45)

        for idx, (group, values) in enumerate(zip(groups, values_per_group)):
            if not values.size:
                continue
            median = float(np.median(values))
            spread = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            ax.fill_between([idx - 0.16, idx + 0.16], median - spread, median + spread, color=color_map[group], alpha=0.14, zorder=1)
            ax.hlines([median - spread, median + spread], idx - 0.12, idx + 0.12, colors='#555555', linestyles='--', linewidth=1.0, alpha=0.9, zorder=2)
            ax.scatter(idx, median, s=55, color='#111111', marker='D', edgecolors='white', linewidths=0.7, zorder=3)

        labels = [
            (
                f"{group}\nGap {stats[group]['gap_mean']:.1f}±{stats[group].get('gap_std', 0.0):.1f} nm\n"
                f"Tip {stats[group]['tip_mean']:.3f}±{stats[group].get('tip_std', 0.0):.3f}"
            ) if group in stats else (xtick_formatter(group) if xtick_formatter else group)
            for group in groups
        ]
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlim(-0.5, len(groups) - 0.5)
        ax.set_xlabel('Group', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelsize=11)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.grid(axis='y', color='#E4E4E4', linewidth=0.7, alpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_sensitivity_vs_sem_metrics_group(
    group_values,
    group_stats,
    output_path,
    xlabel_gap='Gap Mean (nm)',
    xlabel_tip='Tip Curvature Mean',
    ylabel='Sensitivity (mV/nm)',
    secondary_group_values=None,
    secondary_ylabel='SNR'
):
    """Scatter sensitivity per group vs. gap/tip metrics with error bars."""
    if not group_values or not group_stats:
        return

    groups = sorted(set(group_values) & set(group_stats))
    if not groups:
        return

    datasets = [(group_values, ylabel)]
    if secondary_group_values:
        datasets.append((secondary_group_values, secondary_ylabel))

    colors = ['#2C7BB6', '#ABD9E9', '#FDAE61', '#D7191C']
    color_map = {grp: colors[i % len(colors)] for i, grp in enumerate(groups)}

    gap_mean = np.asarray([group_stats[g]['gap_mean'] for g in groups], dtype=float)
    gap_std = np.asarray([group_stats[g].get('gap_std', 0.0) for g in groups], dtype=float)
    tip_mean = np.asarray([group_stats[g]['tip_mean'] for g in groups], dtype=float)
    tip_std = np.asarray([group_stats[g].get('tip_std', 0.0) for g in groups], dtype=float)

    fig, axes = plt.subplots(len(datasets), 2, figsize=(10, 4.8 * len(datasets)), squeeze=False)

    for row, (values_dict, row_label) in enumerate(datasets):
        values_per_group = [np.asarray(values_dict.get(group, []), dtype=float) for group in groups]
        medians = np.asarray([float(np.median(vals)) if vals.size else np.nan for vals in values_per_group], dtype=float)
        spreads = np.asarray([float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0 for vals in values_per_group], dtype=float)

        for col, (x_vals, x_err, xlabel) in enumerate(((gap_mean, gap_std, xlabel_gap), (tip_mean, tip_std, xlabel_tip))):
            ax = axes[row, col]
            for idx, group in enumerate(groups):
                y_val = medians[idx]
                if np.isnan(y_val):
                    continue
                color = color_map[group]
                ax.errorbar(
                    x_vals[idx],
                    y_val,
                    yerr=spreads[idx],
                    xerr=x_err[idx],
                    fmt='o',
                    color=color,
                    ecolor=color,
                    capsize=4,
                    markersize=8
                )
                ax.annotate(group, (x_vals[idx], y_val), textcoords="offset points", xytext=(4, 6), fontsize=9)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.grid(color='#E4E4E4', linewidth=0.7, alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axes[row, 0].set_ylabel(row_label, fontsize=12, fontweight='bold')

    fig.suptitle('Metrics vs. Group Statistics', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_sensitivity_vs_sem_metrics_points(
    measurement_series,
    sem_measurements,
    output_path,
    metrics=None,
    sensitivity_fn=None,
    ylabel='Sensitivity P95 |mV/nm|',
    secondary_measurement_values=None,
    secondary_ylabel='SNR 3x3'
):
    """Scatter per-measurement sensitivity vs. SEM metrics with optional regression."""
    if not measurement_series or not sem_measurements:
        return

    shared_ids = [mid for mid in sorted(measurement_series) if mid in sem_measurements]
    if not shared_ids:
        return

    metrics = metrics or [('gap_mean', 'Gap Mean (nm)'), ('tip_mean', 'Tip Curvature Mean')]
    sensitivity_fn = sensitivity_fn or compute_sensitivity_stat

    color_cycle = ['#2C7BB6', '#ABD9E9', '#FDAE61', '#D7191C']
    group_colors = {}
    entries = []
    for measurement_id in shared_ids:
        values = np.asarray(measurement_series[measurement_id].get('values', []), dtype=float)
        if not values.size:
            continue
        group = measurement_series[measurement_id].get('group') or 'Unknown'
        entry = {
            'id': measurement_id,
            'group': group,
            'sensitivity': float(sensitivity_fn(values)),
            'sem': sem_measurements[measurement_id]
        }
        if secondary_measurement_values and measurement_id in secondary_measurement_values:
            entry['secondary'] = float(secondary_measurement_values[measurement_id])
        entries.append(entry)
        if group not in group_colors:
            group_colors[group] = color_cycle[len(group_colors) % len(color_cycle)]

    if not entries:
        return

    rows = 2 if any('secondary' in entry for entry in entries) else 1
    fig, axes = plt.subplots(rows, len(metrics), figsize=(5.5 * len(metrics), 4.8 * rows), squeeze=False)

    def add_correlation(ax, x_vals, y_vals):
        if len(x_vals) < 2:
            return
        x = np.asarray(x_vals, dtype=float)
        y = np.asarray(y_vals, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size < 2 or np.unique(x).size < 2:
            return
        coeff = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.02, 0.95, f'r = {coeff:.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
        try:
            slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(x.min(), x.max(), 100)
            ax.plot(grid, slope * grid + intercept, color='#444444', linewidth=1.1, alpha=0.6)
        except np.linalg.LinAlgError:
            pass

    def render_row(row_idx, value_key, row_label):
        for col, metric_cfg in enumerate(metrics):
            metric_key, metric_label = metric_cfg[:2]
            show_corr = bool(metric_cfg[2]) if len(metric_cfg) > 2 else False
            ax = axes[row_idx, col]
            x_vals, y_vals = [], []
            for entry in entries:
                if value_key not in entry:
                    continue
                x_val = entry['sem'].get(metric_key)
                y_val = entry[value_key]
                if x_val is None or y_val is None:
                    continue
                color = group_colors.get(entry['group'], '#333333')
                ax.scatter(x_val, y_val, color=color, s=50)
                ax.annotate(entry['id'], (x_val, y_val), textcoords="offset points", xytext=(4, 5), fontsize=8)
                x_vals.append(x_val)
                y_vals.append(y_val)
            if show_corr:
                add_correlation(ax, x_vals, y_vals)
            ax.set_xlabel(metric_label, fontsize=12)
            ax.grid(color='#E4E4E4', linewidth=0.7, alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        axes[row_idx, 0].set_ylabel(row_label, fontsize=12, fontweight='bold')

    render_row(0, 'sensitivity', ylabel)
    if rows == 2:
        render_row(1, 'secondary', secondary_ylabel)

    fig.suptitle('Per-Measurement Metrics vs. SEM Features', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_confocal_snr_histogram(results_dict, output_path, bins=5):
    """Histogram of 3x3 SNR values extracted from confocal analysis."""
    snrs = [float(entry['snr_3x3']) for entry in results_dict.values() if 'snr_3x3' in entry]
    if not snrs:
        return

    snr_array = np.asarray(snrs, dtype=float)
    mean_val = float(np.mean(snr_array))
    std_val = float(np.std(snr_array))

    fig, ax = plt.subplots()
    ax.hist(snr_array, bins=bins, edgecolor='black')
    ax.set_xlabel('SNR', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title(f'SNR Distribution\nMean: {mean_val:.2f}, Std: {std_val:.2f}', fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_confocal_snr_violin(
    results_dict,
    output_path,
    label_pattern=r'[A-D][1-6]',
    jitter_scale=0.04
):
    """Single violin plot with jittered points and labels for SNR values."""
    pattern = re.compile(label_pattern)
    entries = [
        (pattern.search(key).group(), float(value['snr_3x3']))
        for key, value in results_dict.items()
        if 'snr_3x3' in value and pattern.search(key)
    ]
    if not entries:
        return

    entries.sort()
    labels, snr_values = zip(*entries)
    snr_array = np.asarray(snr_values, dtype=float)
    mean_val = float(np.mean(snr_array))
    std_val = float(np.std(snr_array))

    fig, ax = plt.subplots(figsize=(8, 6))
    violin = ax.violinplot([snr_array], positions=[0], widths=0.7, showmeans=True, showmedians=True)
    for body in violin.get('bodies', []):
        body.set_facecolor('dodgerblue')
        body.set_edgecolor('black')
        body.set_alpha(0.7)
    for element in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
        if element in violin:
            violin[element].set_color('black')
            violin[element].set_linewidth(1)

    jitter = np.random.normal(0, jitter_scale, len(snr_array))
    ax.scatter(jitter, snr_array, s=70, color='dodgerblue', edgecolors='black', linewidths=1.2, alpha=0.8)
    for x_val, y_val, label in zip(jitter, snr_array, labels):
        ax.text(x_val + 0.01, y_val, label, fontsize=8, va='center')

    ax.set_ylabel('SNR', fontsize=16, fontweight='bold')
    ax.set_title('SNR Distribution', fontsize=18, fontweight='bold')
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlim(-0.5, 0.5)

    y_min, y_max = snr_array.min(), snr_array.max()
    pad = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.text(0, y_max + pad * 0.1, f'μ={mean_val:.2f}\nσ={std_val:.2f}', ha='center', va='bottom', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_peak_wavelength_histogram(spectra_data, output_path, bins=20):
    """Plot distribution of peak wavelengths across spectra."""
    if not spectra_data:
        return

    peak_wavelengths = [
        spectrum[np.argmax(spectrum[:, 1]), 0]
        for spectrum in spectra_data.values()
        if spectrum.size
    ]
    if not peak_wavelengths:
        return

    peak_array = np.asarray(peak_wavelengths, dtype=float)
    mean_peak = float(np.mean(peak_array))
    std_peak = float(np.std(peak_array))

    fig = plt.figure()
    plt.hist(peak_array, bins=bins, edgecolor='black')
    plt.xlabel('Peak Wavelength (nm)')
    plt.ylabel('Count')
    plt.title(f'Distribution of Peak Wavelengths\nMean: {mean_peak:.1f} nm, Std: {std_peak:.1f} nm')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    apd_path = "./Data/APD/2025.08.21 - Sample 13 Power Threshold"
    apd_data, monitor_data, apd_params = apd_functions.apd_load_main(apd_path)
    apd_data_box1, monitor_data_box1, apd_params_box1 = apd_functions.filter_apd(apd_data, monitor_data, apd_params, "*box1*")
    plot_apd(apd_data_box1, monitor_data_box1, apd_params_box1)
    plt.show()
    plot_monitor(apd_data_box1, monitor_data_box1, apd_params_box1)
    plt.show()

    spectra_path = "./Data/Spectra/20250821 - sample13 - after"
    spectra_data, spectra_params = spectra_functions.spectra_main(spectra_path)
    spectra_data_box1, spectra_params_box1 = spectra_functions.filter_spectra(spectra_data, spectra_params, "*box1*")
    spectra_data_5um, spectra_params_5um = spectra_functions.filter_spectra(spectra_data, spectra_params, "*5um*")
    spectra_data_bkg, spectra_params_bkg = spectra_functions.filter_spectra(spectra_data, spectra_params, "*bkg*")
    plot_spectra(spectra_data_box1, spectra_params_box1)
    plt.savefig("spectra_box1.png")
    plt.show()
    plot_spectra(spectra_data_5um, spectra_params_5um)
    plt.savefig("spectra_5um.png")
    plt.show()
    plot_spectra(spectra_data_bkg, spectra_params_bkg)
    plt.savefig("spectra_bkg.png")
    plt.show()
