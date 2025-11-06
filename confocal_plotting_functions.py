import matplotlib.pyplot as plt
import apd_functions
import numpy as np
from scipy.signal import savgol_filter
import re


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




if __name__ == "__main__":
    apd_path = "./Data/APD/2025.08.21 - Sample 13 Power Threshold"
    apd_data, monitor_data, apd_params = apd_functions.apd_load_main(apd_path)
    apd_data_box1, monitor_data_box1, apd_params_box1 = apd_functions.filter_apd(apd_data, monitor_data, apd_params, "*box1*")
