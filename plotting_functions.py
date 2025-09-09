import matplotlib.pyplot as plt
import apd_functions
import numpy as np
from scipy.signal import savgol_filter
import re

import spectra_functions

def plot_apd(apd_data, monitor_data, apd_params, normalize=True, savgol=True, time=None, new_fig=True, power_factor=50):
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
    
    plt.xlabel('Time (s)', fontsize=18, fontweight='bold')
    plt.ylabel('APD Signal', fontsize=18, fontweight='bold')
    if normalize:
        plt.ylabel('Transmission Change (%)', fontsize=18, fontweight='bold')
    plt.title('APD Data', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
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
        
    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.ylabel('arb.u.', fontsize=18, fontweight='bold')
    plt.title('Spectra Data', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()

def plot_spectra_savgol(spectra_data, spectra_params, window_sizes=[11, 21, 31, 51], orders=[1, 2, 3]):

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
    
    # Plot raw data
    plt.plot(data[:, 0], data[:, 1], label=f"Raw - {first_key} - {time_str}", alpha=0.7, linewidth=1)
    
    # Plot savgol filtered versions with different windows and orders
    for window in window_sizes:
        for order in orders:
            # Apply savgol filter with current window size and order
            filtered_y = savgol_filter(data[:, 1], window, order)
            plt.plot(data[:, 0], filtered_y, label=f"Savgol w={window}, o={order}", linewidth=1, alpha=0.9)
        
    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.ylabel('arb.u.', fontsize=18, fontweight='bold')
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


def plot_confocal(confocal_dict):
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


def plot_confocal_image_comparison(confocal_dict_before, confocal_dict_after):
    
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

def plot_confocal_comparison(confocal_dict_before, confocal_dict_after):
    from scipy.optimize import curve_fit
    
    # 2D Gaussian function for fitting
    def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, offset):
        x, y = xy
        return (amplitude * np.exp(-((x-xo)**2/(2*sigma_x**2) + (y-yo)**2/(2*sigma_y**2))) + offset).ravel()
    
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
        
        # Create coordinate grids
        y, x = np.mgrid[:before_data.shape[0], :before_data.shape[1]]
        
        # Fit 2D Gaussian to before data
        try:
            initial_guess = [before_data.max(), before_data.shape[1]//2, before_data.shape[0]//2, 10, 10, before_data.min()]
            popt_before, _ = curve_fit(gaussian_2d, (x, y), before_data.ravel(), p0=initial_guess)
            center_y_before = int(popt_before[2])
        except:
            center_y_before = before_data.shape[0]//2
        
        # Fit 2D Gaussian to after data  
        try:
            initial_guess = [after_data.max(), after_data.shape[1]//2, after_data.shape[0]//2, 10, 10, after_data.min()]
            popt_after, _ = curve_fit(gaussian_2d, (x, y), after_data.ravel(), p0=initial_guess)
            center_y_after = int(popt_after[2])
        except:
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
