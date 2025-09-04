import matplotlib.pyplot as plt
import apd_functions
import numpy as np
from scipy.signal import savgol_filter

import spectra_functions

def plot_apd(apd_data, monitor_data, apd_params, normalize=True, savgol=True, time=None):
    plt.figure(figsize=(16, 8))
    
    # Calculate power for each key and sort by power
    power_data = []
    for key in apd_data.keys():
        average_power = np.mean(monitor_data[key])
        power_factor = apd_params[key]['Power calibration factor (mW/V)']
        power_mw = average_power * float(power_factor)
        power_data.append((power_mw, key))
    
    # Sort by power (ascending order)
    power_data.sort(key=lambda x: x[0])
    
    # Get colors from seismic colormap
    colors = plt.cm.seismic(np.linspace(0.1, 0.9, len(power_data)))
    
    # Plot in power order with colors
    for (power_mw, key), color in zip(power_data, colors):
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
        
        plt.plot(time_axis, normalized_apd, label=f"{key} - {power_mw:.0f} mW", color=color)
    
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


def plot_spectra(spectra_data, spectra_params):
    plt.figure(figsize=(16, 8))
    for key in spectra_data.keys():
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
        plt.plot(spectra_data[key][:, 0], spectra_data[key][:, 1], label=f"{key} - {time_str}")
        
    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.ylabel('a.u.', fontsize=18, fontweight='bold')
    plt.title('Spectra Data', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()


def plot_spectra_transparent(spectra_data, spectra_params):
    # Create figure with transparent background
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_alpha(0)  # Transparent figure background
    ax.patch.set_alpha(0)   # Transparent axis background
    
    for key in spectra_data.keys():
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
        plt.plot(spectra_data[key][:, 0], spectra_data[key][:, 1], label=f"{key} - {time_str}")
    
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
