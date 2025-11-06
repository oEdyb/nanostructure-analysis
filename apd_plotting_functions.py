import matplotlib.pyplot as plt
import apd_functions
import numpy as np
from scipy.signal import savgol_filter
import re


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

if __name__ == "__main__":
    apd_path = "./Data/APD/2025.08.21 - Sample 13 Power Threshold"
    apd_data, monitor_data, apd_params = apd_functions.apd_load_main(apd_path)
    apd_data_box1, monitor_data_box1, apd_params_box1 = apd_functions.filter_apd(apd_data, monitor_data, apd_params, "*box1*")
    plot_apd(apd_data_box1, monitor_data_box1, apd_params_box1)
    plt.show()
    plot_monitor(apd_data_box1, monitor_data_box1, apd_params_box1)
    plt.show()