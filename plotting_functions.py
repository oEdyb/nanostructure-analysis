import matplotlib.pyplot as plt
import apd_functions
import numpy as np

def plot_apd(apd_data, monitor_data, apd_params, normalize=True):
    plt.figure(figsize=(16, 8))
    for key in apd_data.keys():

        if normalize:
            normalized_apd = (apd_data[key]-(apd_data[key][0]))/(apd_data[key][0])*100
        else:
            normalized_apd = apd_data[key]

        average_power = np.mean(monitor_data[key])
        power_factor = apd_params[key]['Power calibration factor (mW/V)']
        duration = apd_params[key]['Duration (s)']
        time_axis = np.linspace(0, duration, len(apd_data[key]))
        
        plt.plot(time_axis, normalized_apd, label=f"{key} - {average_power*float(power_factor):.0f} mW")
    
    plt.xlabel('Time (s)', fontsize=18, fontweight='bold')
    plt.ylabel('APD Signal', fontsize=18, fontweight='bold')
    if normalize:
        plt.ylabel('APD Signal (%)', fontsize=18, fontweight='bold')
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


if __name__ == "__main__":
    apd_path = "./Data/APD/2025.08.21 - Sample 13 Power Threshold"
    apd_data, monitor_data, apd_params = apd_functions.apd_load_main(apd_path)
    plot_apd(apd_data, monitor_data, apd_params)
    plt.show()
    plot_monitor(apd_data, monitor_data, apd_params)
    plt.show()
