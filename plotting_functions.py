import matplotlib.pyplot as plt
import apd_functions
import numpy as np

def plot_apd(apd_data, monitor_data, apd_params):
    for key in apd_data.keys():
        average_power = np.mean(monitor_data[key])
        power_factor = apd_params[key]['Power calibration factor (mW/V)']
        duration = apd_params[key]['Duration (s)']
        time_axis = np.linspace(0, duration, len(apd_data[key]))
        
        plt.plot(time_axis, apd_data[key], label=f"{key} - {average_power*float(power_factor):.0f} mW")
    
    plt.xlabel('Time (s)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    apd_path = "./Data/APD/2025.08.21 - Sample 13 Power Threshold"
    apd_data, monitor_data, apd_params = apd_functions.apd_load_main(apd_path)
    plot_apd(apd_data, monitor_data, apd_params)
