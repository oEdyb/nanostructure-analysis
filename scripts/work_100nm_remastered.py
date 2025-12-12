import os
from nanostructure_analysis import *
from nanostructure_analysis.apd_plotting_functions import *
import matplotlib.pyplot as plt

# Create output directory
output_dir = "plots/100nm_remastered"
os.makedirs(output_dir, exist_ok=True)



# Load confocal data
confocal_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.05.13-14 - Sample 7 Box 1 Illumination"

confocal_data = confocal_main(confocal_path)
confocal_perp_after = filter_confocal(confocal_data, pattern=["after", "perp"])
confocal_perp_before = filter_confocal(confocal_data, pattern=["perp"], exclude=["after"])
confocal_parallel_after = filter_confocal(confocal_data, pattern=["after"], exclude=["perp"])
confocal_parallel_before = filter_confocal(confocal_data, pattern=["*"], exclude=["after", "perp"])

# Filter for all perpendicular and parallel (no before/after distinction)
confocal_perp_all = filter_confocal(confocal_data, pattern=["perp"])
confocal_parallel_all = filter_confocal(confocal_data, pattern=["*"], exclude=["perp"])

# Load APD data
apd_path = r"\\AMIPC045962\Cache (D)\daily_data\apd_traces\2025.05.1314 Sample 7"

apd_data = apd_main(apd_path)
apd_filtered = filter_apd(apd_data, "*", exclude=["signal"])

print(apd_data)
print(apd_filtered)

# Plot APD traces
time_limit_seconds = None # set to None to plot full duration
plot_apd(apd_filtered, time_limit=time_limit_seconds, savgol=True, normalize=True, group_colors=True)
plt.savefig(f"{output_dir}/apd_transmission.png", dpi=300, bbox_inches='tight')
plt.close()

plot_apd(apd_filtered, time_limit=10, savgol=True, normalize=True, group_colors=True)
plt.savefig(f"{output_dir}/apd_transmission_10s.png", dpi=300, bbox_inches='tight')
plt.close()

plot_monitor(apd_filtered, time_limit=time_limit_seconds)
plt.savefig(f"{output_dir}/apd_monitor.png", dpi=300, bbox_inches='tight')
plt.close()

plot_monitor(apd_filtered, time_limit=10)
plt.savefig(f"{output_dir}/apd_monitor_10s.png", dpi=300, bbox_inches='tight')
plt.close()

print(apd_data.params)
# Filter for 300s traces only
def filter_by_duration(apd_data, target_duration=300, tolerance=10):
    """Filter APD data by trace duration."""
    filtered_transmission = {}
    filtered_monitor = {}
    filtered_params = {}

    for key in apd_data.transmission.keys():
        duration = apd_data.params[key]['Duration (s)']
        if abs(duration - target_duration) <= tolerance:
            filtered_transmission[key] = apd_data.transmission[key]
            filtered_monitor[key] = apd_data.monitor[key]
            filtered_params[key] = apd_data.params[key]

    from nanostructure_analysis.apd_functions import APDData
    return APDData(filtered_transmission, filtered_monitor, filtered_params)

# Filter datasets by duration (300s) and polarization
apd_300s = filter_by_duration(apd_data, target_duration=300, tolerance=10)
apd_perp_300s = filter_apd(apd_300s, "*perp*", exclude=["signal"])
apd_not_perp_300s = filter_apd(apd_300s, "*", exclude=["perp", "signal"])

# Plot both perpendicular and not perpendicular on same plot
plot_apd_by_power(
    apd_perp_300s,
    labels=["Perpendicular"],
    colors=['C0'],
    power_threshold=45,
    time_limit=time_limit_seconds,
    new_fig=True,
    savgol=True,
    title="Live Transmission Change vs. Exposure Time",
)
plot_apd_by_power(
    apd_not_perp_300s,
    labels=["Parallel"],
    colors=['C1'],
    power_threshold=45,
    time_limit=time_limit_seconds,
    new_fig=False,
    savgol=True,
    title="Live Transmission Change vs. Exposure Time",
)
plt.savefig(f"{output_dir}/apd_by_power_300s.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 10s version
plot_apd_by_power(
    apd_perp_300s,
    labels=["Perpendicular"],
    colors=['C0'],
    power_threshold=45,
    time_limit=10,
    new_fig=True,
    savgol=True,
    title="Live Transmission Change vs. Exposure Time",
)
plot_apd_by_power(
    apd_not_perp_300s,
    labels=["Parallel"],
    colors=['C1'],
    power_threshold=45,
    time_limit=10,
    new_fig=False,
    savgol=True,
    title="Live Transmission Change vs. Exposure Time",
)
plt.savefig(f"{output_dir}/apd_by_power_10s.png", dpi=300, bbox_inches='tight')
plt.close()


# Plot 1: SNR comparison perpendicular before/after
plot_snr_scatter([confocal_perp_before, confocal_perp_after],
                 labels=["Before", "After"],
                 colors=['C0', 'C1'],
                 markers=["o", "s"],
                 title="Perpendicular",
                 matching_flag=True)
plt.savefig(f"{output_dir}/snr_perp_before_after.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: SNR comparison parallel before/after
plot_snr_scatter([confocal_parallel_before, confocal_parallel_after],
                 labels=["Before", "After"],
                 colors=['C0', 'C1'],
                 markers=["o", "s"],
                 title="Parallel",
                 matching_flag=True)
plt.savefig(f"{output_dir}/snr_parallel_before_after.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: SNR vs time by power - perpendicular
plot_snr_vs_time_by_power([confocal_perp_before, confocal_perp_after], apd_data=apd_data,
                         labels=["Before", "After"],
                         colors=['C0', 'C1'],
                         markers=["o", "s"],
                         power_threshold=45,
                         title="Perpendicular")
plt.savefig(f"{output_dir}/snr_vs_time_by_power_perp.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: SNR vs time by power - parallel
plot_snr_vs_time_by_power([confocal_parallel_before, confocal_parallel_after], apd_data=apd_data,
                         labels=["Before", "After"],
                         colors=['C0', 'C1'],
                         markers=["o", "s"],
                         power_threshold=45,
                         title="Parallel")
plt.savefig(f"{output_dir}/snr_vs_time_by_power_parallel.png", dpi=300, bbox_inches='tight')
plt.close()


# Plot 7: Transmission/Power vs time - perpendicular and parallel
plot_transmission_per_power_vs_time_by_power([confocal_perp_all, confocal_parallel_all], apd_data=apd_data,
                                             labels=["Perpendicular", "Parallel"],
                                             colors=['C0', 'C1'],
                                             markers=["o", "s"],
                                             power_threshold=45,
                                             title="Transmission per Power vs Time")
plt.savefig(f"{output_dir}/transmission_per_power_vs_time.png", dpi=300, bbox_inches='tight')
plt.close()


