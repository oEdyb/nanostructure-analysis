from nanostructure_analysis import *
import matplotlib.pyplot as plt
import numpy as np

apd_path = r"\\AMIPC045962\Cache (D)\daily_data\apd_traces\20250924 - Laser SNR"
data = apd_main(apd_path)

print(data)
print("\nAvailable measurements:", list(data.transmission.keys()))

# Calculate SNR for laser_on measurement
laser_trace = data.transmission["laser_on"]
print(f"\nStd: {np.std(laser_trace[-1000:])}")
print(f"Mean: {np.mean(laser_trace[-1000:])}")
print(f"SNR: {np.mean(laser_trace[-1000:]) / np.std(laser_trace[-1000:]):.2f}")

# Plot traces
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

for label, trace in data.transmission.items():
    time = np.arange(len(trace)) / 1e5  # Assuming 100kHz sampling
    ax1.plot(time, trace, label=label, alpha=0.7)

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Transmission (counts)')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot monitor
for label, trace in data.monitor.items():
    time = np.arange(len(trace)) / 1e5
    ax2.plot(time, trace, label=label, alpha=0.7)

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Monitor (V)')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()
