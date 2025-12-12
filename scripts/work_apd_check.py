from nanostructure_analysis import *
import numpy as np
import matplotlib.pyplot as plt


path = r"\\AMIPC045962\daily_data\apd_traces\2025.08.21 - Sample 13 Power Threshold\20250820_155725_box1_D6_transmission.npy"
data = np.load(path)
print(data)
plt.plot(data)
plt.show()
