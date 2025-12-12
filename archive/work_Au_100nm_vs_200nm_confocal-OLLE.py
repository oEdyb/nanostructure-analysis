from apd_functions import * 
from spectra_functions import *
from ALL_plotting_functions_OLD import *
from confocal_functions import *
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import re
import pandas as pd
import itertools



# ==============================================================================
# DATA PATHS CONFIGURATION
# ==============================================================================

# 100nm samples (Sample 6 and Sample 7)
sample6_apd_path = r"Data\APD\2025.06.11 - Sample 6 Power Threshold"
sample7_apd_path = r"Data\APD\2025.06.03 - Power Threshold Sample 7 box 4"
sample6_confocal_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.06.11 - Sample 6 Power Threshold"
sample7_confocal_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.06.03 - Power Threshold Sample 7 box 4"
# 200nm samples (Sample 13 - normal and high power threshold measurements)
sample13_apd_path = r"Data\APD\2025.08.21 - Sample 13 Power Threshold"
sample13_pt_apd_path = r"Data\APD\2025.09.02 - Sample 13 PT high power"
sample13_confocal_path = r"Data\Confocal\2025.08.21 - Sample 13 Power Threshold box1"
sample13_pt_confocal_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.09.02 - Sample 13 before after break"

# ==============================================================================
# LOAD AND PROCESS 100nm APD DATA
# ==============================================================================
print("Loading 100nm APD data...")

# Load Sample 6 APD data (100nm thickness)
sample6_apd, sample6_monitor, sample6_params = apd_load_main(sample6_apd_path)
sample6_apd, sample6_monitor, sample6_params = filter_apd(
    sample6_apd, sample6_monitor, sample6_params, "*[A-D][1-6]*"
)

# Load Sample 7 APD data (100nm thickness) - exclude problematic C4 measurement
sample7_apd, sample7_monitor, sample7_params = apd_load_main(sample7_apd_path)
sample7_apd, sample7_monitor, sample7_params = filter_apd(
    sample7_apd, sample7_monitor, sample7_params, "*[A-D][1-6]*", exclude=["C4"]
)

# Fix missing power parameters for high power measurements (100mW)
for key in sample6_params.keys():
    if 'D6' in key:
        sample6_params[key]['power'] = 100.0

for key in sample7_params.keys():
    if 'C6' in key:
        sample7_params[key]['power'] = 100.0

# ==============================================================================
# LOAD AND PROCESS 200nm APD DATA  
# ==============================================================================
print("Loading 200nm APD data...")

# Load Sample 13 normal power threshold data
sample13_apd, sample13_monitor, sample13_params = apd_load_main(sample13_apd_path)

# Load Sample 13 high power threshold data 
sample13_pt_apd, sample13_pt_monitor, sample13_pt_params = apd_load_main(sample13_pt_apd_path)

# Combine normal and high power threshold datasets for comprehensive analysis
combined_13_apd = {**sample13_apd, **sample13_pt_apd}
combined_13_monitor = {**sample13_monitor, **sample13_pt_monitor}
combined_13_params = {**sample13_params, **sample13_pt_params}

# Separate data by box regions for focused analysis
sample13_box1_apd, sample13_box1_monitor, sample13_box1_params = filter_apd(
    combined_13_apd, combined_13_monitor, combined_13_params, "*box1*"
)
sample13_box4_apd, sample13_box4_monitor, sample13_box4_params = filter_apd(
    combined_13_apd, combined_13_monitor, combined_13_params, "*box4*[!_D4_*]*"
)

# ==============================================================================
# LOAD AND PROCESS CONFOCAL DATA
# ==============================================================================
print("Loading confocal data...")

# Load 100nm confocal data with caching for faster processing
sample6_confocal_data = load_with_cache(sample6_confocal_path, confocal_main)
sample7_confocal_data = load_with_cache(sample7_confocal_path, confocal_main)

# Separate before/after irradiation images for 100nm samples
sample6_before = filter_confocal(sample6_confocal_data, "*", exclude=["after"])
sample6_after = filter_confocal(sample6_confocal_data, "*after*")
sample7_before = filter_confocal(sample7_confocal_data, "*", exclude=["after"])
sample7_after = filter_confocal(sample7_confocal_data, "*after*")

# Load 200nm confocal data
sample13_confocal_data = load_with_cache(sample13_confocal_path, confocal_main)
sample13_pt_confocal_data = load_with_cache(sample13_pt_confocal_path, confocal_main)

# Combine normal and high power threshold confocal datasets
combined_13_confocal = (
    {**sample13_confocal_data[0], **sample13_pt_confocal_data[0]},  # image_dict
    {**sample13_confocal_data[1], **sample13_pt_confocal_data[1]},  # apd_dict  
    {**sample13_confocal_data[2], **sample13_pt_confocal_data[2]},  # monitor_dict
    {**sample13_confocal_data[3], **sample13_pt_confocal_data[3]},  # xy_dict
    {**sample13_confocal_data[4], **sample13_pt_confocal_data[4]}   # z_dict
)

# Separate before/after irradiation images for 200nm samples (exclude problematic C2)
sample13_box1_before = filter_confocal(combined_13_confocal, "*box1*", exclude=["after", "C2"])
sample13_box1_after = filter_confocal(combined_13_confocal, "*box1*after*", exclude=["C2"])
sample13_box4_before = filter_confocal(combined_13_confocal, "*box4*", exclude=["after", "C2"])
sample13_box4_after = filter_confocal(combined_13_confocal, "*box4*after*", exclude=["C2"])

# ==============================================================================
# ANALYZE CONFOCAL DATA
# ==============================================================================
print("Analyzing confocal data...")

# Analyze 100nm samples - extract SNR, PSF parameters, and intensity statistics
sample6_results_before = analyze_confocal(sample6_before)
sample6_results_after = analyze_confocal(sample6_after)
sample7_results_before = analyze_confocal(sample7_before)
sample7_results_after = analyze_confocal(sample7_after)

# Analyze 200nm samples - separate by box regions
sample13_box1_results_before = analyze_confocal(sample13_box1_before)
sample13_box1_results_after = analyze_confocal(sample13_box1_after)
sample13_box4_results_before = analyze_confocal(sample13_box4_before)
sample13_box4_results_after = analyze_confocal(sample13_box4_after)


import matplotlib.pyplot as plt

# Extract SNR values
snr_6_before = [results['snr_3x3'] for results in sample6_results_before.values() if 'snr_3x3' in results]
snr_6_after = [results['snr_3x3'] for results in sample6_results_after.values() if 'snr_3x3' in results]
snr_13_before = [results['snr_3x3'] for results in {**sample13_box1_results_before, **sample13_box4_results_before}.values() if 'snr_3x3' in results]
snr_13_after = [results['snr_3x3'] for results in {**sample13_box1_results_after, **sample13_box4_results_after}.values() if 'snr_3x3' in results]




import re

# Helper to match keys and extract power
def get_power_from_params(confocal_key, params_dict):
    # Extract pattern (e.g., 'A1') and box (e.g., 'box1') from confocal key
    pattern_match = re.search(r'[A-D][1-6]', confocal_key)
    box_match = re.search(r'box[14]', confocal_key)
    if pattern_match and box_match:
        pattern = pattern_match.group()
        box = box_match.group()
        for param_key in params_dict.keys():
            if pattern in param_key and box in param_key:
                return params_dict[param_key].get('power', None)
    return None

# Collect data for scatter plot
data_points = []

# Sample 6 Before
for key, results in sample6_results_before.items():
    power = get_power_from_params(key, sample6_params)
    if power is not None and 'snr_3x3' in results:
        data_points.append((power, results['snr_3x3'], key))  # Use actual key as label

# Sample 6 After
for key, results in sample6_results_after.items():
    power = get_power_from_params(key, sample6_params)
    if power is not None and 'snr_3x3' in results:
        data_points.append((power, results['snr_3x3'], key))  # Use actual key as label

# Sample 13 Before
combined_13_params_before = {**sample13_box1_params, **sample13_box4_params}
for key, results in {**sample13_box1_results_before, **sample13_box4_results_before}.items():
    power = get_power_from_params(key, combined_13_params)
    if power is not None and 'snr_3x3' in results:
        data_points.append((power, results['snr_3x3'], key))  # Use actual key as label

# Sample 13 After
combined_13_params_after = {**sample13_box1_params, **sample13_box4_params}
for key, results in {**sample13_box1_results_after, **sample13_box4_results_after}.items():
    power = get_power_from_params(key, combined_13_params)
    if power is not None and 'snr_3x3' in results:
        data_points.append((power, results['snr_3x3'], key))  # Use actual key as label

# Prepare for plotting
powers = [p[0] for p in data_points]
snrs = [p[1] for p in data_points]
labels = [p[2] for p in data_points]

# Add jitter to x-axis to separate overlapping points
import random
jittered_powers = [p + random.uniform(-0.5, 0.5) for p in powers]

# Create scatter plot with labels (using actual keys)
unique_labels = list(set(labels))
color_cycle = itertools.cycle(['blue', 'red', 'green', 'orange'])  # Cycle colors
for label in unique_labels:
    color = next(color_cycle)
    idx = [j for j, l in enumerate(labels) if l == label]
    plt.scatter([jittered_powers[j] for j in idx], [snrs[j] for j in idx], color=color, label=label)
    # Add labels to points
    for j in idx:
        plt.annotate(label, (jittered_powers[j], snrs[j]), fontsize=8, ha='center')

plt.title('SNR vs Power: Sample 6 and 13 (Before/After)')
plt.xlabel('Power (mW)')
plt.ylabel('SNR (3x3)')
plt.legend()
plt.show()