from apd_functions import * 
from spectra_functions import *
from ALL_plotting_functions_OLD import *
import matplotlib.pyplot as plt
import os
import pickle
from confocal_functions import *
import pandas as pd
import numpy as np
import re


def load_spectra_cached(spectra_path):
    """Load spectra data with caching to avoid long reload times"""
    # Create cache directory
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache filename from path
    cache_name = spectra_path.replace("/", "_").replace("\\", "_").replace(":", "").replace(" ", "_")
    cache_file = os.path.join(cache_dir, f"spectra_{cache_name}.pkl")
    
    # Try loading from cache first
    if os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            print("Cache corrupted, reloading...")
    
    # Load fresh data if no cache
    print(f"Loading spectra from: {spectra_path}")
    spectra_data, spectra_params = spectra_main(spectra_path)
    
    # Save to cache
    print(f"Saving to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump((spectra_data, spectra_params), f)
    
    return spectra_data, spectra_params

def match_sem_measurements(before_csv, after_csv):
    """
    Match SEM measurements before and after irradiation based on box and position labels
    
    Args:
        before_csv: Path to before irradiation CSV
        after_csv: Path to after irradiation CSV
    
    Returns:
        matched_data: Dictionary with matched measurements
    """
    # Read CSV files
    df_before = pd.read_csv(before_csv)
    df_after = pd.read_csv(after_csv)
    
    # Filter out tilted measurements (we only want non-tilted measurements)
    df_before = df_before[df_before['tilted_flag'] == False]
    df_after = df_after[df_after['tilted_flag'] == False]
    
    matched_data = {}
    
    # Extract box and position from labels
    def extract_box_position(label):
        """Extract box number and grid position from label"""
        # Before format: sample_13_box1_A1_filtered_smoothed.png
        # After format: sample_13_DNH_box1_C1_filtered_smoothed.png
        
        # Extract box number
        box_match = re.search(r'box(\d+)', label)
        if not box_match:
            return None, None
        box_num = box_match.group(1)
        
        # Extract position (A1, B2, C3, D4, etc.)
        # Look for pattern like A1, B2, C3, D4, D5, D6
        pos_match = re.search(r'[A-D][1-6]', label)
        if not pos_match:
            return None, None
        position = pos_match.group(0)
        
        return f"box{box_num}", position
    
    # Create mapping dictionaries
    before_data = {}
    after_data = {}
    
    # Process before measurements
    for _, row in df_before.iterrows():
        box, position = extract_box_position(row['label'])
        if box and position:
            key = f"{box}_{position}"
            before_data[key] = row.to_dict()
    
    # Process after measurements
    for _, row in df_after.iterrows():
        box, position = extract_box_position(row['label'])
        if box and position:
            key = f"{box}_{position}"
            after_data[key] = row.to_dict()
    
    # Match measurements that exist in both datasets
    for key in before_data.keys():
        if key in after_data:
            matched_data[key] = {
                'before': before_data[key],
                'after': after_data[key],
                'box': key.split('_')[0],
                'position': key.split('_')[1]
            }
    
    print(f"Found {len(matched_data)} matched SEM measurements")
    return matched_data

def plot_sem_before_after_comparison(matched_data, parameter, output_folder):
    """
    Plot before vs after comparison for a specific SEM parameter
    
    Args:
        matched_data: Dictionary with matched before/after measurements
        parameter: SEM parameter to plot (e.g., 'gap_width', 'interhole_distance')
        output_folder: Directory to save plots
    """
    # Separate by box
    box1_data = {k: v for k, v in matched_data.items() if v['box'] == 'box1'}
    box4_data = {k: v for k, v in matched_data.items() if v['box'] == 'box4'}
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Box1 plot
    if box1_data:
        before_values = [data['before'][parameter] for data in box1_data.values()]
        after_values = [data['after'][parameter] for data in box1_data.values()]
        positions = [data['position'] for data in box1_data.values()]
        
        x = np.arange(len(positions))
        width = 0.35
        
        ax1.bar(x - width/2, before_values, width, label='Before', color='dodgerblue', alpha=0.8)
        ax1.bar(x + width/2, after_values, width, label='After', color='crimson', alpha=0.8)
        
        ax1.set_xlabel('Position', fontsize=14, fontweight='bold')
        ax1.set_ylabel(f'{parameter.replace("_", " ").title()} (nm)', fontsize=14, fontweight='bold')
        ax1.set_title(f'Box1: {parameter.replace("_", " ").title()} Before vs After', fontsize=16, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(positions, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Box4 plot
    if box4_data:
        before_values = [data['before'][parameter] for data in box4_data.values()]
        after_values = [data['after'][parameter] for data in box4_data.values()]
        positions = [data['position'] for data in box4_data.values()]
        
        x = np.arange(len(positions))
        
        ax2.bar(x - width/2, before_values, width, label='Before', color='dodgerblue', alpha=0.8)
        ax2.bar(x + width/2, after_values, width, label='After', color='crimson', alpha=0.8)
        
        ax2.set_xlabel('Position', fontsize=14, fontweight='bold')
        ax2.set_ylabel(f'{parameter.replace("_", " ").title()} (nm)', fontsize=14, fontweight='bold')
        ax2.set_title(f'Box4: {parameter.replace("_", " ").title()} Before vs After', fontsize=16, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(positions, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"sem_{parameter}_before_after_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sem_before_after_comparison_with_power(matched_data, parameter, apd_params, output_folder):
    """
    Plot before vs after comparison with power information in labels
    
    Args:
        matched_data: Dictionary with matched before/after measurements  
        parameter: SEM parameter to plot
        apd_params: APD parameters containing power information
        output_folder: Directory to save plots
    """
    # Separate by box and get power info
    box1_data = {k: v for k, v in matched_data.items() if v['box'] == 'box1'}
    box4_data = {k: v for k, v in matched_data.items() if v['box'] == 'box4'}
    
    def get_power_for_position(position, apd_params, box):
        """Get power value for a given position and box"""
        for apd_key in apd_params.keys():
            if position in apd_key and box in apd_key and 'power' in apd_params[apd_key]:
                try:
                    return float(apd_params[apd_key]['power'])
                except (ValueError, TypeError):
                    continue
        return None
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Add main title
    fig.suptitle('200nm gold', fontsize=20, fontweight='bold')
    
    # Box1 plot (190 nm)
    if box1_data:
        # Get power info and sort by power
        position_power_data = []
        for data in box1_data.values():
            power = get_power_for_position(data['position'], apd_params, data['box'])
            if power:
                position_power_data.append((data['position'], power, data['before'][parameter], data['after'][parameter]))
        
        # Sort by power
        position_power_data.sort(key=lambda x: x[1])
        
        positions = [item[0] for item in position_power_data]
        powers = [item[1] for item in position_power_data]
        before_values = [item[2] for item in position_power_data]
        after_values = [item[3] for item in position_power_data]
        
        # Create labels with power info
        position_labels = [f'{pos}\n({power:.1f} mW)' for pos, power in zip(positions, powers)]
        
        x = np.arange(len(positions))
        width = 0.35
        
        ax1.bar(x - width/2, before_values, width, label='Before', color='dodgerblue', alpha=0.8)
        ax1.bar(x + width/2, after_values, width, label='After', color='crimson', alpha=0.8)
        
        ax1.set_xlabel('Structure & Power', fontsize=14, fontweight='bold')
        ax1.set_ylabel(f'{parameter.replace("_", " ").title()} (nm)', fontsize=14, fontweight='bold')
        ax1.set_title(f'190 nm: {parameter.replace("_", " ").title()} Before vs After', fontsize=16, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(position_labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Box4 plot (220 nm)
    if box4_data:
        # Get power info and sort by power
        position_power_data = []
        for data in box4_data.values():
            power = get_power_for_position(data['position'], apd_params, data['box'])
            if power:
                position_power_data.append((data['position'], power, data['before'][parameter], data['after'][parameter]))
        
        # Sort by power
        position_power_data.sort(key=lambda x: x[1])
        
        positions = [item[0] for item in position_power_data]
        powers = [item[1] for item in position_power_data]
        before_values = [item[2] for item in position_power_data]
        after_values = [item[3] for item in position_power_data]
        
        # Create labels with power info
        position_labels = [f'{pos}\n({power:.1f} mW)' for pos, power in zip(positions, powers)]
        
        x = np.arange(len(positions))
        
        ax2.bar(x - width/2, before_values, width, label='Before', color='dodgerblue', alpha=0.8)
        ax2.bar(x + width/2, after_values, width, label='After', color='crimson', alpha=0.8)
        
        ax2.set_xlabel('Position (Power)', fontsize=14, fontweight='bold')
        ax2.set_ylabel(f'{parameter.replace("_", " ").title()} (nm)', fontsize=14, fontweight='bold')
        ax2.set_title(f'220 nm: {parameter.replace("_", " ").title()} Before vs After', fontsize=16, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(position_labels, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"sem_{parameter}_before_after_comparison_with_power.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sem_scatter_before_after_with_power(matched_data, parameter, apd_params, output_folder):
    """
    Plot before vs after scatter plot with power information, sorted by power
    
    Args:
        matched_data: Dictionary with matched before/after measurements  
        parameter: SEM parameter to plot
        apd_params: APD parameters containing power information
        output_folder: Directory to save plots
    """
    # Separate by box and get power info
    box1_data = {k: v for k, v in matched_data.items() if v['box'] == 'box1'}
    box4_data = {k: v for k, v in matched_data.items() if v['box'] == 'box4'}
    
    def get_power_for_position(position, apd_params, box):
        """Get power value for a given position and box"""
        for apd_key in apd_params.keys():
            if position in apd_key and box in apd_key and 'power' in apd_params[apd_key]:
                try:
                    return float(apd_params[apd_key]['power'])
                except (ValueError, TypeError):
                    continue
        return None
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Add main title
    fig.suptitle('200nm gold', fontsize=20, fontweight='bold')
    
    # Box1 plot (190 nm)
    if box1_data:
        # Get power info and sort by power
        position_power_data = []
        for data in box1_data.values():
            power = get_power_for_position(data['position'], apd_params, data['box'])
            if power:
                position_power_data.append((data['position'], power, data['before'][parameter], data['after'][parameter]))
        
        # Sort by power
        position_power_data.sort(key=lambda x: x[1])
        
        positions = [item[0] for item in position_power_data]
        powers = [item[1] for item in position_power_data]
        before_values = [item[2] for item in position_power_data]
        after_values = [item[3] for item in position_power_data]
        
        # Use power as x-axis positions
        x_positions = powers
        
        # Plot before data (blue circles)
        ax1.scatter(x_positions, before_values, s=120, color='dodgerblue', alpha=0.8, 
                   marker='o', edgecolors='black', linewidths=1.5, label='Before')
        
        # Plot after data (red squares)
        ax1.scatter(x_positions, after_values, s=120, color='crimson', alpha=0.8, 
                   marker='s', edgecolors='black', linewidths=1.5, label='After')
        
        # Add text labels for positions near each point
        # for i, (power, before_val, after_val, pos) in enumerate(zip(powers, before_values, after_values, positions)):
        #     # Alternate labels slightly above/below to avoid overlap
        #     offset = 1.5 if i % 2 == 0 else -1.5
        #     ax1.text(power, max(before_val, after_val) + offset, pos, 
        #             fontsize=9, ha='center', va='bottom' if i % 2 == 0 else 'top')
        
        ax1.set_xlabel('Power (mW)', fontsize=14, fontweight='bold')
        ax1.set_ylabel(f'{parameter.replace("_", " ").title()} (nm)', fontsize=14, fontweight='bold')
        ax1.set_title(f'190 nm: {parameter.replace("_", " ").title()} Before vs After', fontsize=16, fontweight='bold')
        ax1.set_ylim(15, 40)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Box4 plot (220 nm)
    if box4_data:
        # Get power info and sort by power
        position_power_data = []
        for data in box4_data.values():
            power = get_power_for_position(data['position'], apd_params, data['box'])
            if power:
                position_power_data.append((data['position'], power, data['before'][parameter], data['after'][parameter]))
        
        # Sort by power
        position_power_data.sort(key=lambda x: x[1])
        
        positions = [item[0] for item in position_power_data]
        powers = [item[1] for item in position_power_data]
        before_values = [item[2] for item in position_power_data]
        after_values = [item[3] for item in position_power_data]
        
        # Use power as x-axis positions
        x_positions = powers
        
        # Plot before data (blue circles)
        ax2.scatter(x_positions, before_values, s=120, color='dodgerblue', alpha=0.8, 
                   marker='o', edgecolors='black', linewidths=1.5, label='Before')
        
        # Plot after data (red squares)
        ax2.scatter(x_positions, after_values, s=120, color='crimson', alpha=0.8, 
                   marker='s', edgecolors='black', linewidths=1.5, label='After')
        
        # Add text labels for positions near each point
        # for i, (power, before_val, after_val, pos) in enumerate(zip(powers, before_values, after_values, positions)):
        #     # Alternate labels slightly above/below to avoid overlap
        #     offset = 1.5 if i % 2 == 0 else -1.5
        #     ax2.text(power, max(before_val, after_val) + offset, pos, 
        #             fontsize=9, ha='center', va='bottom' if i % 2 == 0 else 'top')
        
        ax2.set_xlabel('Power (mW)', fontsize=14, fontweight='bold')
        ax2.set_ylabel(f'{parameter.replace("_", " ").title()} (nm)', fontsize=14, fontweight='bold')
        ax2.set_title(f'220 nm: {parameter.replace("_", " ").title()} Before vs After', fontsize=16, fontweight='bold')
        ax2.set_ylim(15, 40)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"sem_{parameter}_scatter_before_after_with_power.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_snr_before_after_with_power(confocal_before, confocal_after, results_before, results_after, apd_params, output_folder):
    """
    Plot SNR before vs after with power information, sorted by power
    
    Args:
        confocal_before: Confocal data before irradiation
        confocal_after: Confocal data after irradiation
        results_before: Analysis results before irradiation
        results_after: Analysis results after irradiation
        apd_params: APD parameters containing power information
        output_folder: Directory to save plots
    """
    # Extract image_dicts from tuples
    confocal_dict_before = confocal_before[0]
    confocal_dict_after = confocal_after[0]
    
    def get_power_for_position(position, apd_params, box):
        """Get power value for a given position and box"""
        for apd_key in apd_params.keys():
            if position in apd_key and box in apd_key and 'power' in apd_params[apd_key]:
                try:
                    return float(apd_params[apd_key]['power'])
                except (ValueError, TypeError):
                    continue
        return None
    
    # Find common [A-D][1-6] patterns and separate by box
    pattern = r'[A-D][1-6]'
    
    # Box1 data
    box1_before = {}
    box1_after = {}
    for key in confocal_dict_before.keys():
        match = re.search(pattern, key)
        if match and 'box1' in key:
            position = match.group()
            box1_before[position] = {
                'snr': results_before[key]['snr_3x3'] if key in results_before and 'snr_3x3' in results_before[key] else None
            }
    
    for key in confocal_dict_after.keys():
        match = re.search(pattern, key)
        if match and 'box1' in key:
            position = match.group()
            box1_after[position] = {
                'snr': results_after[key]['snr_3x3'] if key in results_after and 'snr_3x3' in results_after[key] else None
            }
    
    # Box4 data
    box4_before = {}
    box4_after = {}
    for key in confocal_dict_before.keys():
        match = re.search(pattern, key)
        if match and 'box4' in key:
            position = match.group()
            box4_before[position] = {
                'snr': results_before[key]['snr_3x3'] if key in results_before and 'snr_3x3' in results_before[key] else None
            }
    
    for key in confocal_dict_after.keys():
        match = re.search(pattern, key)
        if match and 'box4' in key:
            position = match.group()
            box4_after[position] = {
                'snr': results_after[key]['snr_3x3'] if key in results_after and 'snr_3x3' in results_after[key] else None
            }
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Add main title
    fig.suptitle('200nm gold', fontsize=20, fontweight='bold')
    
    # Box1 plot (190 nm)
    if box1_before and box1_after:
        # Get power info and sort by power
        position_power_data = []
        for position in box1_before.keys():
            if position in box1_after and box1_before[position]['snr'] is not None and box1_after[position]['snr'] is not None:
                power = get_power_for_position(position, apd_params, 'box1')
                if power:
                    position_power_data.append((position, power, box1_before[position]['snr'], box1_after[position]['snr']))
        
        # Sort by power
        position_power_data.sort(key=lambda x: x[1])
        
        positions = [item[0] for item in position_power_data]
        powers = [item[1] for item in position_power_data]
        before_values = [item[2] for item in position_power_data]
        after_values = [item[3] for item in position_power_data]
        
        # Create x-axis positions (0, 1, 2, ...)
        x_positions = np.arange(len(positions))
        
        # Plot before data (blue circles)
        ax1.scatter(x_positions, before_values, s=120, color='dodgerblue', alpha=0.8, 
                   marker='o', edgecolors='black', linewidths=1.5, label='Before')
        
        # Plot after data (red squares)
        ax1.scatter(x_positions, after_values, s=120, color='crimson', alpha=0.8, 
                   marker='s', edgecolors='black', linewidths=1.5, label='After')
        
        # Create labels with power info
        position_labels = [f'{pos}\n{power:.1f} mW' for pos, power in zip(positions, powers)]
        
        ax1.set_xlabel('Structure & Power', fontsize=14, fontweight='bold')
        ax1.set_ylabel('SNR (3x3 center)', fontsize=14, fontweight='bold')
        ax1.set_title(f'190 nm: SNR Before vs After', fontsize=16, fontweight='bold')
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(position_labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Box4 plot (220 nm)
    if box4_before and box4_after:
        # Get power info and sort by power
        position_power_data = []
        for position in box4_before.keys():
            if position in box4_after and box4_before[position]['snr'] is not None and box4_after[position]['snr'] is not None:
                power = get_power_for_position(position, apd_params, 'box4')
                if power:
                    position_power_data.append((position, power, box4_before[position]['snr'], box4_after[position]['snr']))
        
        # Sort by power
        position_power_data.sort(key=lambda x: x[1])
        
        positions = [item[0] for item in position_power_data]
        powers = [item[1] for item in position_power_data]
        before_values = [item[2] for item in position_power_data]
        after_values = [item[3] for item in position_power_data]
        
        # Create x-axis positions (0, 1, 2, ...)
        x_positions = np.arange(len(positions))
        
        # Plot before data (blue circles)
        ax2.scatter(x_positions, before_values, s=120, color='dodgerblue', alpha=0.8, 
                   marker='o', edgecolors='black', linewidths=1.5, label='Before')
        
        # Plot after data (red squares)
        ax2.scatter(x_positions, after_values, s=120, color='crimson', alpha=0.8, 
                   marker='s', edgecolors='black', linewidths=1.5, label='After')
        
        # Create labels with power info
        position_labels = [f'{pos}\n{power:.1f} mW' for pos, power in zip(positions, powers)]
        
        ax2.set_xlabel('Position (Power)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('SNR (3x3 center)', fontsize=14, fontweight='bold')
        ax2.set_title(f'220 nm: SNR Before vs After', fontsize=16, fontweight='bold')
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(position_labels, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "snr_scatter_before_after_with_power.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sem_scatter_before_after(matched_data, parameter, output_folder):
    """
    Create scatter plot showing before vs after values for a SEM parameter
    
    Args:
        matched_data: Dictionary with matched before/after measurements
        parameter: SEM parameter to plot
        output_folder: Directory to save plots
    """
    plt.figure(figsize=(10, 8))
    
    # Separate by box
    box1_data = {k: v for k, v in matched_data.items() if v['box'] == 'box1'}
    box4_data = {k: v for k, v in matched_data.items() if v['box'] == 'box4'}
    
    # Plot Box1 data
    if box1_data:
        before_vals = [data['before'][parameter] for data in box1_data.values()]
        after_vals = [data['after'][parameter] for data in box1_data.values()]
        plt.scatter(before_vals, after_vals, s=100, alpha=0.8, color='dodgerblue', 
                   label=f'Box1 (n={len(box1_data)})', edgecolors='black', linewidths=1)
    
    # Plot Box4 data
    if box4_data:
        before_vals = [data['before'][parameter] for data in box4_data.values()]
        after_vals = [data['after'][parameter] for data in box4_data.values()]
        plt.scatter(before_vals, after_vals, s=100, alpha=0.8, color='crimson', 
                   label=f'Box4 (n={len(box4_data)})', edgecolors='black', linewidths=1)
    
    # Add diagonal line (no change line)
    all_before = [data['before'][parameter] for data in matched_data.values()]
    all_after = [data['after'][parameter] for data in matched_data.values()]
    min_val = min(min(all_before), min(all_after))
    max_val = max(max(all_before), max(all_after))
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='No Change')
    
    plt.xlabel(f'{parameter.replace("_", " ").title()} Before (nm)', fontsize=14, fontweight='bold')
    plt.ylabel(f'{parameter.replace("_", " ").title()} After (nm)', fontsize=14, fontweight='bold')
    plt.title(f'{parameter.replace("_", " ").title()}: Before vs After Irradiation', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_folder, f"sem_{parameter}_scatter_before_after.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sem_power_scatter_before_after(matched_data, parameter, apd_params, output_folder):
    """
    Create scatter plot showing SEM parameter vs power, with before/after comparison
    
    Args:
        matched_data: Dictionary with matched before/after measurements
        parameter: SEM parameter to plot
        apd_params: APD parameters containing power information
        output_folder: Directory to save plots
    """
    plt.figure(figsize=(14, 8))
    
    # Extract power values for each position
    powers_before, values_before, powers_after, values_after = [], [], [], []
    boxes_before, boxes_after = [], []
    position_labels = []
    
    for key, data in matched_data.items():
        position = data['position']
        
        # Find matching APD parameter key for this position
        power = None
        for apd_key in apd_params.keys():
            if position in apd_key and 'power' in apd_params[apd_key]:
                try:
                    power = float(apd_params[apd_key]['power'])
                    break
                except (ValueError, TypeError):
                    continue
        
        if power and power > 0.01:  # Valid power found
            powers_before.append(power)
            values_before.append(data['before'][parameter])
            boxes_before.append(data['box'])
            
            powers_after.append(power)
            values_after.append(data['after'][parameter])
            boxes_after.append(data['box'])
            position_labels.append(f"{data['box']}_{position}")
    
    # Plot before data
    for box in ['box1', 'box4']:
        box_powers = [p for p, b in zip(powers_before, boxes_before) if b == box]
        box_values = [v for v, b in zip(values_before, boxes_before) if b == box]
        box_labels = [l for l, b in zip(position_labels, boxes_before) if b == box]
        
        if box_powers:
            color = 'dodgerblue' if box == 'box1' else 'crimson'
            marker = 'o' if box == 'box1' else 's'
            scatter = plt.scatter(box_powers, box_values, s=120, alpha=0.7, color=color,
                       marker=marker, edgecolors='black', linewidths=1.5,
                       label=f'{box.title()} Before')
            
            # Add position labels
            for i, (x, y, label) in enumerate(zip(box_powers, box_values, box_labels)):
                plt.annotate(label.split('_')[1], (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, alpha=0.8)
    
    # Plot after data with different styling
    for box in ['box1', 'box4']:
        box_powers = [p for p, b in zip(powers_after, boxes_after) if b == box]
        box_values = [v for v, b in zip(values_after, boxes_after) if b == box]
        box_labels = [l for l, b in zip(position_labels, boxes_after) if b == box]
        
        if box_powers:
            color = 'navy' if box == 'box1' else 'darkred'
            marker = 'o' if box == 'box1' else 's'
            plt.scatter(box_powers, box_values, s=120, alpha=0.9, color=color,
                       marker=marker, edgecolors='white', linewidths=2,
                       label=f'{box.title()} After')
    
    plt.xlabel('Power (mW)', fontsize=14, fontweight='bold')
    plt.ylabel(f'{parameter.replace("_", " ").title()} (nm)', fontsize=14, fontweight='bold')
    plt.title(f'{parameter.replace("_", " ").title()} vs Power: Before vs After Irradiation', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_folder, f"sem_{parameter}_power_scatter_before_after.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sem_change_distribution(matched_data, parameter, output_folder):
    """
    Plot distribution of changes in SEM parameter
    
    Args:
        matched_data: Dictionary with matched before/after measurements
        parameter: SEM parameter to analyze
        output_folder: Directory to save plots
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate changes
    box1_changes = []
    box4_changes = []
    
    for key, data in matched_data.items():
        change = data['after'][parameter] - data['before'][parameter]
        if data['box'] == 'box1':
            box1_changes.append(change)
        else:
            box4_changes.append(change)
    
    # Create histogram
    bins = np.linspace(min(box1_changes + box4_changes), max(box1_changes + box4_changes), 15)
    
    plt.hist(box1_changes, bins=bins, alpha=0.7, label=f'Box1 (μ={np.mean(box1_changes):.1f} nm)', 
             color='dodgerblue', edgecolor='black')
    plt.hist(box4_changes, bins=bins, alpha=0.7, label=f'Box4 (μ={np.mean(box4_changes):.1f} nm)', 
             color='crimson', edgecolor='black')
    
    # Add vertical line at zero change
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='No Change')
    
    plt.xlabel(f'Change in {parameter.replace("_", " ").title()} (nm)', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=14, fontweight='bold')
    plt.title(f'Distribution of Changes in {parameter.replace("_", " ").title()}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_folder, f"sem_{parameter}_change_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sem_violin_comparison(matched_data, parameter, output_folder):
    """
    Create violin plot comparing SEM parameter distributions before and after irradiation
    
    Args:
        matched_data: Dictionary with matched before/after measurements
        parameter: SEM parameter to plot (e.g., 'gap_width', 'radius_top')
        output_folder: Directory to save plots
    """
    # Extract data for Box1 and Box4, before and after
    box1_before = []
    box1_after = []
    box4_before = []
    box4_after = []
    
    for key, data in matched_data.items():
        if data['box'] == 'box1':
            box1_before.append(data['before'][parameter])
            box1_after.append(data['after'][parameter])
        else:  # box4
            box4_before.append(data['before'][parameter])
            box4_after.append(data['after'][parameter])
    
    # Prepare data for violin plot
    data = [box1_before, box1_after, box4_before, box4_after]
    
    # Multi-line labels with interhole distances
    labels = ['190 nm\nBefore', 
              '190 nm\nAfter', 
              '220 nm\nBefore', 
              '220 nm\nAfter']
    
    # Create violin plot with styling matching existing plots
    plt.figure(figsize=(16, 8))
    violin_parts = plt.violinplot(data, positions=range(len(labels)), showmeans=True, showmedians=True)
    
    # Color scheme: blue for before, red for after
    colors = ['dodgerblue', 'crimson', 'dodgerblue', 'crimson']
    for i, (pc, color) in enumerate(zip(violin_parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.8)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Style the violin plot elements
    for element in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
        if element in violin_parts:
            violin_parts[element].set_color('black')
            violin_parts[element].set_linewidth(2)
    
    # Add scatter points for individual measurements
    for i, (vals, color) in enumerate(zip(data, colors)):
        if len(vals) > 0:
            # Add jitter for visibility
            x_jitter = np.random.normal(i, 0.05, len(vals))
            plt.scatter(x_jitter, vals, alpha=0.8, s=80, color=color, 
                       edgecolors='black', linewidths=1.5)
    
    # Customize plot
    plt.xticks(range(len(labels)), labels, fontsize=18, fontweight='bold')
    plt.ylabel(f'{parameter.replace("_", " ").title()} (nm)', fontsize=18, fontweight='bold')
    plt.title(f'{parameter.replace("_", " ").title()} Comparison: Before vs After Irradiation', 
              fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    # Determine y-axis range for consistent text placement
    all_vals = [val for sublist in data for val in sublist]
    y_min, y_max = min(all_vals), max(all_vals)
    y_range = y_max - y_min
    
    # Set y-axis limits with padding for text boxes (15% extra at top)
    plt.ylim(y_min - (y_range * 0.05), y_max + (y_range * 0.15))
    
    # Add statistics text at consistent height
    text_y_position = y_max + (y_range * 0.05)
    
    for i, (vals, label) in enumerate(zip(data, labels)):
        if len(vals) > 0:
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            # Position text at consistent height above all data
            plt.text(i, text_y_position, f'μ={mean_val:.1f}\nσ={std_val:.1f}', 
                    ha='center', va='bottom', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"sem_{parameter}_violin_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Output directory setup
    output_folder = "plots/Au_200nm"
    os.makedirs(output_folder, exist_ok=True)

    # Data paths
    apd_path = "./Data/APD/2025.08.21 - Sample 13 Power Threshold"
    pt_spectra_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250902 - sample13 PT"
    pt_apd_path = r"\\AMIPC045962\Cache (D)\daily_data\apd_traces\2025.09.02 - Sample 13 PT high power"
    after_spectra_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250821 - sample13 - after"
    before_spectra_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250812 - sample13"
    confocal_path = r"Data\Confocal\2025.08.21 - Sample 13 Power Threshold box1"
    pt_confocal_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.09.02 - Sample 13 before after break"

    # Load and filter APD data from both normal and PT datasets
    apd_data, monitor_data, apd_params = apd_load_main(apd_path)
    pt_apd_data, pt_monitor_data, pt_apd_params = apd_load_main(pt_apd_path)
    
    # Combine normal and PT APD data
    combined_apd = {**apd_data, **pt_apd_data}
    combined_monitor = {**monitor_data, **pt_monitor_data}
    combined_params = {**apd_params, **pt_apd_params}
    
    # Filter combined datasets for Box1 and Box4
    box1_apd, box1_monitor, box1_params = filter_apd(combined_apd, combined_monitor, combined_params, "*box1*")
    box4_apd, box4_monitor, box4_params = filter_apd(combined_apd, combined_monitor, combined_params, "*box4*[!_D4_*]*")
    
    # Load PT spectra data
    pt_spectra, pt_spectra_params = load_spectra_cached(pt_spectra_path)
    
    # Plot combined APD traces for Box1 (includes both normal and PT data)
    plot_apd(box1_apd, box1_monitor, box1_params, new_fig=True)
    plt.savefig(os.path.join(output_folder, "apd_box1_combined.png"))
    plt.close()

    plot_apd(box1_apd, box1_monitor, box1_params, new_fig=True, time=30)
    plt.savefig(os.path.join(output_folder, "apd_box1_combined_30s.png"))
    plt.close()

    # Plot combined APD traces for Box4 (includes both normal and PT data)
    plot_apd(box4_apd, box4_monitor, box4_params, new_fig=True)
    plt.savefig(os.path.join(output_folder, "apd_box4_combined.png"))
    plt.close()

    plot_apd(box4_apd, box4_monitor, box4_params, new_fig=True, time=30)
    plt.savefig(os.path.join(output_folder, "apd_box4_combined_30s.png"))
    plt.close()

    # Load and filter spectra data
    spectra_after, spectra_after_params = load_spectra_cached(after_spectra_path)
    spectra_before, spectra_before_params = load_spectra_cached(before_spectra_path)
    
    # Filter spectra data
    box1_after_spec, _ = filter_spectra(spectra_after, spectra_after_params, "*box1*", average=True)
    box4_before_spec, _ = filter_spectra(spectra_before, spectra_before_params, "*box4*", average=True)
    box4_after_spec, _ = filter_spectra(spectra_after, spectra_after_params, "*box4*", average=True)
    box1_before_spec, _ = filter_spectra(spectra_before, spectra_before_params, "*box1*", average=True)
    ref_5um_before, _ = filter_spectra(spectra_before, spectra_before_params, "*5um*", average=True)
    bkg_before, _ = filter_spectra(spectra_before, spectra_before_params, "*bkg*", average=True)
    ref_5um_after, _ = filter_spectra(spectra_after, spectra_after_params, "*5um*_z_locked*")
    bkg_after, _ = filter_spectra(spectra_after, spectra_after_params, "*bkg*")
    bkg_10000ms, bkg_10000ms_params = filter_spectra(spectra_after, spectra_after_params, "*bkg*_10000ms*")

    # Plot Savgol filter analysis
    plot_spectra_savgol(bkg_10000ms, bkg_10000ms_params, window_sizes=[21], orders=[1, 2, 3, 4, 5])
    plt.savefig(os.path.join(output_folder, "spectra_bkg_10000ms_savgol.png"))
    plt.close()

    # Normalize spectra with specific backgrounds and references
    norm_box1_after = normalize_spectra_bkg(box1_after_spec, bkg_after['bkg_10000ms_1'], ref_5um_after['5um_100ms_z_locked_1'], bkg_after['bkg_100ms_3'], 
                                        savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    norm_box4_after = normalize_spectra_bkg(box4_after_spec, bkg_after['bkg_10000ms_1'], ref_5um_after['5um_100ms_z_locked_1'], bkg_after['bkg_100ms_3'], 
                                        savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    norm_box1_before = normalize_spectra_bkg(box1_before_spec, bkg_before['bkg_single_track'], ref_5um_before['5um_ref_single_track_50ms'], bkg_before['bkg_single_track_50ms'], 
                                        savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)
    norm_box4_before = normalize_spectra_bkg(box4_before_spec, bkg_before['bkg_single_track'], ref_5um_before['5um_ref_single_track_50ms'], bkg_before['bkg_single_track_50ms'], 
                                        savgol_before_bkg=False, savgol_after_div=True, savgol_after_div_window=131, savgol_after_div_order=1)

    # Plot raw spectra
    plot_spectra(box1_after_spec, spectra_after_params)
    plt.savefig(os.path.join(output_folder, "spectra_box1_after.png"))
    plt.close()

    plot_spectra(box4_after_spec, spectra_after_params)
    plt.savefig(os.path.join(output_folder, "spectra_box4_after.png"))
    plt.close()

    plot_spectra(box1_before_spec, spectra_before_params)
    plt.savefig(os.path.join(output_folder, "spectra_box1_before.png"))
    plt.close()

    plot_spectra(box4_before_spec, spectra_before_params)
    plt.savefig(os.path.join(output_folder, "spectra_box4_before.png"))
    plt.close()

    plot_spectra(bkg_10000ms, bkg_10000ms_params)
    plt.savefig(os.path.join(output_folder, "spectra_bkg_10000ms.png"))
    plt.close()

    plot_spectra(ref_5um_after, spectra_after_params)
    plt.savefig(os.path.join(output_folder, "spectra_ref_5um.png"))
    plt.close()

    # Plot normalized spectra
    plot_spectra(norm_box1_before, spectra_before_params)
    plt.savefig(os.path.join(output_folder, "normalized_spectra_before.png"))
    plt.close()

    plot_spectra(norm_box1_after, spectra_after_params)
    plt.savefig(os.path.join(output_folder, "normalized_spectra_after.png"))
    plt.close()

    plot_spectra(norm_box4_after, spectra_after_params)
    plt.savefig(os.path.join(output_folder, "normalized_spectra_box4_after.png"))
    plt.close()

    plot_spectra(norm_box4_before, spectra_before_params)
    plt.savefig(os.path.join(output_folder, "normalized_spectra_box4_before.png"))
    plt.close()

    # Plot before/after comparison
    plot_spectra(norm_box1_after, spectra_after_params)
    plot_spectra(norm_box1_before, spectra_before_params, new_fig=False, linestyle='--')
    plt.savefig(os.path.join(output_folder, "normalized_spectra_comparison.png"))
    plt.close()

    # Plot Box4 before/after comparison
    plot_spectra(norm_box4_after, spectra_after_params)
    plot_spectra(norm_box4_before, spectra_before_params, new_fig=False, linestyle='--')
    plt.savefig(os.path.join(output_folder, "normalized_spectra_box4_comparison.png"))
    plt.close()

    # Create averaged comparison
    first_key = list(norm_box1_after.keys())[0]
    wavelength = norm_box1_after[first_key][:, 0]
    intensities_after = [norm_box1_after[key][:, 1] for key in norm_box1_after.keys()]
    averaged_intensity_after = np.mean(intensities_after, axis=0)
    averaged_normalized = np.column_stack([wavelength, averaged_intensity_after])
    
    first_key_before = list(norm_box1_before.keys())[0]
    wavelength_before = norm_box1_before[first_key_before][:, 0]
    intensities_before = [norm_box1_before[key][:, 1] for key in norm_box1_before.keys()]
    averaged_intensity_before = np.mean(intensities_before, axis=0)
    averaged_normalized_before = np.column_stack([wavelength_before, averaged_intensity_before])
    
    plot_spectra({'After': averaged_normalized}, {'After': spectra_after_params[list(spectra_after_params.keys())[0]]}, linestyle='-')
    plot_spectra({'Before': averaged_normalized_before}, {'Before': spectra_before_params[list(spectra_before_params.keys())[0]]}, new_fig=False, linestyle='--')
    plt.savefig(os.path.join(output_folder, "averaged_normalized_spectra_comparison.png"))
    plt.close()

    # Create Box4 averaged comparison
    first_key_box4 = list(norm_box4_after.keys())[0]
    wavelength_box4 = norm_box4_after[first_key_box4][:, 0]
    intensities_after_box4 = [norm_box4_after[key][:, 1] for key in norm_box4_after.keys()]
    averaged_intensity_after_box4 = np.mean(intensities_after_box4, axis=0)
    averaged_normalized_box4 = np.column_stack([wavelength_box4, averaged_intensity_after_box4])
    
    first_key_before_box4 = list(norm_box4_before.keys())[0]
    wavelength_before_box4 = norm_box4_before[first_key_before_box4][:, 0]
    intensities_before_box4 = [norm_box4_before[key][:, 1] for key in norm_box4_before.keys()]
    averaged_intensity_before_box4 = np.mean(intensities_before_box4, axis=0)
    averaged_normalized_before_box4 = np.column_stack([wavelength_before_box4, averaged_intensity_before_box4])
    
    plot_spectra({'After': averaged_normalized_box4}, {'After': spectra_after_params[list(spectra_after_params.keys())[0]]}, linestyle='-')
    plot_spectra({'Before': averaged_normalized_before_box4}, {'Before': spectra_before_params[list(spectra_before_params.keys())[0]]}, new_fig=False, linestyle='--')
    plt.savefig(os.path.join(output_folder, "averaged_normalized_spectra_box4_comparison.png"))
    plt.close()

    # Load and combine confocal data from both datasets
    confocal_data = load_with_cache(confocal_path, confocal_main)
    pt_confocal_data = load_with_cache(pt_confocal_path, confocal_main)
    
    # Combine confocal datasets (merge all dictionaries in the tuple)
    combined_confocal = (
        {**confocal_data[0], **pt_confocal_data[0]},  # image_dict
        {**confocal_data[1], **pt_confocal_data[1]},  # apd_dict  
        {**confocal_data[2], **pt_confocal_data[2]},  # monitor_dict
        {**confocal_data[3], **pt_confocal_data[3]},  # xy_dict
        {**confocal_data[4], **pt_confocal_data[4]}   # z_dict
    )
    
    # Filter combined confocal images
    box1_confocal_before = filter_confocal(combined_confocal, "*box1*", exclude=["after", "C2"])
    box1_confocal_after = filter_confocal(combined_confocal, "*box1*after*", exclude=["C2"])
    box4_confocal_before = filter_confocal(combined_confocal, "*box4*", exclude=["after", "C2"])
    box4_confocal_after = filter_confocal(combined_confocal, "*box4*after*", exclude=["C2"])
    
    print(f"Combined Box1 confocal before: {list(box1_confocal_before[0].keys())}")
    print(f"Combined Box1 confocal after: {list(box1_confocal_after[0].keys())}")
    print(f"Combined Box4 confocal before: {list(box4_confocal_before[0].keys())}")
    print(f"Combined Box4 confocal after: {list(box4_confocal_after[0].keys())}")

    # Analyze combined confocal data (includes APD trace statistics and max values)
    box1_results_before = analyze_confocal(box1_confocal_before)
    box1_results_after = analyze_confocal(box1_confocal_after)
    box4_results_before = analyze_confocal(box4_confocal_before)
    box4_results_after = analyze_confocal(box4_confocal_after)

    # Plot confocal image comparisons
    plot_confocal_image_comparison(box1_confocal_before, box1_confocal_after)
    plt.savefig(os.path.join(output_folder, "confocal_image_comparison_box1.png"))
    plt.close()

    plot_confocal_comparison(box1_confocal_before, box1_confocal_after)
    plt.savefig(os.path.join(output_folder, "confocal_comparison_box1.png"))
    plt.close()

    # Extract popt for plotting functions that expect the old format
    popt_box1_before = {k: v['popt'] for k, v in box1_results_before.items()}
    popt_box1_after = {k: v['popt'] for k, v in box1_results_after.items()}

    plot_confocal_comparison(box1_confocal_before, box1_confocal_after, popt_box1_before, popt_box1_after)
    plt.savefig(os.path.join(output_folder, "confocal_comparison_with_fits_box1.png"))
    plt.close()

    # Plot confocal SNR analysis - combined datasets (normal + PT)
    plot_confocal_snr(box1_confocal_before, box1_results_before, box1_params, label="Box1 Before")
    plot_confocal_snr(box1_confocal_after, box1_results_after, box1_params, new_fig=False, marker='s', label="Box1 After")
    plt.savefig(os.path.join(output_folder, "confocal_snr_box1_combined.png"))
    plt.close()

    plot_confocal_snr(box4_confocal_before, box4_results_before, box4_params, label="Box4 Before")
    plot_confocal_snr(box4_confocal_after, box4_results_after, box4_params, new_fig=False, marker='s', label="Box4 After")
    plt.savefig(os.path.join(output_folder, "confocal_snr_box4_combined.png"))
    plt.close()

    # Plot confocal max value scatter analysis - combined datasets
    plot_confocal_scatters(box1_confocal_before, box1_confocal_after, box1_results_before, box1_results_after, box1_params, label="Box1")
    plot_confocal_scatters(box4_confocal_before, box4_confocal_after, box4_results_before, box4_results_after, box4_params, new_fig=False, marker='s', label="Box4")
    plt.savefig(os.path.join(output_folder, "confocal_max_value_scatter_combined.png"))
    plt.close()

    # Plot simple SNR before vs after comparison - separate plots
    plot_snr_before_after(box1_confocal_before, box1_confocal_after, box1_results_before, box1_results_after, label="Box1")
    plt.savefig(os.path.join(output_folder, "snr_before_vs_after_box1.png"))
    plt.close()

    plot_snr_before_after(box4_confocal_before, box4_confocal_after, box4_results_before, box4_results_after, label="Box4", marker='s')
    plt.savefig(os.path.join(output_folder, "snr_before_vs_after_box4.png"))
    plt.close()

    # Plot SNR vs power for before and after measurements
    plot_snr_vs_power(box1_confocal_before, box1_results_before, box1_params, label="Box1 Before")
    plot_snr_vs_power(box1_confocal_after, box1_results_after, box1_params, new_fig=False, marker='s', label="Box1 After")
    plt.savefig(os.path.join(output_folder, "snr_vs_power_box1.png"))
    plt.close()

    plot_snr_vs_power(box4_confocal_before, box4_results_before, box4_params, label="Box4 Before")
    plot_snr_vs_power(box4_confocal_after, box4_results_after, box4_params, new_fig=False, marker='s', label="Box4 After")
    plt.savefig(os.path.join(output_folder, "snr_vs_power_box4.png"))
    plt.close()

    # NEW: Plot SNR before/after comparison with power (same format as gap width plots)
    # Combine all confocal data and results for plotting
    combined_confocal_before = (
        {**box1_confocal_before[0], **box4_confocal_before[0]},  # image_dict
        {**box1_confocal_before[1], **box4_confocal_before[1]},  # apd_dict  
        {**box1_confocal_before[2], **box4_confocal_before[2]},  # monitor_dict
        {**box1_confocal_before[3], **box4_confocal_before[3]},  # xy_dict
        {**box1_confocal_before[4], **box4_confocal_before[4]}   # z_dict
    )
    
    combined_confocal_after = (
        {**box1_confocal_after[0], **box4_confocal_after[0]},  # image_dict
        {**box1_confocal_after[1], **box4_confocal_after[1]},  # apd_dict  
        {**box1_confocal_after[2], **box4_confocal_after[2]},  # monitor_dict
        {**box1_confocal_after[3], **box4_confocal_after[3]},  # xy_dict
        {**box1_confocal_after[4], **box4_confocal_after[4]}   # z_dict
    )
    
    combined_results_before = {**box1_results_before, **box4_results_before}
    combined_results_after = {**box1_results_after, **box4_results_after}
    combined_params = {**box1_params, **box4_params}
    
    plot_snr_before_after_with_power(combined_confocal_before, combined_confocal_after, combined_results_before, combined_results_after, combined_params, output_folder)

    # Calculate total PSF size and add to results (convert pixels to nm: 2000nm/20px = 100nm/px)
    for results in [box1_results_before, box1_results_after, box4_results_before, box4_results_after]:
        for key in results:
            results[key]['psf_total_size'] = (results[key]['sigma_x'] * results[key]['sigma_y']) ** 0.5 * 100

    # Plot PSF total size vs power
    plot_confocal_parameter_scatter(box1_confocal_before, box1_results_before, box1_params, parameter='psf_total_size', label="Box1 Before")
    plot_confocal_parameter_scatter(box1_confocal_after, box1_results_after, box1_params, parameter='psf_total_size', new_fig=False, marker='s', label="Box1 After")
    plt.savefig(os.path.join(output_folder, "psf_total_size_vs_power_box1.png"))
    plt.close()

    plot_confocal_parameter_scatter(box4_confocal_before, box4_results_before, box4_params, parameter='psf_total_size', label="Box4 Before")
    plot_confocal_parameter_scatter(box4_confocal_after, box4_results_after, box4_params, parameter='psf_total_size', new_fig=False, marker='s', label="Box4 After")
    plt.savefig(os.path.join(output_folder, "psf_total_size_vs_power_box4.png"))
    plt.close()

    # SEM analysis plots - separate by box, exclude C1/C2 from box1
    sem_csv_path = r"Data\SEM\SEM_measurements_20250910_sample_13_after_irradiation.csv"
    combined_params = {**box1_params, **box4_params}
    
    # Box1 plots (excludes C1, C2)
    plot_sem_vs_power(sem_csv_path, combined_params, parameter='gap_width', box_filter='Box1')
    plt.savefig(os.path.join(output_folder, "sem_gap_width_vs_power_box1.png"))
    plt.close()

    plot_sem_vs_power(sem_csv_path, combined_params, parameter='interhole_distance', box_filter='Box1')
    plt.savefig(os.path.join(output_folder, "sem_interhole_distance_vs_power_box1.png"))
    plt.close()

    plot_sem_vs_power(sem_csv_path, combined_params, parameter='radius_top', box_filter='Box1')
    plt.savefig(os.path.join(output_folder, "sem_radius_top_vs_power_box1.png"))
    plt.close()

    # Box4 plots
    plot_sem_vs_power(sem_csv_path, combined_params, parameter='gap_width', box_filter='Box4')
    plt.savefig(os.path.join(output_folder, "sem_gap_width_vs_power_box4.png"))
    plt.close()

    plot_sem_vs_power(sem_csv_path, combined_params, parameter='interhole_distance', box_filter='Box4')
    plt.savefig(os.path.join(output_folder, "sem_interhole_distance_vs_power_box4.png"))
    plt.close()

    plot_sem_vs_power(sem_csv_path, combined_params, parameter='radius_top', box_filter='Box4')
    plt.savefig(os.path.join(output_folder, "sem_radius_top_vs_power_box4.png"))
    plt.close()

    # SEM before/after comparison analysis
    print("Analyzing SEM before/after measurements...")
    
    # CSV file paths
    sem_before_csv = r"Data\SEM\SEM_measurements_20250304_DNHs_pristine_200nm_sample_13.csv"
    sem_after_csv = r"Data\SEM\SEM_measurements_20250910_sample_13_after_irradiation.csv"
    
    # Match before and after measurements
    matched_sem_data = match_sem_measurements(sem_before_csv, sem_after_csv)
    
    # Parameters to analyze
    sem_parameters = ['gap_width', 'interhole_distance', 'radius_top', 'radius_bottom']
    
    # Create comparison plots for each parameter
    for parameter in sem_parameters:
        print(f"Creating SEM plots for {parameter}...")
        
        # Bar chart comparison (before vs after by position)
        plot_sem_before_after_comparison(matched_sem_data, parameter, output_folder)
        
        # Bar chart comparison WITH POWER information in labels
        plot_sem_before_after_comparison_with_power(matched_sem_data, parameter, combined_params, output_folder)
        
        # Scatter plot WITH POWER information in labels
        plot_sem_scatter_before_after_with_power(matched_sem_data, parameter, combined_params, output_folder)
        
        # Scatter plot (before vs after correlation)
        plot_sem_scatter_before_after(matched_sem_data, parameter, output_folder)
        
        # Power-based scatter plot (parameter vs power, before vs after)
        plot_sem_power_scatter_before_after(matched_sem_data, parameter, combined_params, output_folder)
        
        # Change distribution histogram
        plot_sem_change_distribution(matched_sem_data, parameter, output_folder)
        
        # Violin plot comparison (before vs after for both boxes)
        plot_sem_violin_comparison(matched_sem_data, parameter, output_folder)
    
    print("SEM before/after analysis complete!")
