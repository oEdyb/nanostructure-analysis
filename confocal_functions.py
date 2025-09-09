import glob
import os
import numpy as np
import ast
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pickle
import hashlib
from plotting_functions import *


def load_with_cache(file_path, loader_func):
    # Create cache filename from path hash
    path_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    cache_file = f"cache/confocal_{path_hash}.pkl"
    
    # Try to load from cache
    if os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    # Load fresh data and cache it
    print(f"Loading fresh data and caching to: {cache_file}")
    data = loader_func(file_path)
    
    # Ensure cache directory exists
    os.makedirs("cache", exist_ok=True)
    
    # Save to cache
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    
    return data


def get_confocal(file_path):
    # Get all confocal files with _image pattern as base files
    confocal_pattern = "*_image*"
    path = os.path.join(file_path, confocal_pattern)
    path_to_confocal = glob.glob(path)

    # Print the files that match the pattern
    print(f"\nFiles matching {confocal_pattern}:")
    for file in path_to_confocal:
        print(f"{os.path.basename(file)}")
    return path_to_confocal

def get_confocal_data(file_path):
    # Initialize dictionaries for different data types
    image_dict = {}
    xy_dict = {}
    z_dict = {}
    apd_dict = {}
    monitor_dict = {}
    
    for file in file_path:
        # Get base name without _image
        base_name = os.path.basename(file).replace("_image", "")
        
        # Load data if files exist
        image_dict[base_name] = np.load(file) if os.path.exists(file) else None
        xy_dict[base_name] = np.load(file.replace("_image", "_xy_coords")) if os.path.exists(file.replace("_image", "_xy_coords")) else None
        z_dict[base_name] = np.load(file.replace("_image", "_z_scan")) if os.path.exists(file.replace("_image", "_z_scan")) else None
        apd_dict[base_name] = np.load(file.replace("_image", "_confocal_apd_traces")) if os.path.exists(file.replace("_image", "_confocal_apd_traces")) else None
        monitor_dict[base_name] = np.load(file.replace("_image", "_confocal_monitor_traces")) if os.path.exists(file.replace("_image", "_confocal_monitor_traces")) else None
        
        print(f"Loaded {base_name}")
        
    print(f"Loaded {len(image_dict)} confocal measurements")
    return image_dict, apd_dict, monitor_dict, xy_dict, z_dict


def filter_confocal(data_dict, pattern, exclude=None):
    # Use fnmatch to filter filenames directly, the params dict is not used here
    filtered_data = {k: v for k, v in data_dict.items() if glob.fnmatch.fnmatch(k, pattern)}
    if exclude:
        filtered_data = {k: v for k, v in filtered_data.items() if all(exclude not in k for exclude in exclude)}
    print(f"Found {len(filtered_data)} files matching '{pattern}'")
    return filtered_data

def confocal_main(file_path):
    path_to_confocal = get_confocal(file_path)
    data_dict = get_confocal_data(path_to_confocal)
    return data_dict




if __name__ == "__main__":

    # EXAMPLE USAGE WITHOUT CACHE
    # image_dict, apd_dict, monitor_dict, xy_dict, z_dict = confocal_main(r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.08.12 - Sample 13 Before box1 box4")
    
    # EXAMPLE USAGE WITH CACHE
    data_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.09.02 - Sample 13 before after break"
    image_dict, apd_dict, monitor_dict, xy_dict, z_dict = load_with_cache(data_path, confocal_main)
    image_dict_before = filter_confocal(image_dict, "*[!_after*]", ["C2"])
    image_dict_after = filter_confocal(image_dict, "*after*", ["C2"])

    plot_confocal_image_comparison(image_dict_before, image_dict_after)
    plt.show()

    plot_confocal_comparison(image_dict_before, image_dict_after)
    plt.show()






    




