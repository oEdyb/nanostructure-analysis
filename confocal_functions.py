import glob
import os
import numpy as np
import ast
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pickle
from scipy.optimize import curve_fit
from apd_functions import apd_load_main, filter_apd

###############################################
# LOAD WITH CACHE -> FILTER -> GET CONFOCAL DATA -> PLOT
###############################################

def load_with_cache(file_path, loader_func):
    # Create cache filename from path
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    cache_name = file_path.replace("/", "_").replace("\\", "_").replace(":", "").replace(" ", "_")
    cache_file = os.path.join(cache_dir, f"confocal_{cache_name}.pkl")

    # Try loading from cache first
    if os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            print("Cache corrupted, reloading...")

    # Load fresh data if no cache
    print(f"Loading fresh data and caching to: {cache_file}")
    data = loader_func(file_path)

    # Save to cache
    print(f"Saving to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

    return data


def get_confocal(file_path):
    # Get all confocal files with _image pattern as base files
    confocal_pattern = "*_image*"
    path = os.path.join(file_path, confocal_pattern)
    path_to_confocal = glob.glob(path)
    return path_to_confocal

def get_confocal_data(file_path):
    # Initialize dictionaries for different data types
    image_dict = {}
    xy_dict = {}
    z_dict = {}
    apd_dict = {}
    monitor_dict = {}
    
    for file in file_path:
        # Get base name without _image and .npy extension
        base_name = os.path.basename(file).replace("_image", "").replace(".npy", "")
        
        # Load data if files exist
        image_dict[base_name] = np.load(file) if os.path.exists(file) else None
        xy_dict[base_name] = np.load(file.replace("_image", "_xy_coords")) if os.path.exists(file.replace("_image", "_xy_coords")) else None
        z_dict[base_name] = np.load(file.replace("_image", "_z_scan")) if os.path.exists(file.replace("_image", "_z_scan")) else None
        
        # Try different APD trace naming conventions
        apd_file_new = file.replace("_image", "_confocal_apd_traces")
        apd_file_old = file.replace("_image", "_confocal_traces")
        monitor_file = file.replace("_image", "_confocal_monitor_traces")
        
        if os.path.exists(apd_file_new):
            # New format: separate APD and monitor files
            apd_dict[base_name] = np.load(apd_file_new)
            monitor_dict[base_name] = np.load(monitor_file) if os.path.exists(monitor_file) else None
        elif os.path.exists(apd_file_old):
            # Old format: only APD traces, no separate monitor
            apd_dict[base_name] = np.load(apd_file_old)
            monitor_dict[base_name] = None
        else:
            apd_dict[base_name] = None
            monitor_dict[base_name] = None

    print(f"Loaded {len(image_dict)} confocal measurements")
    return image_dict, apd_dict, monitor_dict, xy_dict, z_dict


def filter_confocal(confocal_data, pattern, exclude=None):
    # Extract image_dict from tuple and filter it
    image_dict, apd_dict, monitor_dict, xy_dict, z_dict = confocal_data
    filtered_image = {k: v for k, v in image_dict.items() if glob.fnmatch.fnmatch(k, pattern)}
    filtered_apd = {k: v for k, v in apd_dict.items() if glob.fnmatch.fnmatch(k, pattern)}
    filtered_monitor = {k: v for k, v in monitor_dict.items() if glob.fnmatch.fnmatch(k, pattern)}
    filtered_xy = {k: v for k, v in xy_dict.items() if glob.fnmatch.fnmatch(k, pattern)}
    filtered_z = {k: v for k, v in z_dict.items() if glob.fnmatch.fnmatch(k, pattern)}
    
    if exclude:
        # Exclude files that contain any of the exclude strings
        filtered_image = {k: v for k, v in filtered_image.items() if not any(excl in k for excl in exclude)}
        filtered_apd = {k: v for k, v in filtered_apd.items() if not any(excl in k for excl in exclude)}
        filtered_monitor = {k: v for k, v in filtered_monitor.items() if not any(excl in k for excl in exclude)}
        filtered_xy = {k: v for k, v in filtered_xy.items() if not any(excl in k for excl in exclude)}
        filtered_z = {k: v for k, v in filtered_z.items() if not any(excl in k for excl in exclude)}
    
    print(f"Found {len(filtered_image)} files matching '{pattern}'")
    return (filtered_image, filtered_apd, filtered_monitor, filtered_xy, filtered_z)


def analyze_confocal(confocal_data):
    # Extract image_dict and apd_dict from tuple
    image_dict, apd_dict = confocal_data[0], confocal_data[1]
    
    # 2D Gaussian function for fitting
    def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, offset):
        x, y = xy
        return (amplitude * np.exp(-((x-xo)**2/(2*sigma_x**2) + (y-yo)**2/(2*sigma_y**2))) + offset).ravel()

    # Initialize results dict
    results_dict = {}
    
    for key in image_dict.keys():
        y, x = np.mgrid[:image_dict[key].shape[0], :image_dict[key].shape[1]]
        
        # Fit 2D Gaussian to data
        try:
            initial_guess = [image_dict[key].max(), image_dict[key].shape[1]//2, image_dict[key].shape[0]//2, 5, 5, image_dict[key].min()]
            popt, _ = curve_fit(gaussian_2d, (x, y), image_dict[key].ravel(), p0=initial_guess)
        except:
            # Return default values if fit fails
            popt = [image_dict[key].max(), image_dict[key].shape[1]//2, image_dict[key].shape[0]//2, 5, 5, image_dict[key].min()]
        
        # Calculate APD trace statistics if available
        # We now calculate SNR for each individual trace in the 3x3 region, then average the SNRs.
        trace_stats = {}
        if key in apd_dict.keys() and apd_dict[key] is not None:
            traces = apd_dict[key]  # Shape: (height, width, trace_length)
            cy, cx = traces.shape[0]//2, traces.shape[1]//2  # Image center

            # Use gaussian center if within middle 3x3, else use image center
            gy, gx = int(popt[2]), int(popt[1])  # Gaussian center
            if abs(gy - cy) <= 1 and abs(gx - cx) <= 1:
                cy, cx = gy, gx

            # Extract 3x3 region around center
            center_3x3 = traces[cy-1:cy+2, cx-1:cx+2]  # shape: (3, 3, trace_length)

            # Calculate SNR for each individual trace in the 3x3 region
            snrs = []
            means = []
            stds = []
            for i in range(center_3x3.shape[0]):
                for j in range(center_3x3.shape[1]):
                    trace = center_3x3[i, j][500:]
                    mean = np.mean(trace)
                    std = np.std(trace)
                    snr = mean / std if std > 0 else 0
                    snrs.append(snr)
                    means.append(mean)
                    stds.append(std)

            # Average SNR, mean, std over the 9 traces
            avg_snr = np.mean(snrs)
            avg_mean = np.mean(means)
            avg_std = np.mean(stds)

            # Store results in trace_stats
            trace_stats = {
                'snr_3x3': avg_snr,
                'mean_3x3': avg_mean,
                'std_3x3': avg_std
            }
        # Extract PSF parameters from popt
        psf_params = {
            'amplitude': popt[0],
            'x_center': popt[1], 
            'y_center': popt[2],
            'sigma_x': popt[3],
            'sigma_y': popt[4],
            'offset': popt[5]
        }
        
        results_dict[key] = {'popt': popt, 'max_value': image_dict[key].max(), **trace_stats, **psf_params}
    
    return results_dict


def confocal_main(file_path):
    path_to_confocal = get_confocal(file_path)
    data_dict = get_confocal_data(path_to_confocal)
    return data_dict


def get_all_confocal(file_path):
    confocal_data = load_with_cache(file_path, confocal_main)
    return confocal_data




if __name__ == "__main__":

    # EXAMPLE USAGE WITHOUT CACHE
    # image_dict, apd_dict, monitor_dict, xy_dict, z_dict = confocal_main(r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.08.12 - Sample 13 Before box1 box4")
    
    # EXAMPLE USAGE WITH CACHE
    data_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.09.02 - Sample 13 before after break"
    apd_path = r"Data\APD\2025.09.02 - Sample 13 PT high power"

    apd_data, monitor_data, apd_params = apd_load_main(apd_path)
    apd_data_box1, monitor_data_box1, apd_params_box1 = filter_apd(apd_data, monitor_data, apd_params, "*box1*")
    apd_data_box4, monitor_data_box4, apd_params_box4 = filter_apd(apd_data, monitor_data, apd_params, "*box4*")
    confocal_data = load_with_cache(data_path, confocal_main)
    confocal_data_before = filter_confocal(confocal_data, "box1*[!_after*]", ["C2"])
    confocal_data_after = filter_confocal(confocal_data, "*box1*after*", ["C2"])
    
    results_dict_before = analyze_confocal(confocal_data_before)
    results_dict_after = analyze_confocal(confocal_data_after)

    # Extract popt for plotting functions that expect the old format
    popt_before = {k: v['popt'] for k, v in results_dict_before.items()}
    popt_after = {k: v['popt'] for k, v in results_dict_after.items()}





