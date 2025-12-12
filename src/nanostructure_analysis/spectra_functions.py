import glob
import os
import re
import numpy as np
import ast
import matplotlib.pyplot as plt
import pickle
import scipy.sparse as sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

def get_spectra(file_path):
    # Get all spectra in the folder, assuming the spectra are in the format *_spectrum.dat
    spectrum_pattern = "*_spectrum.dat"
    path = os.path.join(file_path, spectrum_pattern)
    path_to_spectra = glob.glob(path)
    return path_to_spectra

def get_spectra_data(file_path):
    # Get the data from the spectra, and store it in a dictionary, preserving the filename as the key
    data_dict = {}
    for file in file_path:
        data = np.loadtxt(file)
        base_name = os.path.basename(file).replace("_spectrum.dat", "")
        # Split the name and take only the middle parts (remove first 2 parts)
        name_parts = base_name.split('_')
        base_name = '_'.join(name_parts[2:])
        
        # Handle duplicate keys by adding suffix
        if base_name in data_dict:
            counter = 1
            while f"{base_name}_{counter}" in data_dict:
                counter += 1
            base_name = f"{base_name}_{counter}"

        data_dict[base_name] = data
    print(f"Number of spectra: {len(data_dict.keys())}")
    return data_dict

def get_spectra_params(file_path):
    # Get the parameters from the spectra, and store it in a dictionary, preserving the filename as the key
    params_dict = {}
    for file in file_path:
        file = file.replace("_spectrum.dat", "_params.txt")
        # Read the dictionary from the text file and safely evaluate it, since it is in a dictionary format in my case
        # Try different encodings to handle files with special characters
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        params_text = None
        
        for encoding in encodings:
            try:
                with open(file, 'r', encoding=encoding) as f:
                    params_text = f.read().strip()
                break
            except UnicodeDecodeError:
                continue
        
        if params_text is None:
            print(f"Warning: Could not decode file {file} with any encoding, skipping...")
            continue
            
        params = ast.literal_eval(params_text) 
        base_name = os.path.basename(file).replace("_params.txt", "")
        # Split the name and take only the middle parts (remove first 2 parts)
        name_parts = base_name.split('_')
        base_name = '_'.join(name_parts[2:])
        
        # Handle duplicate keys by adding suffix
        if base_name in params_dict:
            counter = 1
            while f"{base_name}_{counter}" in params_dict:
                counter += 1
            base_name = f"{base_name}_{counter}"

        # Dictionary of dictionaries with the filename as the key
        params_dict[base_name] = params
    print(f"Number of spectra: {len(params_dict.keys())}")
    return params_dict

def filter_spectra(data_dict, params_dict, pattern, average=False, exclude=None):
    # Use fnmatch to filter filenames directly

    filtered_data = {k: v for k, v in data_dict.items() if glob.fnmatch.fnmatch(k, pattern)}
    filtered_params = {k: v for k, v in params_dict.items() if glob.fnmatch.fnmatch(k, pattern)}
    if exclude:
        for excl_pattern in exclude:
            filtered_data = {k: v for k, v in filtered_data.items() if not glob.fnmatch.fnmatch(k, excl_pattern)}
            filtered_params = {k: v for k, v in filtered_params.items() if not glob.fnmatch.fnmatch(k, excl_pattern)}
    if not average:
        print(f"Found {len(filtered_data)} files matching '{pattern}'")
        return filtered_data, filtered_params
    
    # Group duplicate files by base name (remove _1, _2 etc suffixes)
    groups = {}
    for key in filtered_data.keys():
        base_key = key.split('_')[:-1] if key.split('_')[-1].isdigit() else [key]
        base_key = '_'.join(base_key) if base_key else key
        groups.setdefault(base_key, []).append(key)
    
    # Average groups and filter cosmic rays
    averaged_data, averaged_params = {}, {}
    for base_key, keys in groups.items():
        if len(keys) == 1:
            # Single file, no averaging needed
            averaged_data[base_key] = filtered_data[keys[0]]
            averaged_params[base_key] = filtered_params[keys[0]]
        else:
            # Multiple files, average with cosmic ray filtering
            spectra = np.array([filtered_data[k][:, 1] for k in keys])
            wavelengths = filtered_data[keys[0]][:, 0]
            
            # Filter cosmic rays: use median if any point deviates >3 std from mean
            mean_spectrum = np.mean(spectra, axis=0)
            std_spectrum = np.std(spectra, axis=0)
            filtered_spectrum = np.where(np.any(np.abs(spectra - mean_spectrum) > 2 * std_spectrum, axis=0),
                                       np.median(spectra, axis=0), mean_spectrum)
            
            averaged_data[base_key] = np.column_stack((wavelengths, filtered_spectrum))
            averaged_params[base_key] = filtered_params[keys[0]]  # Use first file's params
    
    print(f"Found {len(averaged_data)} averaged files matching '{pattern}'")
    return averaged_data, averaged_params


def baseline_als(y, lam=1e6, p=0.5, niter=10):
    """Asymmetric least-squares baseline, exposes lambda and asymmetry."""
    y = np.asarray(y, dtype=float)
    length = y.size
    difference = sparse.csc_matrix(np.diff(np.eye(length), 2))
    weights = np.ones(length)
    for _ in range(niter):
        weight_matrix = sparse.spdiags(weights, 0, length, length)
        system = weight_matrix + lam * difference.dot(difference.transpose())
        baseline = spsolve(system, weights * y)
        weights = p * (y > baseline) + (1 - p) * (y < baseline)
    return baseline


def compute_sensitivity_stat(values, percentile=95):
    """Return percentile of absolute sensitivity values."""
    magnitude = np.abs(np.asarray(values, dtype=float))
    if magnitude.size == 0:
        return 0.0
    return float(np.percentile(magnitude, percentile))


def spectra_main(file_path, savgol_filter=True):
    path_to_spectra = get_spectra(file_path)
    data_dict = get_spectra_data(path_to_spectra)
    params_dict = get_spectra_params(path_to_spectra)
    return data_dict, params_dict


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




def normalize_spectra_baseline(data_dict, baseline_dict, ref_dict, lam=1e7, p=0.5):
    """Normalize: (spectrum - baseline) / reference, then ALS baseline and normalize to max."""
    baseline = next(iter(baseline_dict.values()))
    ref = next(iter(ref_dict.values()))
    normalized_data = {}
    for key, data in data_dict.items():
        y = (data[:, 1] - baseline[:, 1]) / ref[:, 1]
        y = baseline_als(y, lam, p)
        y = y / np.max(y)
        normalized_data[key] = np.column_stack((data[:, 0], y))
    return normalized_data


def normalize_spectra(data_dict, ref_dict, lam=1e7, p=0.5, normalize_max=True):
    """Normalize: spectrum / reference, then normalize to max."""
    ref = next(iter(ref_dict.values()))
    normalized_data = {}
    for key, data in data_dict.items():
        y = data[:, 1] / ref[:, 1]
        y = baseline_als(y, lam, p)
        if normalize_max:
            y = y / np.max(y)
        normalized_data[key] = np.column_stack((data[:, 0], y))
    return normalized_data


if __name__ == "__main__":

    # EXAMPLE USAGE
    data_dict, params_dict = spectra_main("./Data/Spectra/20250821 - sample13 - after")

    box1_data, box1_params = filter_spectra(data_dict, params_dict, "*box1*")
    um5_data, um5_params = filter_spectra(data_dict, params_dict, "*5um*")
    bkg_data, bkg_params = filter_spectra(data_dict, params_dict, "*bkg*")



    normalized_data = normalize_spectra_bkg(box1_data, bkg_data['bkg_10000ms_1'], um5_data['5um_100ms_z_locked_1'], bkg_data['bkg_100ms_3'])


    for filename, data in normalized_data.items():
        plt.plot(data[:, 0], data[:, 1], label=filename)
    plt.legend()
    plt.show()

    for filename, data in um5_data.items():
        plt.plot(data[:, 0], data[:, 1], label=filename)
    plt.legend()


    for filename, data in bkg_data.items():
        plt.plot(data[:, 0], data[:, 1], label=filename)
    plt.legend()









