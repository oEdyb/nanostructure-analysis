"""
APD (Avalanche Photodiode) time trace data loading and analysis functions.
"""

import glob
import os
import ast
import pickle
import numpy as np
from typing import Dict, NamedTuple, Optional

from . import config


# ============================================================================
# CONSTANTS (imported from config)
# ============================================================================

CACHE_DIR = config.CACHE_DIR
APD_SAVE_DIR = config.APD_SAVE_DIR
DEFAULT_DOWNSAMPLE_FACTOR = config.DEFAULT_DOWNSAMPLE_FACTOR
DEFAULT_POWER_FACTOR = config.DEFAULT_POWER_FACTOR


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class APDData(NamedTuple):
    transmission: Dict
    monitor: Dict
    params: Dict

    def __repr__(self) -> str:
        """Pretty representation showing measurement statistics."""
        n_meas = len(self.transmission)

        if n_meas == 0:
            return "APDData(measurements=0)"

        # Calculate stats
        avg_power = np.mean([p.get('power', 0) for p in self.params.values()])
        powers = [p.get('power', 0) for p in self.params.values()]
        power_range = f"{min(powers):.2f}-{max(powers):.2f}" if powers else "N/A"

        # Get trace length from first measurement
        first_trace = next(iter(self.transmission.values()))
        trace_len = len(first_trace) if first_trace is not None else 0

        # Sample keys
        sample_keys = list(self.transmission.keys())[:3]

        return (f"APDData(measurements={n_meas})\n"
                f"  Transmission: {n_meas}, Monitor: {len(self.monitor)}\n"
                f"  Trace length: {trace_len} points\n"
                f"  Power: {avg_power:.2f}mW (range: {power_range}mW)\n"
                f"  Sample keys: {sample_keys}")


# ============================================================================
# DATA LOADING
# ============================================================================

def downsample_trace(data, factor):
    """Downsample by averaging chunks."""
    n_points = len(data) // factor * factor
    return data[:n_points].reshape(-1, factor).mean(axis=1)


def preprocess_apd_directory(file_path: str, downsample_factor: int = DEFAULT_DOWNSAMPLE_FACTOR):
    """Downsample raw APD files and save to local APD/ directory."""
    pattern = os.path.join(file_path, "*_transmission.npy")
    files = glob.glob(pattern)

    if not files:
        print(f"No APD files found in {file_path}")
        return

    original_folder = os.path.basename(file_path)
    save_folder = os.path.join(APD_SAVE_DIR, original_folder)
    os.makedirs(save_folder, exist_ok=True)

    print(f"Downsampling {len(files)} files by factor of {downsample_factor}")
    print(f"Saving to: {save_folder}")

    for file in files:
        base_name = os.path.basename(file).replace("_transmission.npy", "")

        # Load and downsample
        apd_data = downsample_trace(np.load(file), downsample_factor)
        monitor_data = downsample_trace(np.load(file.replace("_transmission.npy", "_monitor.npy")), downsample_factor)

        # Save
        np.save(os.path.join(save_folder, f"{base_name}_transmission.npy"), apd_data)
        np.save(os.path.join(save_folder, f"{base_name}_monitor.npy"), monitor_data)

        # Copy params
        with open(file.replace("_transmission.npy", "_params.txt"), 'r') as f:
            params_content = f.read()
        with open(os.path.join(save_folder, f"{base_name}_params.txt"), 'w') as f:
            f.write(params_content)

        print(f"  Processed: {base_name}")


def get_apd_data(file_path: str, default_power_factor: Optional[float] = None) -> APDData:
    """Load APD data from local APD directory."""
    # Handle both Windows and Linux paths correctly
    # Replace Windows backslashes with forward slashes for consistent path handling
    normalized_path = file_path.replace('\\', '/')
    folder_name = os.path.basename(normalized_path)
    apd_path = os.path.join(APD_SAVE_DIR, folder_name)

    if not os.path.exists(apd_path):
        raise FileNotFoundError(
            f"APD directory not found: {apd_path}\n"
            f"Run preprocess_apd_directory('{file_path}') first"
        )

    pattern = os.path.join(apd_path, "*_transmission.npy")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No APD files found in {apd_path}")

    transmission, monitor, params = {}, {}, {}

    for file in files:
        # Extract label (remove first 2 timestamp parts)
        base_name = os.path.basename(file).replace("_transmission.npy", "")
        label = '_'.join(base_name.split('_')[2:])

        # Load data
        transmission[label] = np.load(file)
        monitor[label] = np.load(file.replace("_transmission.npy", "_monitor.npy"))

        # Load params
        with open(file.replace("_transmission.npy", "_params.txt"), 'r') as f:
            params[label] = ast.literal_eval(f.read().strip())

        # Calculate power
        power_factor = params[label].get('Power calibration factor (mW/V)',
                                        default_power_factor or DEFAULT_POWER_FACTOR)
        params[label]['power'] = np.mean(monitor[label]) * power_factor

    return APDData(transmission, monitor, params)


def load_with_cache(file_path: str, default_power_factor: Optional[float] = None) -> APDData:
    """Load APD data with caching."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Generate cache filename
    cache_name = file_path.replace("/", "_").replace("\\", "_").replace(":", "").replace(" ", "_")
    cache_file = os.path.join(CACHE_DIR, f"apd_{cache_name}.pkl")

    # Try loading from cache
    if os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Cache corrupted ({e}), reloading...")

    # Load fresh data
    print(f"Loading fresh data from: {file_path}")
    data = get_apd_data(file_path, default_power_factor)

    # Save to cache
    print(f"Saving to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

    return data


def apd_main(file_path: str, default_power_factor: Optional[float] = None) -> APDData:
    """Main entry point for loading APD data."""
    return load_with_cache(file_path, default_power_factor)


# Backward compatibility alias
apd_load_main = apd_main


# ============================================================================
# DATA FILTERING
# ============================================================================

def filter_apd(data: APDData, pattern, exclude=None):
    """Filter APD data by filename patterns.

    Args:
        data: APDData object to filter
        pattern: Pattern or list of patterns to include (e.g., "box1" or "*box1*")
        exclude: Optional list of patterns to exclude

    Returns:
        Filtered APDData object
    """
    patterns = [pattern] if isinstance(pattern, str) else pattern
    exclude = [exclude] if isinstance(exclude, str) else (exclude or [])

    # Auto-add wildcards
    patterns = [f"*{p}*" if not p.startswith("*") and not p.endswith("*") else p for p in patterns]
    exclude = [f"*{e}*" if not e.startswith("*") and not e.endswith("*") else e for e in exclude]

    # Get all keys
    all_keys = set(data.transmission.keys())

    # Filter: must match ALL patterns (AND logic)
    filtered_keys = set(all_keys)
    for pat in patterns:
        filtered_keys = {k for k in filtered_keys if glob.fnmatch.fnmatch(k, pat)}

    # Exclude
    for excl_pat in exclude:
        filtered_keys = {k for k in filtered_keys if not glob.fnmatch.fnmatch(k, excl_pat)}

    print(f"Found {len(filtered_keys)} measurements matching pattern")

    return APDData(
        {k: v for k, v in data.transmission.items() if k in filtered_keys},
        {k: v for k, v in data.monitor.items() if k in filtered_keys},
        {k: v for k, v in data.params.items() if k in filtered_keys}
    )


if __name__ == "__main__":
    # Example usage
    path = r"\\AMIPC045962\Cache (D)\daily_data\apd_traces\2025.05.1314 Sample 7"

    # First time: preprocess raw data
    # preprocess_apd_directory(path, downsample_factor=100)

    # Load data (cached)
    data = apd_main(path)
    print(data)

    # Filter
    filtered = filter_apd(data, "box1", exclude="after")
    print(filtered)
