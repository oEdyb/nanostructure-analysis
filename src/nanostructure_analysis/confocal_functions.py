"""
Confocal microscopy data loading, filtering, and analysis functions.

This module handles:
- Loading confocal image scans and APD traces from .npy files
- Caching data for faster subsequent loads
- Filtering measurements by filename patterns
- Analyzing point spread functions (PSF) via 2D Gaussian fits
- Computing signal-to-noise ratios (SNR) from APD traces
"""

import glob
import os
import pickle
from typing import Dict, List, Optional, Tuple, Callable, NamedTuple
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from . import config


# ============================================================================
# CONSTANTS (imported from config)
# ============================================================================

CACHE_DIR = config.CACHE_DIR
TRACE_SKIP_INITIAL_SAMPLES = config.TRACE_SKIP_INITIAL_SAMPLES
CENTER_REGION_SIZE = config.CENTER_REGION_SIZE
DEFAULT_SIGMA_GUESS = config.DEFAULT_SIGMA_GUESS

DATA_TYPE_REGISTRY = {
    # New formats
    'apd_traces': '_confocal_apd_traces',
    'monitor_traces': '_confocal_monitor_traces',
    'z_scan_traces': '_z_scan_traces',
    'image': '_image',
    'xy_coords': '_xy_coords',
    'z_scan': '_z_scan',

    # Old formats (will be normalized)
    'apd_traces_old1': '_confocal_traces',
    'apd_traces_old2': '_traces',  # Very old format
    'xy_coords_old': '_coords', 
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class ConfocalData(NamedTuple):
    images: Dict
    apd_traces: Dict
    monitor_traces: Dict
    xy_coords: Dict
    z_scans: Dict
    z_scan_traces: Dict

    def __repr__(self) -> str:
        """Pretty representation showing number of measurements."""
        def count_data(d):
            """Count non-None values in nested dict."""
            return sum(1 for main_dict in d.values()
                        for v in main_dict.values() if v is not None)

        n_imgs = count_data(self.images)
        n_apd = count_data(self.apd_traces)
        n_monitor = count_data(self.monitor_traces)
        n_xy = count_data(self.xy_coords)
        n_z = count_data(self.z_scans)
        n_z_traces = count_data(self.z_scan_traces)

        main_names = list(self.images.keys())[:3]

        return (f"ConfocalData(main_names={len(self.images)})\n"
                f"  Images: {n_imgs}, APD: {n_apd}, Monitor: {n_monitor}\n"
                f"  XY: {n_xy}, Z-scans: {n_z}, Z-traces: {n_z_traces}\n"
                f"  Sample keys: {main_names}")



# ============================================================================
# DATA LOADING
# ============================================================================

def safe_load(filepath: str):
    try: 
        data = np.load(filepath)
        return data
    except Exception:
        print(f"Warning: Could not load file {filepath}")
        return None

def parse_npy_filename(filename: str):
    base = filename.replace('.npy', '')
    base = base.rsplit("\\", 1)[-1]

    for data_type, pattern in DATA_TYPE_REGISTRY.items():
        if pattern in base:
            parts = base.rsplit(pattern, 1)
            if len(parts) == 2:
                # Normalize old formats to standard keys
                if 'apd_traces_old' in data_type:
                    data_type = 'apd_traces'  # All old APD formats -> 'apd_traces'
                elif data_type == 'xy_coords_old':  
                    data_type = 'xy_coords'

                return parts[0].rstrip('_'), data_type, parts[1].lstrip('_')

    return None


def get_confocal_data(file_path: str) -> ConfocalData:
    # Step 1: Group files by main_name
    grouped = {}  # {main_name: {data_type: filepath}}

    for npy_file in glob.glob(os.path.join(file_path, "*.npy")):
        parsed = parse_npy_filename(npy_file)
        if parsed is None:
            continue

        main_name, data_type, index = parsed

        if main_name not in grouped:
            grouped[main_name] = {}
        if data_type not in grouped[main_name]:
            grouped[main_name][data_type] = {}
        grouped[main_name][data_type][index] = npy_file

    # Step 2: Load all data types for each main_name
    images, apd_traces, monitor_traces, xy_coords, z_scans, z_scan_traces = {}, {}, {}, {}, {}, {}

    for main_name, data_types in grouped.items():
        # For each data type, load all indices
        images[main_name] = {idx: safe_load(fp) for idx, fp in data_types.get('image', {}).items()}
        xy_coords[main_name] = {idx: safe_load(fp) for idx, fp in data_types.get('xy_coords', {}).items()}
        z_scans[main_name] = {idx: safe_load(fp) for idx, fp in data_types.get('z_scan', {}).items()}
        apd_traces[main_name] = {idx: safe_load(fp) for idx, fp in data_types.get('apd_traces', {}).items()}
        monitor_traces[main_name] = {idx: safe_load(fp) for idx, fp in data_types.get('monitor_traces', {}).items()}
        z_scan_traces[main_name] = {idx: safe_load(fp) for idx, fp in data_types.get('z_scan_traces', {}).items()}

    return ConfocalData(images, apd_traces, monitor_traces, xy_coords, z_scans, z_scan_traces)
            


def filter_confocal(data: ConfocalData, pattern, exclude=None):
    """Filter confocal data by filename patterns.

    Args:
        data: ConfocalData object to filter
        pattern: Pattern or list of patterns to include (e.g., "box1" or "*box1*")
        exclude: Optional list of patterns to exclude

    Returns:
        Filtered ConfocalData object
    """
    patterns = [pattern] if isinstance(pattern, str) else pattern
    exclude = exclude or []

    # Auto-add wildcards if not present
    patterns = [f"*{p}*" if not p.startswith("*") and not p.endswith("*") else p for p in patterns]
    exclude = [f"*{e}*" if not e.startswith("*") and not e.endswith("*") else e for e in exclude]

     # Get all unique main_names across all data types
    all_keys = set()
    all_keys.update(data.images.keys())
    all_keys.update(data.apd_traces.keys())
    all_keys.update(data.monitor_traces.keys())
    all_keys.update(data.xy_coords.keys())
    all_keys.update(data.z_scans.keys())
    all_keys.update(data.z_scan_traces.keys())

    # Filter: must match ALL patterns (AND logic)
    filtered_names = set(all_keys)
    for pat in patterns:
        filtered_names = {k for k in filtered_names if glob.fnmatch.fnmatch(k, pat)}

    for excl_pat in exclude:
        filtered_names = {k for k in filtered_names if not glob.fnmatch.fnmatch(k, excl_pat)}

    print(f"Found {len(filtered_names)} main_names matching pattern")

    # Build filtered ConfocalData
    return ConfocalData(
        {k: v for k, v in data.images.items() if k in filtered_names},
        {k: v for k, v in data.apd_traces.items() if k in filtered_names},
        {k: v for k, v in data.monitor_traces.items() if k in filtered_names},
        {k: v for k, v in data.xy_coords.items() if k in filtered_names},
        {k: v for k, v in data.z_scans.items() if k in filtered_names},
        {k: v for k, v in data.z_scan_traces.items() if k in filtered_names}
    )


def load_with_cache(file_path: str) -> ConfocalData:
    """Load confocal data with caching."""

    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Generate cache filename from path
    cache_name = file_path.replace("/", "_").replace("\\", "_").replace(":", "").replace(" ", "_")
    cache_file = os.path.join(CACHE_DIR, f"confocal_{cache_name}.pkl")

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
    data = get_confocal_data(file_path)

    # Save to cache
    print(f"Saving to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

    return data


# Then use it like this:
def confocal_main(file_path: str) -> ConfocalData:
    return load_with_cache(file_path)


if __name__ == "__main__":
    path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\20251129 - Sample 20 White Laser scans - Good Signal"
    data = confocal_main(path)
    print(data)



