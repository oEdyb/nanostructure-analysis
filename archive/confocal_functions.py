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
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from .apd_functions import apd_main as apd_load_main, filter_apd
from . import config


# ============================================================================
# CONSTANTS (imported from config)
# ============================================================================

CACHE_DIR = config.CACHE_DIR
TRACE_SKIP_INITIAL_SAMPLES = config.TRACE_SKIP_INITIAL_SAMPLES
CENTER_REGION_SIZE = config.CENTER_REGION_SIZE
DEFAULT_SIGMA_GUESS = config.DEFAULT_SIGMA_GUESS


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class ConfocalData(NamedTuple):
    """Container for all confocal measurement data.

    Attributes:
        images: Dictionary of 2D confocal scan images {label: array}
        apd_traces: Dictionary of 3D APD time traces {label: array(height, width, time)}
        monitor_traces: Dictionary of 3D monitor time traces {label: array(height, width, time)}
        xy_coords: Dictionary of XY coordinate arrays {label: array}
        z_scans: Dictionary of Z-axis scan data {label: array}
    """
    images: Dict[str, Optional[npt.NDArray]]
    apd_traces: Dict[str, Optional[npt.NDArray]]
    monitor_traces: Dict[str, Optional[npt.NDArray]]
    xy_coords: Dict[str, Optional[npt.NDArray]]
    z_scans: Dict[str, Optional[npt.NDArray]]

    def __repr__(self) -> str:
        """Pretty representation showing number of measurements."""
        return (f"ConfocalData(measurements={len(self.images)}, "
                f"with_apd={sum(1 for v in self.apd_traces.values() if v is not None)})")


class PSFParameters(NamedTuple):
    """Point spread function parameters from 2D Gaussian fit.

    Attributes:
        amplitude: Peak intensity above offset
        x_center: Horizontal center position (pixels)
        y_center: Vertical center position (pixels)
        sigma_x: Horizontal standard deviation (pixels)
        sigma_y: Vertical standard deviation (pixels)
        offset: Background intensity level
    """
    amplitude: float
    x_center: float
    y_center: float
    sigma_x: float
    sigma_y: float
    offset: float


class TraceStatistics(NamedTuple):
    """APD trace statistics from center region.

    Attributes:
        snr: Average signal-to-noise ratio (mean/std)
        mean_counts: Average photon count rate
        std_counts: Standard deviation of count rate
    """
    snr: float
    mean_counts: float
    std_counts: float


class ConfocalAnalysisResult(NamedTuple):
    """Complete analysis results for a single confocal measurement.

    Attributes:
        psf: Point spread function fit parameters
        max_intensity: Peak pixel value in the image
        trace_stats: APD trace statistics (None if no traces available)
    """
    psf: PSFParameters
    max_intensity: float
    trace_stats: Optional[TraceStatistics]


# ============================================================================
# CACHING UTILITIES
# ============================================================================

def load_with_cache(file_path: str, loader_func: Callable) -> ConfocalData:
    """Load data with automatic caching to speed up subsequent loads.

    Creates a cache file in the cache/ directory. On first load, runs loader_func
    and saves the result. On subsequent loads, returns the cached data directly.

    Args:
        file_path: Path to the data directory
        loader_func: Function to call for loading fresh data (takes file_path as argument)

    Returns:
        Loaded confocal data (from cache or fresh)

    Example:
        >>> data = load_with_cache("path/to/data", confocal_main)
    """
    # Create cache directory if needed
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Generate cache filename from path (sanitize for filesystem)
    cache_name = (file_path.replace("/", "_")
                           .replace("\\", "_")
                           .replace(":", "")
                           .replace(" ", "_"))
    cache_file = os.path.join(CACHE_DIR, f"confocal_{cache_name}.pkl")

    # Try loading from cache first
    if os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, AttributeError) as e:
            print(f"Cache corrupted ({e.__class__.__name__}), reloading...")

    # Load fresh data if no valid cache exists
    print(f"Loading fresh data and caching to: {cache_file}")
    data = loader_func(file_path)

    # Save to cache
    print(f"Saving to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

    return data


# ============================================================================
# DATA LOADING
# ============================================================================

def get_confocal_files(file_path: str) -> List[str]:
    """Find all confocal measurement files in a directory.

    Searches for files with _image*.npy pattern as base files. Also includes
    z_scan files that don't have a corresponding image file.

    Args:
        file_path: Directory path to search

    Returns:
        List of full paths to confocal measurement files
    """
    # Get all image files as base measurements
    image_files = set(glob.glob(os.path.join(file_path, "*_image*.npy")))
    z_files = glob.glob(os.path.join(file_path, "*_z_scan*.npy"))

    # Add z_scan files that don't have a corresponding image file
    for z_file in z_files:
        base = os.path.basename(z_file).replace("_z_scan", "_image")
        if not any(base in f for f in image_files):
            image_files.add(z_file)

    return list(image_files)


def get_confocal_data(file_paths: List[str]) -> ConfocalData:
    """Load all confocal data from a list of measurement files.

    For each measurement, loads:
    - Image scan (*_image.npy or *_z_scan.npy)
    - XY coordinates (*_xy_coords.npy)
    - Z scan data (*_z_scan.npy)
    - APD traces (*_confocal_apd_traces.npy or old *_confocal_traces.npy)
    - Monitor traces (*_confocal_monitor_traces.npy, if available)

    Args:
        file_paths: List of paths to confocal measurement files

    Returns:
        ConfocalData containing all loaded measurements

    Note:
        Handles both old format (single traces file) and new format
        (separate APD and monitor files)
    """
    # Initialize dictionaries for different data types
    images = {}
    apd_traces = {}
    monitor_traces = {}
    xy_coords = {}
    z_scans = {}

    for file in file_paths:
        # Extract base name from file path
        if "_image" in file:
            base_name = os.path.basename(file).replace("_image", "").replace(".npy", "")
        elif "_z_scan" in file:
            base_name = os.path.basename(file).replace("_z_scan", "").replace(".npy", "")
        else:
            continue

        # Helper to safely load .npy file if it exists
        def safe_load(filepath: str) -> Optional[npt.NDArray]:
            return np.load(filepath) if os.path.exists(filepath) else None

        # Load image data (prefer _image over _z_scan)
        image_file = file.replace("_z_scan", "_image")
        images[base_name] = safe_load(image_file)

        # Load coordinate data
        xy_file = file.replace("_image", "_xy_coords").replace("_z_scan", "_xy_coords")
        xy_coords[base_name] = safe_load(xy_file)

        # Load z-scan data
        z_file = file.replace("_image", "_z_scan")
        z_scans[base_name] = safe_load(z_file)

        # Load APD traces - check both new and old naming conventions
        apd_file_new = file.replace("_image", "_confocal_apd_traces").replace("_z_scan", "_confocal_apd_traces")
        apd_file_old = file.replace("_image", "_confocal_traces").replace("_z_scan", "_confocal_traces")
        monitor_file = file.replace("_image", "_confocal_monitor_traces").replace("_z_scan", "_confocal_monitor_traces")

        if os.path.exists(apd_file_new):
            # New format: separate APD and monitor files
            apd_traces[base_name] = np.load(apd_file_new)
            monitor_traces[base_name] = safe_load(monitor_file)
        elif os.path.exists(apd_file_old):
            # Old format: only APD traces, no separate monitor
            apd_traces[base_name] = np.load(apd_file_old)
            monitor_traces[base_name] = None
        else:
            apd_traces[base_name] = None
            monitor_traces[base_name] = None

    print(f"Loaded {len(images)} confocal measurements")
    return ConfocalData(
        images=images,
        apd_traces=apd_traces,
        monitor_traces=monitor_traces,
        xy_coords=xy_coords,
        z_scans=z_scans
    )


def confocal_main(file_path: str) -> ConfocalData:
    """Main entry point for loading confocal data from a directory.

    Args:
        file_path: Path to directory containing confocal .npy files

    Returns:
        ConfocalData with all measurements found in the directory
    """
    measurement_files = get_confocal_files(file_path)
    return get_confocal_data(measurement_files)


def get_all_confocal(file_path: str) -> ConfocalData:
    """Load all confocal data with automatic caching.

    This is the recommended function for loading confocal data as it uses
    caching to avoid slow reloads on subsequent runs.

    Args:
        file_path: Path to directory containing confocal .npy files

    Returns:
        ConfocalData with all measurements (from cache or fresh load)
    """
    return load_with_cache(file_path, confocal_main)


# ============================================================================
# DATA FILTERING
# ============================================================================

def filter_confocal(data: ConfocalData,
                   pattern: str,
                   exclude: Optional[List[str]] = None) -> ConfocalData:
    """Filter confocal data by filename pattern with optional exclusions.

    Args:
        data: ConfocalData to filter
        pattern: Glob pattern to match (e.g., "*box1*", "A[1-3]*")
        exclude: List of strings - exclude keys containing any of these substrings

    Returns:
        New ConfocalData containing only matching measurements

    Example:
        >>> # Get box1 measurements, excluding "after" and "C2"
        >>> filtered = filter_confocal(data, "*box1*", exclude=["after", "C2"])
    """
    def filter_dict(d: dict) -> dict:
        """Apply pattern and exclusion filters to a dictionary."""
        # First apply pattern matching
        filtered = {k: v for k, v in d.items() if glob.fnmatch.fnmatch(k, pattern)}

        # Then apply exclusions if specified
        if exclude:
            filtered = {k: v for k, v in filtered.items()
                       if not any(excl in k for excl in exclude)}

        return filtered

    # Apply filters to all data dictionaries
    filtered_images = filter_dict(data.images)
    filtered_apd = filter_dict(data.apd_traces)
    filtered_monitor = filter_dict(data.monitor_traces)
    filtered_xy = filter_dict(data.xy_coords)
    filtered_z = filter_dict(data.z_scans)

    print(f"Found {len(filtered_images)} files matching '{pattern}'")

    return ConfocalData(
        images=filtered_images,
        apd_traces=filtered_apd,
        monitor_traces=filtered_monitor,
        xy_coords=filtered_xy,
        z_scans=filtered_z
    )


# ============================================================================
# IMAGE ANALYSIS - PSF FITTING
# ============================================================================

def gaussian_2d(xy: Tuple[npt.NDArray, npt.NDArray],
                amplitude: float,
                x0: float,
                y0: float,
                sigma_x: float,
                sigma_y: float,
                offset: float) -> npt.NDArray:
    """2D Gaussian function for fitting point spread functions.

    Args:
        xy: Tuple of (x, y) coordinate meshgrids
        amplitude: Peak intensity above offset
        x0: Center x-coordinate
        y0: Center y-coordinate
        sigma_x: Standard deviation in x-direction
        sigma_y: Standard deviation in y-direction
        offset: Background level

    Returns:
        Flattened 1D array of Gaussian values
    """
    x, y = xy
    exponent = -((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))
    return (amplitude * np.exp(exponent) + offset).ravel()


def fit_psf_gaussian(image: npt.NDArray) -> PSFParameters:
    """Fit a 2D Gaussian to an image to extract PSF parameters.

    Args:
        image: 2D array representing confocal scan image

    Returns:
        PSFParameters with fitted values

    Note:
        If fit fails, returns parameters based on image max/min and center position
    """
    height, width = image.shape
    y, x = np.mgrid[:height, :width]

    # Initial parameter guesses
    initial_guess = [
        image.max(),                    # amplitude
        width // 2,                      # x center
        height // 2,                     # y center
        DEFAULT_SIGMA_GUESS,            # sigma_x
        DEFAULT_SIGMA_GUESS,            # sigma_y
        image.min()                      # offset
    ]

    try:
        fitted_params, _ = curve_fit(
            gaussian_2d,
            (x, y),
            image.ravel(),
            p0=initial_guess
        )
    except (RuntimeError, ValueError) as e:
        # Fit failed - return initial guess as fallback
        print(f"Warning: PSF fit failed ({e.__class__.__name__}), using initial guess")
        fitted_params = initial_guess

    return PSFParameters(*fitted_params)


# ============================================================================
# APD TRACE ANALYSIS
# ============================================================================

def calculate_trace_statistics(traces: npt.NDArray,
                               center_y: int,
                               center_x: int) -> TraceStatistics:
    """Calculate SNR statistics from APD traces in a center region.

    Extracts a 3x3 pixel region around the specified center, then computes
    mean, std, and SNR for each of the 9 traces individually. Returns the
    average across all 9 traces.

    Args:
        traces: 3D array of shape (height, width, trace_length)
        center_y: Y-coordinate of center pixel
        center_x: X-coordinate of center pixel

    Returns:
        TraceStatistics with averaged SNR, mean, and std from center region

    Note:
        Skips first TRACE_SKIP_INITIAL_SAMPLES samples to avoid startup transients
    """
    # Extract 3x3 region around center
    half_size = CENTER_REGION_SIZE // 2
    region = traces[center_y - half_size : center_y + half_size + 1,
                   center_x - half_size : center_x + half_size + 1]

    # Calculate statistics for each individual trace in the region
    snrs = []
    means = []
    stds = []

    for i in range(region.shape[0]):
        for j in range(region.shape[1]):
            # Skip initial samples to avoid startup transients
            trace = region[i, j, TRACE_SKIP_INITIAL_SAMPLES:]

            mean = np.mean(trace)
            std = np.std(trace)
            snr = mean / std if std > 0 else 0

            snrs.append(snr)
            means.append(mean)
            stds.append(std)

    # Average across all traces in the region
    return TraceStatistics(
        snr=float(np.mean(snrs)),
        mean_counts=float(np.mean(means)),
        std_counts=float(np.mean(stds))
    )


def get_center_position(image_shape: Tuple[int, int],
                       psf: PSFParameters) -> Tuple[int, int]:
    """Determine best center position for trace analysis.

    Uses Gaussian fit center if it's within the middle 3x3 region,
    otherwise falls back to image center.

    Args:
        image_shape: (height, width) of the image
        psf: Fitted PSF parameters

    Returns:
        (center_y, center_x) in pixel coordinates
    """
    height, width = image_shape
    image_center_y = height // 2
    image_center_x = width // 2

    # Extract Gaussian center and round to nearest pixel
    gaussian_center_y = int(round(psf.y_center))
    gaussian_center_x = int(round(psf.x_center))

    # Check if Gaussian center is within middle 3x3 region
    if (abs(gaussian_center_y - image_center_y) <= 1 and
        abs(gaussian_center_x - image_center_x) <= 1):
        return gaussian_center_y, gaussian_center_x
    else:
        return image_center_y, image_center_x


# ============================================================================
# COMPLETE ANALYSIS PIPELINE
# ============================================================================

def analyze_confocal(data: ConfocalData) -> Dict[str, ConfocalAnalysisResult]:
    """Analyze confocal data: fit PSFs and calculate APD trace statistics.

    For each measurement:
    1. Fits 2D Gaussian to image to extract PSF parameters
    2. If APD traces available, calculates SNR from 3x3 center region

    Args:
        data: ConfocalData to analyze

    Returns:
        Dictionary mapping measurement labels to ConfocalAnalysisResult

    Example:
        >>> results = analyze_confocal(filtered_data)
        >>> for label, result in results.items():
        ...     print(f"{label}: SNR = {result.trace_stats.snr:.1f}")
    """
    results = {}

    for label, image in data.images.items():
        # Skip measurements without image data
        if image is None:
            continue

        # Fit 2D Gaussian to extract PSF parameters
        psf = fit_psf_gaussian(image)
        max_intensity = float(image.max())

        # Calculate APD trace statistics if traces are available
        trace_stats = None
        if label in data.apd_traces and data.apd_traces[label] is not None:
            traces = data.apd_traces[label]

            # Determine best center position for analysis
            center_y, center_x = get_center_position(traces.shape[:2], psf)

            # Calculate statistics from center region
            trace_stats = calculate_trace_statistics(traces, center_y, center_x)

        # Store complete analysis result
        results[label] = ConfocalAnalysisResult(
            psf=psf,
            max_intensity=max_intensity,
            trace_stats=trace_stats
        )

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage demonstrating the refactored API

    # Define data paths
    data_path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.09.02 - Sample 13 before after break"
    apd_path = r"Data\APD\2025.09.02 - Sample 13 PT high power"

    # Load APD data
    apd_data, monitor_data, apd_params = apd_load_main(apd_path)
    apd_data_box1, monitor_data_box1, apd_params_box1 = filter_apd(
        apd_data, monitor_data, apd_params, "*box1*"
    )
    apd_data_box4, monitor_data_box4, apd_params_box4 = filter_apd(
        apd_data, monitor_data, apd_params, "*box4*"
    )

    # Load confocal data with caching
    confocal_data = get_all_confocal(data_path)

    # Filter for specific measurements
    confocal_before = filter_confocal(confocal_data, "box1*[!_after*]", exclude=["C2"])
    confocal_after = filter_confocal(confocal_data, "*box1*after*", exclude=["C2"])

    # Analyze both datasets
    results_before = analyze_confocal(confocal_before)
    results_after = analyze_confocal(confocal_after)

    # Print example results
    print("\nAnalysis Results (Before):")
    for label, result in list(results_before.items())[:3]:
        print(f"\n{label}:")
        print(f"  PSF center: ({result.psf.x_center:.1f}, {result.psf.y_center:.1f})")
        print(f"  PSF width: σx={result.psf.sigma_x:.2f}, σy={result.psf.sigma_y:.2f}")
        if result.trace_stats:
            print(f"  SNR: {result.trace_stats.snr:.1f}")
            print(f"  Mean counts: {result.trace_stats.mean_counts:.1f}")
