"""
Configuration file for nanostructure-analysis package.

All paths, constants, and configuration parameters are centralized here.
"""

import os

# ============================================================================
# DIRECTORY PATHS
# ============================================================================

# Base directories (relative to project root)
CACHE_DIR = "cache"
DATA_DIR = "Data"
PLOTS_DIR = "plots"

# Data subdirectories
SPECTRA_DIR = os.path.join(DATA_DIR, "Spectra")
CONFOCAL_DIR = os.path.join(DATA_DIR, "Confocal")
SEM_DIR = os.path.join(DATA_DIR, "SEM")
APD_RAW_DIR = os.path.join(DATA_DIR, "APD")
APD_DOWNSAMPLED_DIR = os.path.join(DATA_DIR, "apd_downsampled")

# Legacy alias for backward compatibility
APD_SAVE_DIR = APD_DOWNSAMPLED_DIR


# ============================================================================
# APD (AVALANCHE PHOTODIODE) PARAMETERS
# ============================================================================

# Downsampling configuration
DEFAULT_DOWNSAMPLE_FACTOR = 100  # Factor by which to downsample raw APD traces

# Power calibration
DEFAULT_POWER_FACTOR = 50.0  # mW/V - Default power calibration factor

# Trace analysis
TRACE_SKIP_INITIAL_SAMPLES = 500  # Skip first N samples to avoid startup transients
# Seconds to discard at start of APD plots (for baseline drift)
APD_PLOT_DISCARD_SECONDS = 0.4
# ALS smoothing parameters (mirrors spectra baseline smoothing style)
APD_ALS_LAMBDA = 1e6
APD_ALS_ASYMMETRY = 0.5
APD_ALS_NITER = 10


# ============================================================================
# CONFOCAL MICROSCOPY PARAMETERS
# ============================================================================

# PSF analysis
DEFAULT_SIGMA_GUESS = 5  # Initial guess for PSF width in pixels (2D Gaussian fit)
CENTER_REGION_SIZE = 3  # Size of NxN region around PSF center for SNR calculation


# ============================================================================
# PATTERN MATCHING
# ============================================================================

# Grid position pattern for sample locations (e.g., A1, B2, C3, D6)
GRID_POSITION_PATTERN = r'[A-D][1-6]'


# ============================================================================
# PLOTTING DEFAULTS
# ============================================================================

# Figure dimensions
DEFAULT_FIGURE_SIZE = (12, 7)  # (width, height) in inches

# Font sizes (increased for presentation quality)
DEFAULT_TITLE_FONTSIZE = 26
DEFAULT_LABEL_FONTSIZE = 22
DEFAULT_TICK_FONTSIZE = 20
DEFAULT_LEGEND_FONTSIZE = 18

# Line and marker sizes
DEFAULT_LINE_WIDTH = 2.5
DEFAULT_MARKER_SIZE = 10

# Grid styling
DEFAULT_GRID_ALPHA = 0.3
DEFAULT_GRID_LINEWIDTH = 1.0

# Colors for grid positions
GRID_POSITION_COLORS = {
    'A': 'tab:blue',
    'B': 'tab:orange',
    'C': 'tab:green',
    'D': 'tab:red'
}

COLOR_ARRAY = ["#9E9E9E", \
                "#E795BB", \
                "#FFB973", \
                "#6599D4", \
                "#B3D669", \
                "#FFF949"]
# ============================================================================
# FILE NAMING CONVENTIONS
# ============================================================================

# Number of timestamp parts to strip from filenames (e.g., "20250101_143022_sample.dat" -> "sample")
TIMESTAMP_PARTS_TO_STRIP = 2


# ============================================================================
# NETWORK PATHS (for large raw data storage)
# ============================================================================

# Network drive location for large datasets
# Note: Update this path according to your network configuration
NETWORK_DATA_PATH = r"\\AMIPC045962\Cache (D)\daily_data"

# Network data subdirectories
NETWORK_CONFOCAL_PATH = os.path.join(NETWORK_DATA_PATH, "confocal_data")
NETWORK_SPECTRA_PATH = os.path.join(NETWORK_DATA_PATH, "spectra")
NETWORK_APD_PATH = os.path.join(NETWORK_DATA_PATH, "apd_traces")


# ============================================================================
# CACHE SETTINGS
# ============================================================================

# Enable/disable caching
USE_CACHE = True

# Cache file naming format
CACHE_PREFIX = {
    'apd': 'apd_',
    'confocal': 'confocal_',
    'spectra': 'spectra_',
}


# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

# Spectra analysis
SPECTRA_CUTOFF_WAVELENGTH = 950  # nm - Default maximum wavelength
SPECTRA_NORMALIZATION_WAVELENGTH = 740  # nm - Default normalization anchor point
SPECTRA_SAVGOL_WINDOW = 31  # Default Savitzky-Golay filter window size

# Cosmic ray filtering
COSMIC_RAY_SIGMA_THRESHOLD = 2.0  # Standard deviations for outlier detection


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_cache_path(data_type: str, identifier: str) -> str:
    """
    Generate a cache file path.

    Args:
        data_type: Type of data ('apd', 'confocal', 'spectra')
        identifier: Unique identifier (typically sanitized path)

    Returns:
        Full path to cache file
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    prefix = CACHE_PREFIX.get(data_type, '')
    sanitized = identifier.replace("/", "_").replace("\\", "_").replace(":", "").replace(" ", "_")
    return os.path.join(CACHE_DIR, f"{prefix}{sanitized}.pkl")


def sanitize_path_for_cache(path: str) -> str:
    """Convert a path to a safe cache filename component."""
    return path.replace("/", "_").replace("\\", "_").replace(":", "").replace(" ", "_")


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration and create necessary directories."""
    # Create base directories if they don't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Note: Don't auto-create Data directories as they should be managed by user

    return True


# Auto-validate on import
validate_config()
