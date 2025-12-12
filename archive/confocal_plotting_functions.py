"""
Plotting functions for confocal microscopy data visualization.

This module provides functions for:
- Plotting confocal images
- Comparing before/after measurements
- Analyzing SNR and intensity changes vs power
- Visualizing PSF parameters
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter
from scipy import stats
import re
from typing import Dict, Optional, Union

# Import the new data structures
from .confocal_functions import ConfocalData, ConfocalAnalysisResult
from .apd_functions import APDData
from . import config


# ============================================================================
# CONSTANTS (imported from config)
# ============================================================================

GRID_POSITION_PATTERN = config.GRID_POSITION_PATTERN
DEFAULT_FIGURE_SIZE = config.DEFAULT_FIGURE_SIZE
DEFAULT_TITLE_FONTSIZE = config.DEFAULT_TITLE_FONTSIZE
DEFAULT_LABEL_FONTSIZE = config.DEFAULT_LABEL_FONTSIZE
DEFAULT_TICK_FONTSIZE = config.DEFAULT_TICK_FONTSIZE


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_grid_positions(keys: list, pattern: str = GRID_POSITION_PATTERN) -> Dict[str, str]:
    """Extract grid positions from measurement keys.

    Args:
        keys: List of measurement keys (e.g., ["box1_A3_10ms", "box1_B2_20ms"])
        pattern: Regex pattern for grid position (default: [A-D][1-6])

    Returns:
        Dictionary mapping grid position to full key {position: key}
        Example: {"A3": "box1_A3_10ms", "B2": "box1_B2_20ms"}
    """
    positions = {}
    for key in keys:
        match = re.search(pattern, key)
        if match:
            positions[match.group()] = key
    return positions


def get_apd_params_dict(apd_params: Union[Dict, APDData]) -> Dict:
    """Convert APD params to dictionary format.

    Args:
        apd_params: Either a dict or APDData object

    Returns:
        Dictionary of parameters
    """
    if isinstance(apd_params, APDData):
        return apd_params.params
    return apd_params


# ============================================================================
# BASIC PLOTTING
# ============================================================================

def plot_confocal(confocal_data: ConfocalData) -> None:
    """Plot all confocal images in a dataset.

    Creates a separate figure for each measurement showing the 2D confocal scan.

    Args:
        confocal_data: ConfocalData containing images to plot
    """
    for key, data in confocal_data.images.items():
        if data is None:
            continue

        plt.figure(figsize=(8, 6))
        plt.imshow(data)
        plt.colorbar()
        plt.title(key, fontsize=DEFAULT_TITLE_FONTSIZE, fontweight='bold')
        plt.xlabel('X (px)', fontsize=DEFAULT_LABEL_FONTSIZE, fontweight='bold')
        plt.ylabel('Y (px)', fontsize=DEFAULT_LABEL_FONTSIZE, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=DEFAULT_TICK_FONTSIZE)
        plt.tight_layout()
        plt.show()
        plt.close()


# ============================================================================
# BEFORE/AFTER COMPARISON PLOTS
# ============================================================================

def plot_confocal_image_comparison(confocal_before: ConfocalData,
                                   confocal_after: ConfocalData) -> None:
    """Plot side-by-side comparison of before/after confocal images.

    Matches measurements by grid position and displays them with shared colorbar scales.

    Args:
        confocal_before: ConfocalData from before measurements
        confocal_after: ConfocalData from after measurements
    """
    # Find common grid positions
    before_locs = extract_grid_positions(confocal_before.images.keys())
    after_locs = extract_grid_positions(confocal_after.images.keys())

    # Plot only matching locations
    for loc in sorted(set(before_locs.keys()) & set(after_locs.keys())):
        plt.figure(figsize=(12, 5))

        # Get data for both images
        before_data = confocal_before.images[before_locs[loc]]
        after_data = confocal_after.images[after_locs[loc]]

        # Calculate shared min/max for consistent colorbar
        vmin = min(before_data.min(), after_data.min())
        vmax = max(before_data.max(), after_data.max())

        # Before image
        plt.subplot(1, 2, 1)
        plt.imshow(before_data, vmin=vmin, vmax=vmax)
        plt.title(f'Before - {loc}', fontsize=DEFAULT_TITLE_FONTSIZE, fontweight='bold')
        plt.xlabel('X (px)', fontsize=DEFAULT_LABEL_FONTSIZE, fontweight='bold')
        plt.ylabel('Y (px)', fontsize=DEFAULT_LABEL_FONTSIZE, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=DEFAULT_TICK_FONTSIZE)
        plt.tight_layout()

        # After image
        plt.subplot(1, 2, 2)
        plt.imshow(after_data, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(f'After - {loc}', fontsize=DEFAULT_TITLE_FONTSIZE, fontweight='bold')
        plt.xlabel('X (px)', fontsize=DEFAULT_LABEL_FONTSIZE, fontweight='bold')
        plt.ylabel('Y (px)', fontsize=DEFAULT_LABEL_FONTSIZE, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=DEFAULT_TICK_FONTSIZE)
        plt.tight_layout()


def plot_confocal_comparison(confocal_before: ConfocalData,
                            confocal_after: ConfocalData,
                            results_before: Optional[Dict[str, ConfocalAnalysisResult]] = None,
                            results_after: Optional[Dict[str, ConfocalAnalysisResult]] = None) -> None:
    """Plot before/after images with cross-sections.

    Shows images in top row and X cross-section comparison in bottom row.

    Args:
        confocal_before: ConfocalData from before measurements
        confocal_after: ConfocalData from after measurements
        results_before: Analysis results for before (optional, for PSF centers)
        results_after: Analysis results for after (optional, for PSF centers)
    """
    # Find common grid positions
    before_locs = extract_grid_positions(confocal_before.images.keys())
    after_locs = extract_grid_positions(confocal_after.images.keys())

    # Plot only matching locations
    for loc in sorted(set(before_locs.keys()) & set(after_locs.keys())):
        plt.figure(figsize=(10, 8))

        # Get data
        before_data = confocal_before.images[before_locs[loc]]
        after_data = confocal_after.images[after_locs[loc]]

        # Calculate shared min/max for consistent colorbar
        vmin = min(before_data.min(), after_data.min())
        vmax = max(before_data.max(), after_data.max())

        # Extract center positions from PSF fit results
        if results_before is not None and before_locs[loc] in results_before:
            center_y_before = int(results_before[before_locs[loc]].psf.y_center)
        else:
            center_y_before = before_data.shape[0] // 2

        if results_after is not None and after_locs[loc] in results_after:
            center_y_after = int(results_after[after_locs[loc]].psf.y_center)
        else:
            center_y_after = after_data.shape[0] // 2

        # Before image (top left)
        plt.subplot(2, 2, 1)
        plt.imshow(before_data, vmin=vmin, vmax=vmax)
        plt.axhline(center_y_before, color='red', linestyle='--')
        plt.title(f'Before - {loc}', fontsize=18, fontweight='bold')
        plt.xlabel('X (px)', fontsize=14, fontweight='bold')
        plt.ylabel('Y (px)', fontsize=14, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=12)

        # After image (top right)
        plt.subplot(2, 2, 2)
        plt.imshow(after_data, vmin=vmin, vmax=vmax)
        plt.axhline(center_y_after, color='red', linestyle='--')
        plt.colorbar()
        plt.title(f'After - {loc}', fontsize=18, fontweight='bold')
        plt.xlabel('X (px)', fontsize=14, fontweight='bold')
        plt.ylabel('Y (px)', fontsize=14, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=12)

        # X cross sections (bottom, spanning both columns)
        plt.subplot(2, 1, 2)
        plt.plot(before_data[center_y_before, :], label='Before', linewidth=2)
        plt.plot(after_data[center_y_after, :], label='After', linewidth=2)
        plt.xlabel('X (px)', fontsize=14, fontweight='bold')
        plt.ylabel('Intensity', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.title('X Cross Section', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()


# ============================================================================
# INTENSITY CHANGE ANALYSIS
# ============================================================================

def plot_confocal_scatters(confocal_before: ConfocalData,
                          confocal_after: ConfocalData,
                          results_before: Dict[str, ConfocalAnalysisResult],
                          results_after: Dict[str, ConfocalAnalysisResult],
                          apd_params: Union[Dict, APDData],
                          new_fig: bool = True,
                          marker: str = 'o',
                          label: Optional[str] = None) -> None:
    """Plot percent change in max intensity vs power.

    Calculates (after - before) / before * 100 for each matched measurement.

    Args:
        confocal_before: ConfocalData from before measurements
        confocal_after: ConfocalData from after measurements
        results_before: Analysis results for before
        results_after: Analysis results for after
        apd_params: APD parameters (dict or APDData) containing power values
        new_fig: Create new figure if True, add to existing if False
        marker: Matplotlib marker style
        label: Legend label for this dataset
    """
    # Convert APD params to dict if needed
    apd_params_dict = get_apd_params_dict(apd_params)

    # Find common grid positions
    before_locs = extract_grid_positions(confocal_before.images.keys())
    after_locs = extract_grid_positions(confocal_after.images.keys())

    # Find matching patterns in APD params and extract power values
    apd_locs = {}
    for key in apd_params_dict.keys():
        match = re.search(GRID_POSITION_PATTERN, key)
        if match:
            apd_locs[match.group()] = apd_params_dict[key]['power']

    # Get locations that exist in all three datasets
    locations = sorted(set(before_locs.keys()) & set(after_locs.keys()) & set(apd_locs.keys()))

    # Extract max values and calculate percent change
    before_max = [results_before[before_locs[loc]].max_intensity for loc in locations]
    after_max = [results_after[after_locs[loc]].max_intensity for loc in locations]
    percent_change = [(after - before) / before * 100 for before, after in zip(before_max, after_max)]
    powers = [apd_locs[loc] for loc in locations]

    # Create new figure only if requested
    if new_fig:
        plt.figure(figsize=DEFAULT_FIGURE_SIZE)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='No change')

    plt.scatter(powers, percent_change, s=100, alpha=0.7, marker=marker,
               edgecolors='black', linewidths=1, label=label)

    # Set labels and formatting - only on new figure
    if new_fig:
        plt.xlabel('Power (mW)', fontsize=14, fontweight='bold')
        plt.ylabel('Max Value Change (%)', fontsize=14, fontweight='bold')
        plt.title('Max Value Change vs Power', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()

    plt.legend(fontsize=12)


# ============================================================================
# SNR ANALYSIS PLOTS
# ============================================================================

def plot_confocal_snr(confocal_data: ConfocalData,
                     results_dict: Dict[str, ConfocalAnalysisResult],
                     apd_params: Union[Dict, APDData],
                     new_fig: bool = True,
                     marker: str = 'o',
                     label: Optional[str] = None) -> None:
    """Plot SNR vs power for confocal measurements.

    Args:
        confocal_data: ConfocalData to plot
        results_dict: Analysis results containing SNR values
        apd_params: APD parameters (dict or APDData) containing power values
        new_fig: Create new figure if True, add to existing if False
        marker: Matplotlib marker style
        label: Legend label for this dataset
    """
    # Convert APD params to dict if needed
    apd_params_dict = get_apd_params_dict(apd_params)

    # Find common grid positions
    confocal_locs = extract_grid_positions(confocal_data.images.keys())

    # Find matching patterns in APD params and extract power values
    apd_locs = {}
    for key in apd_params_dict.keys():
        match = re.search(GRID_POSITION_PATTERN, key)
        if match:
            apd_locs[match.group()] = apd_params_dict[key]['power']

    # Get locations that exist in both datasets
    locations = sorted(set(confocal_locs.keys()) & set(apd_locs.keys()))

    # Extract SNR values for matching locations
    snr_values = []
    powers = []
    for loc in locations:
        confocal_key = confocal_locs[loc]
        if confocal_key in results_dict:
            result = results_dict[confocal_key]
            if result.trace_stats is not None:
                snr_values.append(result.trace_stats.snr)
                powers.append(apd_locs[loc])

    # Create new figure only if requested
    if new_fig:
        plt.figure(figsize=DEFAULT_FIGURE_SIZE)

    plt.scatter(powers, snr_values, s=100, alpha=0.7, marker=marker,
               edgecolors='black', linewidths=1, label=label)

    # Set labels and formatting - only on new figure
    if new_fig:
        plt.xlabel('Power (mW)', fontsize=14, fontweight='bold')
        plt.ylabel('SNR (3x3 center)', fontsize=14, fontweight='bold')
        plt.title('SNR vs Power', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()

    plt.legend(fontsize=12)


def plot_snr_before_after(confocal_before: ConfocalData,
                         confocal_after: ConfocalData,
                         results_before: Dict[str, ConfocalAnalysisResult],
                         results_after: Dict[str, ConfocalAnalysisResult],
                         new_fig: bool = True,
                         marker: str = 'o',
                         label: Optional[str] = None) -> None:
    """Plot SNR before vs after on equal axes.

    Diagonal line represents no change. Points below line indicate SNR decrease.

    Args:
        confocal_before: ConfocalData from before measurements
        confocal_after: ConfocalData from after measurements
        results_before: Analysis results for before
        results_after: Analysis results for after
        new_fig: Create new figure if True, add to existing if False
        marker: Matplotlib marker style
        label: Legend label for this dataset
    """
    # Find common grid positions
    before_locs = extract_grid_positions(confocal_before.images.keys())
    after_locs = extract_grid_positions(confocal_after.images.keys())

    # Get locations that exist in both datasets
    locations = sorted(set(before_locs.keys()) & set(after_locs.keys()))

    # Extract SNR values for matching locations
    snr_before = []
    snr_after = []
    for loc in locations:
        before_key = before_locs[loc]
        after_key = after_locs[loc]

        if before_key in results_before and after_key in results_after:
            before_result = results_before[before_key]
            after_result = results_after[after_key]

            if (before_result.trace_stats is not None and
                after_result.trace_stats is not None):
                snr_before.append(before_result.trace_stats.snr)
                snr_after.append(after_result.trace_stats.snr)

    # Create new figure only if requested
    if new_fig:
        plt.figure(figsize=(8, 8))

    plt.scatter(snr_before, snr_after, s=100, alpha=0.7, marker=marker,
               edgecolors='black', linewidths=1, label=label)

    # Set labels and formatting - only on new figure
    if new_fig:
        # Calculate max SNR for equal axis limits
        all_snr = snr_before + snr_after
        max_snr = max(all_snr) if all_snr else 1
        max_lim = max_snr * 1.1  # Add 10% padding

        # Add diagonal line for reference (no change)
        plt.plot([0, max_lim], [0, max_lim], 'k--', alpha=0.5, label='No change')

        plt.xlabel('SNR Before', fontsize=14, fontweight='bold')
        plt.ylabel('SNR After', fontsize=14, fontweight='bold')
        plt.title('SNR Before vs After', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)

        # Set equal axes starting from 0
        plt.xlim(0, max_lim)
        plt.ylim(0, max_lim)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()

    plt.legend(fontsize=12)


def plot_snr_vs_power(confocal_data: ConfocalData,
                     results_dict: Dict[str, ConfocalAnalysisResult],
                     apd_params: Union[Dict, APDData],
                     new_fig: bool = True,
                     marker: str = 'o',
                     label: Optional[str] = None) -> None:
    """Plot SNR vs power with trend line (excluding outliers).

    Fits a linear trend to data with z-score < 2, but plots all points.

    Args:
        confocal_data: ConfocalData to plot
        results_dict: Analysis results containing SNR values
        apd_params: APD parameters (dict or APDData) containing power values
        new_fig: Create new figure if True, add to existing if False
        marker: Matplotlib marker style
        label: Legend label for this dataset
    """
    # Convert APD params to dict if needed
    apd_params_dict = get_apd_params_dict(apd_params)

    # Find common grid positions
    confocal_locs = extract_grid_positions(confocal_data.images.keys())

    # Find matching patterns in APD params and extract power values
    apd_locs = {}
    for key in apd_params_dict.keys():
        match = re.search(GRID_POSITION_PATTERN, key)
        if match:
            apd_locs[match.group()] = apd_params_dict[key]['power']

    # Get locations that exist in both datasets
    locations = sorted(set(confocal_locs.keys()) & set(apd_locs.keys()))

    # Extract SNR values for matching locations
    snr_values = []
    powers = []
    for loc in locations:
        confocal_key = confocal_locs[loc]
        if confocal_key in results_dict:
            result = results_dict[confocal_key]
            if result.trace_stats is not None:
                snr_values.append(result.trace_stats.snr)
                powers.append(apd_locs[loc])

    # Create new figure only if requested
    if new_fig:
        plt.figure(figsize=DEFAULT_FIGURE_SIZE)

    scatter = plt.scatter(powers, snr_values, s=100, alpha=0.7, marker=marker,
                         edgecolors='black', linewidths=1, label=label)

    # Fit trend line excluding outliers from fit only (z-score > 2)
    if len(powers) > 2:
        z_scores = np.abs(stats.zscore(snr_values))
        fit_mask = z_scores < 2  # Use only non-outliers for fitting
        if np.sum(fit_mask) > 1:
            x_fit = np.array(powers)[fit_mask]
            y_fit = np.array(snr_values)[fit_mask]
            coeffs = np.polyfit(x_fit, y_fit, 1)
            x_line = np.linspace(min(powers), max(powers), 100)
            y_line = np.polyval(coeffs, x_line)
            plt.plot(x_line, y_line, '--', color=scatter.get_facecolors()[0],
                    alpha=0.8, linewidth=2)

    # Set labels and formatting - only on new figure
    if new_fig:
        plt.xlabel('Power (mW)', fontsize=14, fontweight='bold')
        plt.ylabel('SNR (3x3 center)', fontsize=14, fontweight='bold')
        plt.title('SNR vs Power', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()

    plt.legend(fontsize=12)


# ============================================================================
# GENERIC PARAMETER PLOTTING
# ============================================================================

def plot_confocal_parameter_scatter(confocal_data: ConfocalData,
                                   results_dict: Dict[str, ConfocalAnalysisResult],
                                   apd_params: Union[Dict, APDData],
                                   parameter: str = 'snr',
                                   new_fig: bool = True,
                                   marker: str = 'o',
                                   label: Optional[str] = None) -> None:
    """Generic scatter plot for any confocal parameter vs power.

    Args:
        confocal_data: ConfocalData to plot
        results_dict: Analysis results containing parameter values
        apd_params: APD parameters (dict or APDData) containing power values
        parameter: Parameter to plot. Options:
            - 'snr': Signal-to-noise ratio
            - 'mean_counts': Mean photon counts
            - 'sigma_x': PSF width in x
            - 'sigma_y': PSF width in y
            - 'amplitude': PSF amplitude
        new_fig: Create new figure if True, add to existing if False
        marker: Matplotlib marker style
        label: Legend label for this dataset

    Example:
        >>> plot_confocal_parameter_scatter(data, results, apd_params,
        ...                                 parameter='sigma_x', label='100nm disks')
    """
    # Convert APD params to dict if needed
    apd_params_dict = get_apd_params_dict(apd_params)

    # Find common grid positions
    confocal_locs = extract_grid_positions(confocal_data.images.keys())

    # Find matching patterns in APD params and extract power values
    apd_locs = {}
    for key in apd_params_dict.keys():
        match = re.search(GRID_POSITION_PATTERN, key)
        if match:
            apd_locs[match.group()] = apd_params_dict[key]['power']

    # Get locations that exist in both datasets
    locations = sorted(set(confocal_locs.keys()) & set(apd_locs.keys()))

    # Extract parameter values for matching locations
    param_values = []
    powers = []
    for loc in locations:
        confocal_key = confocal_locs[loc]
        if confocal_key not in results_dict:
            continue

        result = results_dict[confocal_key]

        # Extract value based on parameter name
        value = None
        if parameter == 'snr' and result.trace_stats is not None:
            value = result.trace_stats.snr
        elif parameter == 'mean_counts' and result.trace_stats is not None:
            value = result.trace_stats.mean_counts
        elif parameter == 'std_counts' and result.trace_stats is not None:
            value = result.trace_stats.std_counts
        elif hasattr(result.psf, parameter):
            value = getattr(result.psf, parameter)

        if value is not None:
            param_values.append(value)
            powers.append(apd_locs[loc])

    # Create new figure only if requested
    if new_fig:
        plt.figure(figsize=DEFAULT_FIGURE_SIZE)

    plt.scatter(powers, param_values, s=100, alpha=0.7, marker=marker,
               edgecolors='black', linewidths=1, label=label)

    # Set labels and formatting - only on new figure
    if new_fig:
        plt.xlabel('Power (mW)', fontsize=14, fontweight='bold')
        ylabel = parameter.replace('_', ' ').title()
        if 'psf' in parameter.lower() and 'size' in parameter.lower():
            ylabel += ' (nm)'
        plt.ylabel(ylabel, fontsize=14, fontweight='bold')
        plt.title(f'{parameter.replace("_", " ").title()} vs Power',
                 fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()

    plt.legend(fontsize=12)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage demonstrating the updated API
    from apd_functions import load_apd_data, filter_apd
    from confocal_functions import get_all_confocal, filter_confocal, analyze_confocal

    # Load APD data
    apd_path = "./Data/APD/2025.08.21 - Sample 13 Power Threshold"
    apd_data = load_apd_data(apd_path)
    apd_box1 = filter_apd(apd_data, "*box1*")

    print(f"Loaded {apd_data}")
    print(f"Filtered to {apd_box1}")
