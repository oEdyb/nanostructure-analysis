import numpy as np
import matplotlib.pyplot as plt
import re
from .confocal_functions import ConfocalData, confocal_main, filter_confocal
from .apd_functions import APDData
from . import config


# ============================================================================
# CONSTANTS (imported from config)
# ============================================================================

DEFAULT_FIGURE_SIZE = config.DEFAULT_FIGURE_SIZE
DEFAULT_TITLE_FONTSIZE = config.DEFAULT_TITLE_FONTSIZE
DEFAULT_LABEL_FONTSIZE = config.DEFAULT_LABEL_FONTSIZE
DEFAULT_TICK_FONTSIZE = config.DEFAULT_TICK_FONTSIZE
GRID_POSITION_PATTERN = config.GRID_POSITION_PATTERN


def calculate_center_snr(apd_traces: dict, skip_samples: int = 500) -> dict:
    """Calculate SNR from 3x3 region around center of APD traces.

    Method (for paper):
    1. Extract 3x3 pixel region centered at (height//2, width//2)
    2. For each of the 9 pixels, extract time trace after skipping initial samples
    3. Calculate SNR = mean/std for each individual trace
    4. Average all 9 SNR values

    Args:
        apd_traces: Dict of {main_name: {index: 3D array (height, width, time)}}
        skip_samples: Number of initial samples to skip (default: 500)

    Returns:
        Dict of {main_name: {index: snr_value}}
    """
    snr_results = {}

    for main_name, indices in apd_traces.items():
        snr_results[main_name] = {}

        for index, trace_array in indices.items():
            if trace_array is None:
                continue

            # Get dimensions and compute center of mass from top 70% pixels
            h, w, t = trace_array.shape

            # Average over time to get 2D intensity map
            intensity_map = np.mean(trace_array[:, :, skip_samples:], axis=2)

            # Threshold: keep only top 70% of pixel values
            threshold = np.percentile(intensity_map, 70)
            mask = intensity_map >= threshold

            # Compute center of mass
            y_coords, x_coords = np.where(mask)
            weights = intensity_map[mask]
            center_y = int(np.average(y_coords, weights=weights))
            center_x = int(np.average(x_coords, weights=weights))

            # Extract 3x3 region around center
            region_3x3 = trace_array[center_y - 1 : center_y + 2,
                                     center_x - 1 : center_x + 2,
                                     skip_samples:]

            # Calculate SNR for each of the 9 pixels
            snr_list = []
            for i in range(3):
                for j in range(3):
                    trace = region_3x3[i, j, :]
                    mean = np.mean(trace)
                    std = np.std(trace)
                    snr = mean / std if std > 0 else 0
                    snr_list.append(snr)

            # Average SNR across all 9 pixels
            snr_results[main_name][index] = float(np.mean(snr_list))

    return snr_results


def extract_position_id(name):
    """Extract position like 'A1', 'B3' from filename."""
    match = re.search(r'[A-D][1-6]', name)
    return match.group(0) if match else None


def plot_snr_scatter(data, labels=None, colors=None, markers=None, matching_flag=False, title="APD Trace SNR (3x3 Center Region)"):
    """Plot SNR as a scatter plot for all measurements.

    Args:
        data: ConfocalData object or list of ConfocalData objects
        labels: List of legend labels for each dataset (optional)
        colors: List of colors for each dataset (optional)
        markers: List of marker styles for each dataset (optional)
        matching_flag: If True, match entries by position ID (e.g., A1, B3)
        title: Plot title
    """
    # Convert single data to list
    if isinstance(data, ConfocalData):
        data_list = [data]
    else:
        data_list = data

    # Set defaults
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(data_list))]
    if colors is None:
        colors = plt.cm.tab10(range(len(data_list)))
    if markers is None:
        markers = ['o'] * len(data_list)

    # Create plot
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)

    if matching_flag:
        # Organize by position ID
        all_positions = set()
        dataset_snr_by_pos = []

        for dataset in data_list:
            snr_dict = calculate_center_snr(dataset.apd_traces)
            pos_snr = {}
            for main_name, indices in snr_dict.items():
                pos_id = extract_position_id(main_name)
                if pos_id:
                    for index, snr in indices.items():
                        pos_snr.setdefault(pos_id, []).append(snr)
                    all_positions.add(pos_id)
            dataset_snr_by_pos.append(pos_snr)

        # Sort positions
        sorted_positions = sorted(all_positions)

        # Plot each dataset
        for i, (pos_snr, label, color, marker) in enumerate(zip(dataset_snr_by_pos, labels, colors, markers)):
            for x, pos in enumerate(sorted_positions):
                if pos in pos_snr:
                    for snr in pos_snr[pos]:
                        ax.scatter(x, snr, s=150, alpha=0.7, color=color, marker=marker, label=label if x == 0 else "", linewidths=2)

        ax.set_xticks(range(len(sorted_positions)))
        ax.set_xticklabels(sorted_positions)

    else:
        x_offset = 0
        for dataset, label, color, marker in zip(data_list, labels, colors, markers):
            snr_dict = calculate_center_snr(dataset.apd_traces)
            snr_values = []
            for main_name, indices in snr_dict.items():
                for index, snr in indices.items():
                    snr_values.append(snr)

            x_positions = range(x_offset, x_offset + len(snr_values))
            ax.scatter(x_positions, snr_values, s=150, alpha=0.7, color=color, marker=marker, label=label, linewidths=2)
            x_offset += len(snr_values)

    ax.set_ylabel('SNR', fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_xlabel('Position' if matching_flag else 'Measurement Index', fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.tick_params(labelsize=DEFAULT_TICK_FONTSIZE, width=1.5, length=6)
    ax.legend(fontsize=config.DEFAULT_LEGEND_FONTSIZE, frameon=False)
    ax.grid(alpha=config.DEFAULT_GRID_ALPHA, linewidth=config.DEFAULT_GRID_LINEWIDTH)

    plt.tight_layout()
    return fig, ax



def plot_snr_vs_time_by_power(confocal_data, apd_data: APDData, labels=None, colors=None, markers=None, power_threshold=45, title="SNR vs Time by Power"):
    """Plot SNR vs measurement time, grouped by power.

    Args:
        confocal_data: ConfocalData object or list of ConfocalData objects
        apd_data: APDData object (matched by position ID)
        labels: List of legend labels for each confocal dataset (optional)
        colors: List of colors for each dataset (optional)
        markers: List of marker styles for each dataset (optional)
        power_threshold: Power threshold in mW to separate groups (default: 45)
        title: Plot title
    """
    # Convert single data to list
    if isinstance(confocal_data, ConfocalData):
        data_list = [confocal_data]
    else:
        data_list = confocal_data

    # Set defaults - clean publication markers
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(data_list))]
    if colors is None:
        colors = ['C0', 'C1', 'C2', 'C3'][:len(data_list)]  # Matplotlib default color cycle
    if markers is None:
        markers = ['o', 's'][:len(data_list)]

    # Plot
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)

    for dataset, label, color, marker in zip(data_list, labels, colors, markers):
        # Calculate SNR
        snr_dict = calculate_center_snr(dataset.apd_traces)

        # Match by position ID and collect data
        low_power_data = []
        high_power_data = []

        for main_name, indices in snr_dict.items():
            pos_id = extract_position_id(main_name)
            if not pos_id:
                continue

            # Find matching APD measurement
            for apd_label, params in apd_data.params.items():
                if pos_id in apd_label:
                    power = params['power']
                    time = params.get('Duration (s)', 1.0)

                    for index, snr in indices.items():
                        if power < power_threshold:
                            low_power_data.append((time, snr))
                        else:
                            high_power_data.append((time, snr))
                    break

        # Sort by time and plot with lines
        if low_power_data:
            low_power_data.sort()
            times, snrs = zip(*low_power_data)
            ax.plot(times, snrs, marker=marker, markersize=config.DEFAULT_MARKER_SIZE,
                   linewidth=config.DEFAULT_LINE_WIDTH,
                   color=color, linestyle='-', label=label, alpha=0.8)

        if high_power_data:
            high_power_data.sort()
            times, snrs = zip(*high_power_data)
            ax.plot(times, snrs, marker=marker, markersize=config.DEFAULT_MARKER_SIZE,
                   linewidth=config.DEFAULT_LINE_WIDTH,
                   color=color, linestyle='--', alpha=0.8)

    ax.set_xscale('log')
    ax.set_xlabel('Exposure time (s)', fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel('SNR', fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.tick_params(labelsize=DEFAULT_TICK_FONTSIZE, width=1.5, length=6)
    ax.legend(fontsize=config.DEFAULT_LEGEND_FONTSIZE, frameon=False)

    # Add text annotation for power levels
    ax.text(0.02, 0.02, 'Solid: 30 mW, Dashed: 60 mW',
            transform=ax.transAxes, fontsize=16, ha='left', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    ax.grid(True, alpha=config.DEFAULT_GRID_ALPHA, linestyle=':', linewidth=config.DEFAULT_GRID_LINEWIDTH)

    plt.tight_layout()
    return fig, ax


def plot_transmission_per_power_vs_time_by_power(confocal_data, apd_data: APDData, labels=None, colors=None, markers=None, power_threshold=45, title="Transmission per Power vs Time"):
    """Plot median transmission divided by power vs measurement time, grouped by power.

    Similar to plot_snr_vs_time_by_power but plots transmission/power instead of SNR.
    Uses confocal positions to match with APD measurements, then calculates median
    transmission from APD traces and divides by power.

    Args:
        confocal_data: ConfocalData object or list of ConfocalData objects
        apd_data: APDData object (matched by position ID)
        labels: List of legend labels for each confocal dataset (optional)
        colors: List of colors for each dataset (optional)
        markers: List of marker styles for each dataset (optional)
        power_threshold: Power threshold in mW to separate groups (default: 45)
        title: Plot title
    """
    # Convert single data to list
    if isinstance(confocal_data, ConfocalData):
        data_list = [confocal_data]
    else:
        data_list = confocal_data

    # Set defaults
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(data_list))]
    if colors is None:
        colors = ['C0', 'C1', 'C2', 'C3'][:len(data_list)]
    if markers is None:
        markers = ['o', 's'][:len(data_list)]

    # Plot
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)

    for dataset, label, color, marker in zip(data_list, labels, colors, markers):
        # Match by position ID and collect data
        low_power_data = []
        high_power_data = []

        # Get confocal positions
        snr_dict = calculate_center_snr(dataset.apd_traces)

        for main_name, indices in snr_dict.items():
            pos_id = extract_position_id(main_name)
            if not pos_id:
                continue

            # Find matching APD measurement
            for apd_label, params in apd_data.params.items():
                if pos_id in apd_label:
                    power = params['power']
                    time = params.get('Duration (s)', 1.0)

                    # Get median transmission from APD trace
                    transmission_trace = apd_data.transmission[apd_label]
                    median_transmission = np.median(transmission_trace)

                    # Calculate transmission per power (convert V to mV)
                    transmission_per_power = (median_transmission * 1000) / power

                    # Group by power
                    if power < power_threshold:
                        low_power_data.append((time, transmission_per_power))
                    else:
                        high_power_data.append((time, transmission_per_power))
                    break

        # Sort by time and plot with lines
        if low_power_data:
            low_power_data.sort()
            times, trans_per_power = zip(*low_power_data)
            ax.plot(times, trans_per_power, marker=marker, markersize=config.DEFAULT_MARKER_SIZE,
                   linewidth=config.DEFAULT_LINE_WIDTH,
                   color=color, linestyle='-', label=label, alpha=0.8)

        if high_power_data:
            high_power_data.sort()
            times, trans_per_power = zip(*high_power_data)
            ax.plot(times, trans_per_power, marker=marker, markersize=config.DEFAULT_MARKER_SIZE,
                   linewidth=config.DEFAULT_LINE_WIDTH,
                   color=color, linestyle='--', alpha=0.8)

    ax.set_xscale('log')
    ax.set_xlabel('Exposure time (s)', fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel('Transmission / Power (mV/mW)', fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.tick_params(labelsize=DEFAULT_TICK_FONTSIZE, width=1.5, length=6)
    ax.legend(fontsize=config.DEFAULT_LEGEND_FONTSIZE, frameon=False)

    # Add text annotation for power levels
    low_power_text = f"{power_threshold - 15:.0f} mW"
    high_power_text = f"{power_threshold + 15:.0f} mW"
    ax.text(0.02, 0.02, f'Solid: {low_power_text}, Dashed: {high_power_text}',
            transform=ax.transAxes, fontsize=16, ha='left', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    ax.grid(True, alpha=config.DEFAULT_GRID_ALPHA, linestyle=':', linewidth=config.DEFAULT_GRID_LINEWIDTH)

    plt.tight_layout()
    return fig, ax





if __name__ == "__main__":
    path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.05.13-14 - Sample 7 Box 1 Illumination"
    data = confocal_main(path)
    data_after = filter_confocal(data, pattern=["perp", "after"], exclude=[])
    data_before = filter_confocal(data, pattern=["perp"], exclude=["after"])
    print(data)
    # Plot SNR for all measurements
    plot_snr_scatter([data_before, data_after],
                     labels=["Before", "After"],
                     colors=["blue", "orange"],
                     markers=["o", "s"],
                     title="SNR Before and After",
                     matching_flag=True)
    plt.show()
