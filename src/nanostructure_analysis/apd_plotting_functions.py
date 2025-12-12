"""
APD (Avalanche Photodiode) plotting functions.

All plotting functions accept APDData objects from apd_functions.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import re

from . import config


def _group_color_map(sorted_keys, fallback_colors):
    """Generate color mapping for grid position groups (A, B, C, D)."""
    preferred = config.GRID_POSITION_COLORS
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    color_by_group = {}
    for key in sorted_keys:
        match = re.search(config.GRID_POSITION_PATTERN, key)
        if not match:
            continue
        letter = match.group()[0]
        if letter in color_by_group:
            continue
        idx = len(color_by_group)
        fallback = default_cycle[idx % len(default_cycle)] if default_cycle else fallback_colors[idx % len(fallback_colors)]
        color_by_group[letter] = preferred.get(letter, fallback)
    return color_by_group


def _get_sort_key(key):
    """Sort key function for grid positions [A-D][1-6]."""
    match = re.search(config.GRID_POSITION_PATTERN, key)
    if match:
        letter_part = match.group()[0]
        number_part = int(match.group()[1])
        return (ord(letter_part), number_part)
    return (999, 999)


def plot_apd(
    apd_data,
    normalize=True,
    savgol=True,
    time_limit=None,
    new_fig=True,
    log_scale=False,
    group_colors=False,
    linestyle='-',
    legend_flag=True,
    discard_initial=None,
):
    """Plot APD transmission traces.

    Args:
        apd_data: APDData object from apd_functions
        normalize: If True, plot as percentage change from initial value
        savgol: If True, apply Savitzky-Golay smoothing filter
        time_limit: Maximum time to plot (seconds), None for full trace
        new_fig: Create new figure if True
        log_scale: Use logarithmic y-axis scale
        group_colors: Color by grid position group (A, B, C, D)
        linestyle: Line style ('-', '--', '-.', ':')
        legend_flag: Show legend if True
        discard_initial: Seconds to discard at start (default from config)
    """
    if new_fig:
        plt.figure(figsize=(16, 8))

    if discard_initial is None:
        discard_initial = config.APD_PLOT_DISCARD_SECONDS

    # Calculate power for each key and sort by power
    power_data = []
    for key in apd_data.transmission.keys():
        power_mw = apd_data.params[key].get('power', 0)
        power_data.append((power_mw, key))

    # Sort by power (ascending order)
    power_data.sort(key=lambda x: x[0])

    # Generate colors
    sorted_keys = [key for _, key in power_data]
    # Use COLOR_ARRAY from config, cycling if needed
    colors = [config.COLOR_ARRAY[i % len(config.COLOR_ARRAY)] for i in range(len(sorted_keys))]
    grouped_color_map = _group_color_map(sorted_keys, colors) if group_colors else {}

    # Plot in power order
    for i, (power_mw, key) in enumerate(power_data):
        data = apd_data.transmission[key]
        duration = apd_data.params[key]['Duration (s)']
        time_axis = np.linspace(0, duration, len(data))

        # Discard initial samples
        if discard_initial > 0:
            mask_discard = time_axis >= discard_initial
            data = data[mask_discard]
            time_axis = time_axis[mask_discard]

        # Apply time limit
        if time_limit is not None:
            mask = time_axis <= time_limit
            data = data[mask]
            time_axis = time_axis[mask]

        # Normalize
        if normalize:
            if len(data) > 0:
                normalized_apd = (data - data[0]) / data[0] * 100
            else:
                normalized_apd = data
        else:
            normalized_apd = data

        # Smooth
        if savgol and len(normalized_apd) > 51:
            normalized_apd = savgol_filter(normalized_apd, 51, 3)
            # Re-normalize to ensure first point is 0 after smoothing
            if normalize and len(normalized_apd) > 0:
                normalized_apd = normalized_apd - normalized_apd[0]

        # Determine color
        color = colors[i]
        if group_colors:
            match = re.search(r'([A-D])[1-6]', key)
            if match:
                color = grouped_color_map.get(match.group(1), color)

        plt.plot(
            time_axis,
            normalized_apd,
            label=f"{key} - {power_mw:.1f} mW",
            linestyle=linestyle,
            color=color,
            linewidth=config.DEFAULT_LINE_WIDTH,
        )

    # Add reference lines at �10% if normalized
    if normalize:
        plt.axhline(y=10, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        plt.axhline(y=-10, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    # Set logarithmic scale if requested
    if log_scale:
        plt.yscale('log')

    plt.xlabel('Time (s)', fontsize=config.DEFAULT_LABEL_FONTSIZE)
    if normalize:
        plt.ylabel('Transmission Change (%)', fontsize=config.DEFAULT_LABEL_FONTSIZE)
    else:
        plt.ylabel('Transmission (a.u.)', fontsize=config.DEFAULT_LABEL_FONTSIZE)

    plt.grid(True, alpha=config.DEFAULT_GRID_ALPHA, linestyle='--', linewidth=config.DEFAULT_GRID_LINEWIDTH)
    if legend_flag:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=config.DEFAULT_LEGEND_FONTSIZE, frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=config.DEFAULT_TICK_FONTSIZE, width=1.5, length=6)
    plt.tight_layout()


def plot_monitor(apd_data, time_limit=None, new_fig=True, legend_flag=True):
    """Plot monitor signal traces.

    Args:
        apd_data: APDData object from apd_functions
        time_limit: Maximum time to plot (seconds), None for full trace
        new_fig: Create new figure if True
        legend_flag: Show legend if True
    """
    if new_fig:
        plt.figure(figsize=(16, 8))

    sorted_keys = sorted(apd_data.transmission.keys(), key=_get_sort_key)
    # Use COLOR_ARRAY from config, cycling if needed
    colors = [config.COLOR_ARRAY[i % len(config.COLOR_ARRAY)] for i in range(len(sorted_keys))]

    for i, key in enumerate(sorted_keys):
        duration = apd_data.params[key]['Duration (s)']
        data = apd_data.monitor[key]
        time_axis = np.linspace(0, duration, len(data))

        if time_limit is not None:
            mask = time_axis <= time_limit
            data = data[mask]
            time_axis = time_axis[mask]

        power_mw = apd_data.params[key].get('power', 0)
        plt.plot(
            time_axis,
            data,
            label=f"{key} - {power_mw:.1f} mW",
            color=colors[i],
            linewidth=config.DEFAULT_LINE_WIDTH,
        )

    plt.xlabel('Time (s)', fontsize=config.DEFAULT_LABEL_FONTSIZE)
    plt.ylabel('Monitor Signal (V)', fontsize=config.DEFAULT_LABEL_FONTSIZE)
    plt.grid(True, alpha=config.DEFAULT_GRID_ALPHA, linestyle='--', linewidth=config.DEFAULT_GRID_LINEWIDTH)
    if legend_flag:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=config.DEFAULT_LEGEND_FONTSIZE, frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=config.DEFAULT_TICK_FONTSIZE, width=1.5, length=6)
    plt.tight_layout()


def plot_apd_by_power(
    apd_data,
    labels=None,
    colors=None,
    markers=None,
    power_threshold=45,
    normalize=True,
    savgol=True,
    time_limit=None,
    discard_initial=None,
    new_fig=True,
    lines=False,
    title="APD Transmission by Power",
):
    """Plot APD transmission traces grouped by power level.

    Similar to plot_snr_vs_time_by_power from confocal_plotting_functions,
    but plots full transmission traces instead of SNR values.
    Uses solid lines for low power (<threshold) and dashed for high power.

    Args:
        apd_data: APDData object or list of APDData objects
        labels: List of legend labels for each dataset (optional)
        colors: List of colors for each dataset (optional)
        markers: List of marker styles for each dataset (optional)
        power_threshold: Power threshold in mW to separate groups (default: 45)
        normalize: If True, plot as percentage change from initial value
        savgol: If True, apply Savitzky-Golay smoothing filter
        time_limit: Maximum time to plot (seconds), None for full trace
        discard_initial: Seconds to discard at start (default from config)
        new_fig: Create new figure if True, otherwise use current axes
        title: Plot title
    """
    # Convert single data to list
    if hasattr(apd_data, 'transmission'):  # Check if it's an APDData object
        data_list = [apd_data]
    else:
        data_list = apd_data

    # Set defaults
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(data_list))]
    if colors is None:
        colors = ['C0', 'C1', 'C2', 'C3'][:len(data_list)]
    if markers is None:
        markers = ['o', 's'][:len(data_list)]

    if discard_initial is None:
        discard_initial = config.APD_PLOT_DISCARD_SECONDS

    # Create plot or use existing axes
    if new_fig:
        fig, ax = plt.subplots(figsize=config.DEFAULT_FIGURE_SIZE)
    else:
        ax = plt.gca()
        fig = plt.gcf()

    # Plot each dataset
    for dataset, label, color, marker in zip(data_list, labels, colors, markers):
        low_power_traces = []
        high_power_traces = []

        # Separate traces by power
        for key, trace_data in dataset.transmission.items():
            power = dataset.params[key]['power']
            duration = dataset.params[key]['Duration (s)']
            time_axis = np.linspace(0, duration, len(trace_data))

            # Discard initial samples
            if discard_initial > 0:
                mask_discard = time_axis >= discard_initial
                trace_data = trace_data[mask_discard]
                time_axis = time_axis[mask_discard]

            # Apply time limit
            if time_limit is not None:
                mask = time_axis <= time_limit
                trace_data = trace_data[mask]
                time_axis = time_axis[mask]

            # Normalize
            if normalize:
                if len(trace_data) > 0:
                    trace_data = (trace_data - trace_data[0]) / trace_data[0] * 100
                else:
                    continue

            # Smooth
            if savgol and len(trace_data) > 51:
                trace_data = savgol_filter(trace_data, 1001, 2)
                # Re-normalize to ensure first point is 0 after smoothing
                if normalize and len(trace_data) > 0:
                    trace_data = trace_data - trace_data[0]

            # Group by power
            if power < power_threshold:
                low_power_traces.append((time_axis, trace_data, key))
            else:
                high_power_traces.append((time_axis, trace_data, key))

        # Plot low power traces (solid line)
        for time_axis, trace_data, key in low_power_traces:
            ax.plot(
                time_axis,
                trace_data,
                color=color,
                linestyle='-',
                linewidth=config.DEFAULT_LINE_WIDTH,
                alpha=0.8,
                label=label if key == low_power_traces[0][2] else "",
            )

        # Plot high power traces (dashed line)
        for time_axis, trace_data, key in high_power_traces:
            ax.plot(
                time_axis,
                trace_data,
                color=color,
                linestyle='--',
                linewidth=config.DEFAULT_LINE_WIDTH,
                alpha=0.8,
            )

    # Add reference lines at ±10% if normalized
    if normalize and lines:
        ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axhline(y=-10, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Time (s)', fontsize=config.DEFAULT_LABEL_FONTSIZE)
    if normalize:
        ax.set_ylabel('Transmission Change (%)', fontsize=config.DEFAULT_LABEL_FONTSIZE)
    else:
        ax.set_ylabel('Transmission (a.u.)', fontsize=config.DEFAULT_LABEL_FONTSIZE)
    ax.tick_params(labelsize=config.DEFAULT_TICK_FONTSIZE, width=1.5, length=6)
    ax.legend(fontsize=config.DEFAULT_LEGEND_FONTSIZE, frameon=False)

    # Add text annotation for power levels
    low_power_text = f"{power_threshold - 15:.0f} mW"
    high_power_text = f"{power_threshold + 15:.0f} mW"
    ax.text(
        0.02, 0.02,
        f'Solid: {low_power_text}, Dashed: {high_power_text}',
        transform=ax.transAxes,
        fontsize=16,
        ha='left',
        va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none')
    )

    ax.grid(True, alpha=config.DEFAULT_GRID_ALPHA, linestyle=':', linewidth=config.DEFAULT_GRID_LINEWIDTH)

    plt.tight_layout()
    return fig, ax



if __name__ == "__main__":
    # Example usage
    from . import apd_functions

    path = r"Data\apd_downsampled\sample_example"

    # Load data
    apd_data = apd_functions.apd_main(path)

    # Plot transmission
    plot_apd(apd_data, normalize=True, group_colors=True)
    plt.savefig("apd_transmission.png")

    # Plot monitor
    plot_monitor(apd_data)
    plt.savefig("apd_monitor.png")

    plt.show()
