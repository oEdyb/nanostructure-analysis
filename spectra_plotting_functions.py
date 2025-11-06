import matplotlib.pyplot as plt
import apd_functions
import numpy as np
from scipy.signal import savgol_filter
import re

import spectra_functions


def _group_color_map(sorted_keys, fallback_colors):
    preferred = {'A': 'tab:blue', 'B': 'tab:orange', 'C': 'tab:green', 'D': 'tab:red'}
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    color_by_group = {}
    for key in sorted_keys:
        match = re.search(r'([A-D])[1-6]', key)
        if not match:
            continue
        letter = match.group(1)
        if letter in color_by_group:
            continue
        idx = len(color_by_group)
        fallback = default_cycle[idx % len(default_cycle)] if default_cycle else fallback_colors[idx % len(fallback_colors)]
        color_by_group[letter] = preferred.get(letter, fallback)
    return color_by_group

def plot_spectra(spectra_data, spectra_params, cutoff=950, new_fig=True, linestyle='-', legend_flag=True):
    if new_fig:
        plt.figure(figsize=(16, 8))
    
    # Sort keys based on [A-D][1-6] pattern if it exists
    import re
    def get_sort_key(key):
        # Look for pattern like A1, B2, C3, D6 etc in the key
        match = re.search(r'[A-D][1-6]', key)
        if match:
            letter_part = match.group()[0]  # A, B, C, or D
            number_part = int(match.group()[1])  # 1-6
            return (ord(letter_part), number_part)  # Sort by letter first, then number
        return (999, 999)  # Put non-matching keys at the end
    
    sorted_keys = sorted(spectra_data.keys(), key=get_sort_key)
    
    # Define consistent color scheme - use colormap to handle any number of datasets
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(sorted_keys)))
    
    for i, key in enumerate(sorted_keys):
        exposure_time = spectra_params[key]['Exposure time (s)']
        # Dynamic formatting: use appropriate precision based on value magnitude
        if exposure_time >= 1:
            time_str = f"{exposure_time:.0f} s"
        elif exposure_time >= 0.1:
            time_str = f"{exposure_time:.1f} s"
        elif exposure_time >= 0.01:
            time_str = f"{exposure_time:.2f} s" 
        else:
            time_str = f"{exposure_time:.3f} s"
        
        # Filter data below cutoff wavelength
        wavelength = spectra_data[key][:, 0]
        intensity = spectra_data[key][:, 1]
        mask = wavelength <= cutoff
        
        plt.plot(wavelength[mask], intensity[mask], label=f"{key}", 
                linestyle=linestyle, color=colors[i])
        
    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.ylabel('arbitrary units', fontsize=18, fontweight='bold')
    plt.title('Spectra Data', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    if legend_flag:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()

def plot_spectra_integrated_histogram(spectra_data, spectra_params, cutoff=950, new_fig=True, bins='auto'):
    """Plot a histogram of the raw integrated spectra intensities."""
    if new_fig:
        plt.figure(figsize=(12, 6))
    
    def get_sort_key(key):
        import re
        match = re.search(r'[A-D][1-6]', key)
        if match:
            letter_part = match.group()[0]
            number_part = int(match.group()[1])
            return (ord(letter_part), number_part)
        return (999, 999)
    
    sorted_keys = sorted(spectra_data.keys(), key=get_sort_key)
    
    integrated_values = []
    labels = []
    for key in sorted_keys:
        wavelength = spectra_data[key][:, 0]
        intensity = spectra_data[key][:, 1]
        mask = wavelength <= cutoff
        if not np.any(mask):
            continue
        integrated_intensity = np.trapz(intensity[mask], wavelength[mask])
        integrated_values.append(integrated_intensity)
        labels.append(key)
    
    if not integrated_values:
        raise ValueError("No spectra data within the provided cutoff to integrate.")
    
    plt.hist(integrated_values, bins=bins, edgecolor='black', alpha=0.85)
    
    ax = plt.gca()
    ax.set_xlabel('Integrated Intensity (a.u. nm)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Count', fontsize=18, fontweight='bold')
    ax.set_title('Integrated Spectra Histogram', fontsize=22, fontweight='bold')
    ax.grid(True, alpha=0.7, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    
    return dict(zip(labels, integrated_values))

def plot_spectra_wavelength_histogram(spectra_data, spectra_params, wavelength_target=852, new_fig=True, bins='auto'):
    """Plot a histogram of the raw intensities sampled at a specific wavelength."""
    if new_fig:
        plt.figure(figsize=(12, 6))
    
    def get_sort_key(key):
        import re
        match = re.search(r'[A-D][1-6]', key)
        if match:
            letter_part = match.group()[0]
            number_part = int(match.group()[1])
            return (ord(letter_part), number_part)
        return (999, 999)
    
    sorted_keys = sorted(spectra_data.keys(), key=get_sort_key)
    
    sampled_values = []
    labels = []
    for key in sorted_keys:
        wavelength = spectra_data[key][:, 0]
        intensity = spectra_data[key][:, 1]
        if wavelength_target < wavelength.min() or wavelength_target > wavelength.max():
            continue
        sampled_intensity = np.interp(wavelength_target, wavelength, intensity)
        sampled_values.append(sampled_intensity)
        labels.append(key)
    
    if not sampled_values:
        raise ValueError(f"No spectra contain the wavelength {wavelength_target} nm.")
    
    plt.hist(sampled_values, bins=bins, edgecolor='black', alpha=0.85)
    
    ax = plt.gca()
    ax.set_xlabel(f'Intensity at {wavelength_target} nm (a.u.)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Count', fontsize=18, fontweight='bold')
    ax.set_title(f'Spectra Histogram @ {wavelength_target} nm', fontsize=22, fontweight='bold')
    ax.grid(True, alpha=0.7, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    
    return dict(zip(labels, sampled_values))

def plot_spectra_std_scatter(
    spectra_data,
    spectra_params=None,
    cutoff=950,
    new_fig=True,
    marker='o',
    normalize_by_mean=True
):
    """Plot the per-wavelength spread across spectra as a scatter plot.

    When normalize_by_mean is True, the function plots std/mean (coefficient of variation)
    so wavelengths with higher overall intensity do not dominate the visualization.
    """
    if not spectra_data:
        raise ValueError("spectra_data must contain at least one spectrum.")
    
    if new_fig:
        plt.figure(figsize=(16, 8))
    
    first_key = next(iter(spectra_data))
    reference_wavelengths = spectra_data[first_key][:, 0]
    
    stacked_intensities = []
    for data in spectra_data.values():
        wavelengths = data[:, 0]
        intensities = data[:, 1]
        if len(wavelengths) != len(reference_wavelengths) or not np.allclose(wavelengths, reference_wavelengths):
            intensities = np.interp(reference_wavelengths, wavelengths, intensities)
        stacked_intensities.append(intensities)
    
    intensities_array = np.vstack(stacked_intensities)
    std_values = np.std(intensities_array, axis=0)
    mean_values = np.mean(intensities_array, axis=0)
    
    if normalize_by_mean:
        eps = np.finfo(float).eps
        display_values = np.divide(std_values, mean_values, out=np.zeros_like(std_values), where=np.abs(mean_values) > eps)
    else:
        display_values = std_values
    
    mask = reference_wavelengths <= cutoff
    plt.scatter(
        reference_wavelengths[mask],
        display_values[mask],
        s=30,
        alpha=0.85,
        marker=marker,
        color='tab:blue'
    )
    
    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    if normalize_by_mean:
        plt.ylabel('Std / Mean (a.u.)', fontsize=18, fontweight='bold')
        plt.title('Spectra Relative Spread vs Wavelength', fontsize=22, fontweight='bold')
    else:
        plt.ylabel('Standard Deviation (a.u.)', fontsize=18, fontweight='bold')
        plt.title('Spectra Standard Deviation vs Wavelength', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    
    return reference_wavelengths, display_values

def plot_spectra_savgol(spectra_data, spectra_params, window_sizes=[11, 21, 31, 51], orders=[1, 2, 3], cutoff=950):
    from scipy.signal import savgol_filter
    
    plt.figure(figsize=(16, 8))
    
    # Get first spectrum only
    first_key = list(spectra_data.keys())[0]
    data = spectra_data[first_key]
    exposure_time = spectra_params[first_key]['Exposure time (s)']
    
    # Dynamic formatting for exposure time
    if exposure_time >= 1:
        time_str = f"{exposure_time:.0f} s"
    elif exposure_time >= 0.1:
        time_str = f"{exposure_time:.1f} s"
    elif exposure_time >= 0.01:
        time_str = f"{exposure_time:.2f} s" 
    else:
        time_str = f"{exposure_time:.3f} s"
    
    # Filter data below cutoff wavelength
    wavelength = data[:, 0]
    intensity = data[:, 1]
    mask = wavelength <= cutoff
    
    # Plot raw data
    plt.plot(wavelength[mask], intensity[mask], label=f"Raw - {first_key} - {time_str}", alpha=0.7, linewidth=2)
    
    # Plot savgol filtered versions with different windows and orders
    for window in window_sizes:
        for order in orders:
            # Apply savgol filter with current window size and order
            filtered_y = savgol_filter(intensity, window, order)
            plt.plot(wavelength[mask], filtered_y[mask], label=f"Savgol w={window}, o={order}", linewidth=1, alpha=0.9)
        
    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.ylabel('arbitrary units', fontsize=18, fontweight='bold')
    plt.title('Savgol Filter Comparison', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()



def plot_spectra_transparent(spectra_data, spectra_params, cutoff=950, new_fig=True, linestyle='-'):
    # Create figure with transparent background
    if new_fig:
        fig, ax = plt.subplots(figsize=(16, 8))
        fig.patch.set_alpha(0)  # Transparent figure background
        ax.patch.set_alpha(0)   # Transparent axis background

    
    # Sort keys based on [A-D][1-6] pattern if it exists
    def get_sort_key(key):
        # Look for pattern like A1, B2, C3, D6 etc in the key
        match = re.search(r'[A-D][1-6]', key)
        if match:
            letter_part = match.group()[0]  # A, B, C, or D
            number_part = int(match.group()[1])  # 1-6
            return (ord(letter_part), number_part)  # Sort by letter first, then number
        return (999, 999)  # Put non-matching keys at the end
    
    sorted_keys = sorted(spectra_data.keys(), key=get_sort_key)
    
    # Define consistent color scheme - expands to handle many datasets
    colors = plt.cm.tab10(range(10)) if len(sorted_keys) <= 10 else plt.cm.tab20(range(20))
    
    for i, key in enumerate(sorted_keys):
        exposure_time = spectra_params[key]['Exposure time (s)']
        # Dynamic formatting: use appropriate precision based on value magnitude
        if exposure_time >= 1:
            time_str = f"{exposure_time:.0f} s"
        elif exposure_time >= 0.1:
            time_str = f"{exposure_time:.1f} s"
        elif exposure_time >= 0.01:
            time_str = f"{exposure_time:.2f} s" 
        else:
            time_str = f"{exposure_time:.3f} s"
        
        # Filter data below cutoff wavelength
        wavelength = spectra_data[key][:, 0]
        intensity = spectra_data[key][:, 1]
        mask = wavelength <= cutoff
        
        plt.plot(wavelength[mask], intensity[mask], label=f"{key} - {time_str}", 
                linestyle=linestyle, color=colors[i % len(colors)])
    # Remove all visual elements for clean PPT insertion
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('')  # Remove y-axis label
    ax.set_title('')   # Remove title
    ax.grid(False)     # Remove grid
    ax.legend().remove() if ax.get_legend() else None  # Remove legend
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.spines['top'].set_visible(False)     # Remove top border
    ax.spines['right'].set_visible(False)   # Remove right border
    ax.spines['bottom'].set_visible(False)  # Remove bottom border
    ax.spines['left'].set_visible(False)    # Remove left border
    plt.tight_layout()

def plot_spectra_derivative(spectra_data, spectra_params, cutoff=950, new_fig=True, linestyle='-'):
    """Plot the wavelength derivative of each spectrum using an average delta lambda."""
    if new_fig:
        plt.figure(figsize=(16, 8))

    def get_sort_key(key):
        match = re.search(r'[A-D][1-6]', key)
        if match:
            letter_part = match.group()[0]
            number_part = int(match.group()[1])
            return (ord(letter_part), number_part)
        return (999, 999)

    sorted_keys = sorted(spectra_data.keys(), key=get_sort_key)
    fallback_colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(sorted_keys)))
    color_by_group = _group_color_map(sorted_keys, fallback_colors)

    for i, key in enumerate(sorted_keys):
        wavelength = spectra_data[key][:, 0]
        intensity = spectra_data[key][:, 1]

        exposure_time = spectra_params[key]['Exposure time (s)']
        if exposure_time >= 1:
            time_str = f"{exposure_time:.0f} s"
        elif exposure_time >= 0.1:
            time_str = f"{exposure_time:.1f} s"
        elif exposure_time >= 0.01:
            time_str = f"{exposure_time:.2f} s"
        else:
            time_str = f"{exposure_time:.3f} s"

        mask = wavelength <= cutoff
        wavelength_masked = wavelength[mask]
        intensity_masked = intensity[mask]

        if wavelength_masked.size < 2:
            continue

        dlambda_avg = np.mean(np.diff(wavelength_masked))
        if np.isclose(dlambda_avg, 0):
            continue

        derivative = np.diff(intensity_masked) / dlambda_avg
        wavelength_midpoints = wavelength_masked[:-1] + dlambda_avg / 2

        group_match = re.search(r'([A-D])[1-6]', key)
        group_letter = group_match.group(1) if group_match else None
        line_color = color_by_group.get(group_letter, fallback_colors[i])

        plt.plot(
            wavelength_midpoints,
            derivative,
            label=f"{key}",
            linestyle=linestyle,
            color=line_color,
        )

    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.ylabel('dI/dλ (a.u. / nm)', fontsize=18, fontweight='bold')
    plt.title('Spectra Derivative', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()


def plot_integrated_transmission_scatter(spectra_data, cutoff=950, new_fig=True):
    if not spectra_data:
        return
    if new_fig:
        plt.figure(figsize=(10, 6))

    def get_sort_key(key):
        match = re.search(r'[A-D][1-6]', key)
        if match:
            return (ord(match.group()[0]), int(match.group()[1]))
        return (999, key)

    sorted_keys = sorted(spectra_data.keys(), key=get_sort_key)
    fallback_colors = plt.cm.gist_rainbow(np.linspace(0, 1, max(len(sorted_keys), 1)))
    color_by_group = _group_color_map(sorted_keys, fallback_colors)

    grouped_areas = {}
    for key in sorted_keys:
        wavelength = spectra_data[key][:, 0]
        intensity = spectra_data[key][:, 1]
        mask = wavelength <= cutoff
        if mask.sum() < 2:
            continue
        area = np.trapezoid(intensity[mask], wavelength[mask])
        match = re.search(r'([A-D])[1-6]', key)
        group = match.group(1) if match else 'Other'
        grouped_areas.setdefault(group, []).append(area)

    ordered_groups = [g for g in ['A', 'B', 'C', 'D'] if g in grouped_areas]
    ordered_groups.extend(
        group for group in grouped_areas.keys() if group not in ordered_groups
    )
    if not ordered_groups:
        return

    x_positions = {group: idx for idx, group in enumerate(ordered_groups)}

    for idx, group in enumerate(ordered_groups):
        values = grouped_areas.get(group, [])
        if not values:
            continue
        color = color_by_group.get(group, fallback_colors[idx % len(fallback_colors)])
        jitter = (np.random.rand(len(values)) - 0.5) * 0.15
        x_vals = np.full(len(values), x_positions[group]) + jitter
        plt.scatter(
            x_vals,
            values,
            color=color,
            alpha=0.85,
            edgecolors='white',
            linewidths=0.8,
            s=70,
            label=group,
        )

    plt.xlabel('Group', fontsize=18, fontweight='bold')
    plt.ylabel('Integrated transmission (a.u.* nm)', fontsize=18, fontweight='bold')
    plt.title('Integrated Transmission by Group', fontsize=22, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4, axis='y')
    plt.xticks(
        [x_positions[g] for g in ordered_groups],
        ordered_groups,
        fontsize=16,
        fontweight='bold',
    )
    plt.legend(loc='best', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()


def plot_peak_wavelength_histogram(spectra_data, output_path, bins=20):
    """Plot distribution of peak wavelengths across spectra."""
    if not spectra_data:
        return

    peak_wavelengths = [
        spectrum[np.argmax(spectrum[:, 1]), 0]
        for spectrum in spectra_data.values()
        if spectrum.size
    ]
    if not peak_wavelengths:
        return

    peak_array = np.asarray(peak_wavelengths, dtype=float)
    mean_peak = float(np.mean(peak_array))
    std_peak = float(np.std(peak_array))

    fig = plt.figure()
    plt.hist(peak_array, bins=bins, edgecolor='black')
    plt.xlabel('Peak Wavelength (nm)')
    plt.ylabel('Count')
    plt.title(f'Distribution of Peak Wavelengths\nMean: {mean_peak:.1f} nm, Std: {std_peak:.1f} nm')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":

    spectra_path = "./Data/Spectra/20250821 - sample13 - after"
    spectra_data, spectra_params = spectra_functions.spectra_main(spectra_path)
    spectra_data_box1, spectra_params_box1 = spectra_functions.filter_spectra(spectra_data, spectra_params, "*box1*")
    spectra_data_5um, spectra_params_5um = spectra_functions.filter_spectra(spectra_data, spectra_params, "*5um*")
    spectra_data_bkg, spectra_params_bkg = spectra_functions.filter_spectra(spectra_data, spectra_params, "*bkg*")
    plot_spectra(spectra_data_box1, spectra_params_box1)
    plt.savefig("spectra_box1.png")
    plt.show()
    plot_spectra(spectra_data_5um, spectra_params_5um)
    plt.savefig("spectra_5um.png")
    plt.show()
    plot_spectra(spectra_data_bkg, spectra_params_bkg)
    plt.savefig("spectra_bkg.png")
    plt.show()
