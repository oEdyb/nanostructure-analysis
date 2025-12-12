import matplotlib.pyplot as plt
import numpy as np


def plot_peak_wavelength_vs_sem(matched_data, sem_param, new_fig=True):
    """Plot wavelength of peak intensity vs SEM parameter."""
    if new_fig:
        plt.figure(figsize=(10, 6))

    x_vals, y_vals, labels = [], [], []
    for id, data in matched_data.items():
        spectrum = data['spectra']
        peak_idx = np.argmax(spectrum[:, 1])
        peak_wavelength = spectrum[peak_idx, 0]

        sem_value = data['sem'][sem_param]
        x_vals.append(sem_value)
        y_vals.append(peak_wavelength)
        labels.append(id)

    plt.scatter(x_vals, y_vals, s=70, alpha=0.85, edgecolors='white', linewidths=0.8)

    for x, y, label in zip(x_vals, y_vals, labels):
        plt.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10, alpha=0.8)

    plt.xlabel(f'{sem_param}', fontsize=18, fontweight='bold')
    plt.ylabel('Peak Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.title(f'Peak Wavelength vs {sem_param}', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()


def plot_derivative_vs_sem(matched_data, sem_param, wavelength_range=(837, 871), new_fig=True):
    """Plot mean derivative in wavelength range vs SEM parameter."""
    if new_fig:
        plt.figure(figsize=(10, 6))

    x_vals, y_vals, labels = [], [], []
    start, stop = wavelength_range

    for id, data in matched_data.items():
        spectrum = data['spectra']
        wavelength = spectrum[:, 0]
        intensity = spectrum[:, 1]

        mask = (wavelength >= start) & (wavelength <= stop)
        if mask.sum() < 2:
            continue

        wl_masked = wavelength[mask]
        int_masked = intensity[mask]
        dlambda = np.mean(np.diff(wl_masked))

        if np.isclose(dlambda, 0):
            continue

        derivative = np.diff(int_masked) / dlambda
        mean_deriv = np.mean(derivative)

        sem_value = data['sem'][sem_param]
        x_vals.append(sem_value)
        y_vals.append(mean_deriv)
        labels.append(id)

    plt.scatter(x_vals, y_vals, s=70, alpha=0.85, edgecolors='white', linewidths=0.8)

    for x, y, label in zip(x_vals, y_vals, labels):
        plt.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10, alpha=0.8)

    plt.xlabel(f'{sem_param}', fontsize=18, fontweight='bold')
    plt.ylabel(f'Mean dI/dλ (a.u./nm)', fontsize=18, fontweight='bold')
    plt.title(f'Sensitivity vs {sem_param}', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
