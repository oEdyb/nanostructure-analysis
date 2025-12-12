from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from confocal_functions import analyze_confocal, confocal_main, filter_confocal, load_with_cache
from sem_functions import load_sem_measurements
from spectra_functions import baseline_als, filter_spectra, spectra_main


SPECTRA_PATH = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251024 - Sample 21 Gap Widths 24"
REFERENCE_PATH = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251013 - Sample 5 24 of the same"
SEM_CSV_PATH = r"Data/SEM/SEM_measurements_20251029_sample_21_gap_widths.csv"
CONFOCAL_PATH = r"\\AMIPC045962\daily_data\confocal_data\20251024 - Sample 21 Gap Widths 24"

SENSITIVITY_RANGE = (837, 873)
PLOT_DIR = "plots/sample_21_correlations"


def normalize_against_reference(
    spectra_dict: Dict[str, np.ndarray],
    reference_dict: Dict[str, np.ndarray],
    lam: float = 1e5,
    p: float = 0.5
) -> Dict[str, np.ndarray]:
    """Divide spectra by reference and apply ALS baseline correction."""
    if not reference_dict:
        return spectra_dict

    reference_key = next(iter(reference_dict))
    reference_intensity = reference_dict[reference_key][:, 1]

    normalized = {}
    for key, spectrum in spectra_dict.items():
        intensity = spectrum[:, 1] / reference_intensity
        corrected = baseline_als(intensity, lam, p)
        normalized[key] = np.column_stack((spectrum[:, 0], corrected))
    return normalized


def compute_sensitivity_stat(values: Iterable[float]) -> float:
    """Mean intensity within the target spectral window."""
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return 0.0
    return float(np.mean(np.abs(data)))


def compute_sensitivity_series(
    spectra_data: Dict[str, np.ndarray],
    wavelength_range: Tuple[int, int],
    id_pattern: str = r"([A-D])\s*(\d)"
) -> Dict[str, Dict[str, List[float]]]:
    """Build per-measurement sensitivity (dI/dλ) series for downstream stats."""
    compiled = re.compile(id_pattern)
    measurement_series: Dict[str, Dict[str, List[float]]] = {}

    for key, data in spectra_data.items():
        match = compiled.search(key)
        if not match:
            continue
        group_letter, index = match.groups()
        measurement_id = f"{group_letter}{index}"

        mask = (data[:, 0] >= wavelength_range[0]) & (data[:, 0] <= wavelength_range[1])
        if np.count_nonzero(mask) < 2:
            continue

        intensities = data[mask, 1]
        if intensities.size == 0:
            continue

        series_entry = measurement_series.setdefault(
            measurement_id,
            {"group": group_letter, "values": []}
        )
        series_entry["values"].extend(intensities.tolist())

    return measurement_series


def extract_snr_by_measurement(confocal_results: Dict[str, Dict[str, float]], pattern: str = r"([A-D])(\d)") -> Dict[str, float]:
    """Map confocal result keys onto measurement IDs (A1, B3, ...) and pull SNR."""
    compiled = re.compile(pattern)
    snr_map: Dict[str, float] = {}
    for key, result in confocal_results.items():
        match = compiled.search(key)
        if not match:
            continue
        measurement_id = f"{match.group(1)}{match.group(2)}"
        snr_value = result.get("snr_3x3")
        if snr_value is None:
            continue
        snr_map[measurement_id] = float(snr_value)
    return snr_map


def compute_sensitivity_scores(
    spectra_data: Dict[str, np.ndarray],
    wavelength_range: Tuple[int, int]
) -> Dict[str, float]:
    """Return a single sensitivity score per measurement."""
    series = compute_sensitivity_series(spectra_data, wavelength_range=wavelength_range)
    scores: Dict[str, float] = {}
    for measurement_id, entry in series.items():
        values = entry.get("values") or []
        if not values:
            continue
        scores[measurement_id] = compute_sensitivity_stat(values)
    return scores


def collect_feature_keys(sem_measurements: Dict[str, Dict[str, float]]) -> List[str]:
    """Gather numeric SEM feature keys to include in correlation analysis."""
    ignore_keys = {"group", "snr_3x3"}
    feature_names = set()
    for entry in sem_measurements.values():
        for key, value in entry.items():
            if key in ignore_keys:
                continue
            if isinstance(value, (int, float)):
                feature_names.add(key)
    return sorted(feature_names)


def compute_correlations(
    sem_measurements: Dict[str, Dict[str, float]],
    feature_keys: Iterable[str],
    target_values: Dict[str, float]
) -> List[Tuple[str, float, int]]:
    """Compute Pearson correlation between SEM features and the supplied target."""
    results: List[Tuple[str, float, int]] = []
    for feature in feature_keys:
        feature_vector: List[float] = []
        target_vector: List[float] = []
        for measurement_id, entry in sem_measurements.items():
            if feature not in entry:
                continue
            target = target_values.get(measurement_id)
            if target is None:
                continue
            value = entry[feature]
            if value is None:
                continue
            if not np.isfinite(value) or not np.isfinite(target):
                continue
            feature_vector.append(float(value))
            target_vector.append(float(target))
        if len(feature_vector) < 2:
            continue
        corr = float(np.corrcoef(feature_vector, target_vector)[0, 1])
        if not np.isfinite(corr):
            continue
        results.append((feature, corr, len(feature_vector)))
    results.sort(key=lambda item: abs(item[1]), reverse=True)
    return results


def plot_all_spectra(
    spectra_data: Dict[str, np.ndarray],
    output_path: str,
    title: str
) -> None:
    """Overlay all spectra in a single figure for quick inspection."""
    if not spectra_data:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for key in sorted(spectra_data):
        data = spectra_data[key]
        if data.shape[1] < 2:
            continue
        ax.plot(data[:, 0], data[:, 1], linewidth=1.0, alpha=0.8, label=key)

    ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Intensity (a.u.)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(color='#E4E4E4', linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.85)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_correlation_summary(
    snr_correlations: List[Tuple[str, float, int]],
    sensitivity_correlations: List[Tuple[str, float, int]],
    output_path: str,
    top_k: int = 12
) -> None:
    """Create side-by-side bar charts for top correlations."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def select_top(entries: List[Tuple[str, float, int]]) -> List[Tuple[str, float, int]]:
        filtered = [item for item in entries if np.isfinite(item[1])]
        selected = filtered[:top_k]
        gap_hit = next((item for item in filtered if item[0] == "gap_width"), None)
        if gap_hit and all(item[0] != "gap_width" for item in selected):
            selected = selected + [gap_hit]
        return selected

    snr_top = select_top(snr_correlations)
    sensitivity_top = select_top(sensitivity_correlations)
    max_len = max(len(snr_top), len(sensitivity_top), 1)

    fig_height = 0.55 * max_len + 2.5
    fig, axes = plt.subplots(1, 2, figsize=(14, fig_height), sharex=True)
    axes = axes.ravel()

    for ax, data, title in zip(
        axes,
        (snr_top, sensitivity_top),
        ("SNR Correlation", "Sensitivity Correlation")
    ):
        if not data:
            ax.set_visible(False)
            continue
        labels = [f"{feature} (n={count})" for feature, _, count in reversed(data)]
        values = [corr for _, corr, _ in reversed(data)]
        colors = ['#D7191C' if val >= 0 else '#2C7BB6' for val in values]
        positions = np.arange(len(values))
        ax.barh(positions, values, color=colors, alpha=0.85)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(-1, 1)
        ax.set_xlabel('r', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axvline(0, color='#333333', linewidth=1.0)
        ax.grid(axis='x', color='#E4E4E4', linewidth=0.7, alpha=0.7)
        for pos, value in zip(positions, values):
            offset = 0.02 if value >= 0 else -0.02
            ha = 'left' if value >= 0 else 'right'
            ax.text(value + offset, pos, f"{value:+.2f}", va='center', ha=ha, fontsize=9)

    fig.suptitle('Top SEM Feature Correlations', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    spectra_data, spectra_params = spectra_main(SPECTRA_PATH)
    spectra_ref, spectra_params_ref = spectra_main(REFERENCE_PATH)

    spectra_data, spectra_params = filter_spectra(
        spectra_data,
        spectra_params,
        "[A-D]*",
        average=False,
        exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*B4*"]
    )
    spectra_ref, spectra_params_ref = filter_spectra(
        spectra_ref,
        spectra_params_ref,
        "*20um*",
        average=False,
        exclude=["*bias*", "*baseline*"]
    )
    spectra_data = normalize_against_reference(spectra_data, spectra_ref)

    sem_measurements = load_sem_measurements(SEM_CSV_PATH)
    confocal_data = load_with_cache(CONFOCAL_PATH, confocal_main)
    confocal_filtered = filter_confocal(confocal_data, "*", exclude=["after"])
    confocal_results = analyze_confocal(confocal_filtered)
    snr_by_measurement = extract_snr_by_measurement(confocal_results)

    for measurement_id, snr_value in snr_by_measurement.items():
        if measurement_id in sem_measurements:
            sem_measurements[measurement_id]["snr_3x3"] = snr_value

    sensitivity_scores = compute_sensitivity_scores(spectra_data, SENSITIVITY_RANGE)
    feature_keys = collect_feature_keys(sem_measurements)

    snr_correlations = compute_correlations(sem_measurements, feature_keys, snr_by_measurement)
    sensitivity_correlations = compute_correlations(sem_measurements, feature_keys, sensitivity_scores)

    def emit_results(title: str, summary: List[Tuple[str, float, int]]) -> None:
        print(f"\n{title}")
        print("-" * len(title))
        if not summary:
            print("  No overlapping measurements found.")
            return
        for feature, corr, count in summary:
            print(f"  {feature:35s} r={corr:+.3f} (n={count})")

    emit_results("Correlation vs. SNR", snr_correlations)
    emit_results("Correlation vs. Sensitivity", sensitivity_correlations)

    plot_path = os.path.join(PLOT_DIR, "sem_correlation_summary.png")
    plot_correlation_summary(snr_correlations, sensitivity_correlations, plot_path)
    print(f"\nSaved correlation summary plot to {plot_path}")

    spectra_plot_path = os.path.join(PLOT_DIR, "normalized_spectra_overlay.png")
    plot_all_spectra(spectra_data, spectra_plot_path, "Normalized Spectra Overlay")
    print(f"Saved spectra overlay plot to {spectra_plot_path}")

    reference_plot_path = os.path.join(PLOT_DIR, "reference_spectra_overlay.png")
    plot_all_spectra(spectra_ref, reference_plot_path, "Reference Spectra Overlay")
    print(f"Saved reference spectra overlay plot to {reference_plot_path}")


if __name__ == "__main__":
    main()
