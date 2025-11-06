import csv
import os
import re
from typing import Dict, List, Sequence

import numpy as np

from spectra_functions import compute_sensitivity_stat


def load_sem_measurements(
    csv_path: str,
    id_pattern: str = r"_([A-D])(\d)",
    drop_tilted: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Load SEM measurements, collapse repeated rows, and derive handy features.

    Returns a dict keyed by measurement id (e.g. 'A1') with averaged metrics:
    - gap_mean/gap_std (nm)
    - tip_mean/tip_std
    - radius_top, radius_bottom, etc.
    """
    if not os.path.exists(csv_path):
        return {}

    pattern = re.compile(id_pattern)
    grouped_rows: Dict[str, Dict[str, List[float]]] = {}
    groups: Dict[str, str] = {}  # measurement_id -> group letter

    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            label = row.get("label", "")
            if not label:
                continue
            if drop_tilted and (
                "_tilted" in label.lower()
                or row.get("tilted_flag", "").strip().lower() == "true"
            ):
                continue

            match = pattern.search(label)
            if not match:
                continue
            group_letter, index = match.groups()
            measurement_id = f"{group_letter}{index}"
            groups[measurement_id] = group_letter

            numeric_values = {}
            for key, value in row.items():
                if key in {"id", "label", "tilted_flag"}:
                    continue
                try:
                    numeric_values[key] = float(value)
                except (TypeError, ValueError):
                    continue

            if not numeric_values:
                continue

            measurement_store = grouped_rows.setdefault(measurement_id, {})
            for key, value in numeric_values.items():
                measurement_store.setdefault(key, []).append(value)

    processed: Dict[str, Dict[str, float]] = {}
    for measurement_id, value_map in grouped_rows.items():
        entry: Dict[str, float] = {"group": groups[measurement_id]}

        def store_mean_std(key: str, mean_key: str, std_key: str) -> None:
            values = value_map.get(key)
            if not values:
                return
            arr = np.asarray(values, dtype=float)
            entry[mean_key] = float(arr.mean())
            entry[std_key] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0

        store_mean_std("gap_width", "gap_mean", "gap_std")
        store_mean_std("tip_curvature_mean", "tip_mean", "tip_std")

        for key, values in value_map.items():
            arr = np.asarray(values, dtype=float)
            entry[key] = float(arr.mean())

        radius_top = entry.get("radius_top")
        radius_bottom = entry.get("radius_bottom")
        if radius_top is not None and radius_bottom is not None:
            entry["radius_top_minus_radius_bottom"] = abs(radius_top - radius_bottom)
            entry["radius_sum"] = float(radius_top + radius_bottom)
            if radius_bottom != 0:
                entry["radius_top_div_radius_bottom"] = float(radius_top / radius_bottom)

        top_center_x = entry.get("center_top_x")
        bottom_center_x = entry.get("center_bottom_x")
        if top_center_x is not None and bottom_center_x is not None:
            entry["center_top_x_minus_center_bottom_x"] = abs(top_center_x - bottom_center_x)

        top_center_y = entry.get("center_top_y")
        bottom_center_y = entry.get("center_bottom_y")
        if top_center_y is not None and bottom_center_y is not None:
            entry["center_top_y_minus_center_bottom_y"] = abs(top_center_y - bottom_center_y)

        dx = entry.get("center_top_x_minus_center_bottom_x")
        dy = entry.get("center_top_y_minus_center_bottom_y")
        if dx is not None and dy is not None:
            entry["center_offset_magnitude"] = float(np.hypot(dx, dy))

        offset_left = entry.get("offset_left")
        offset_right = entry.get("offset_right")
        if offset_left is not None and offset_right is not None:
            entry["offset_avg"] = float((offset_left + offset_right) / 2)
            entry["offset_diff_abs"] = abs(offset_left - offset_right)

        tilt_mean = entry.get("tilt_tip_degree_mean")
        if tilt_mean is not None:
            entry["tilt_tip_degree_mean_abs"] = abs(tilt_mean)

        tilt_left = entry.get("tilt_tip_degree_left")
        if tilt_left is not None:
            entry["tilt_tip_degree_left_abs"] = abs(tilt_left)

        tilt_right = entry.get("tilt_tip_degree_right")
        if tilt_right is not None:
            entry["tilt_tip_degree_right_abs"] = abs(tilt_right)

        processed[measurement_id] = entry

    return processed


def summarize_sem_groups(sem_measurements: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Collapse per-measurement stats to group-level means and standard deviations."""
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for entry in sem_measurements.values():
        group = entry.get("group")
        if not group:
            continue
        store = grouped.setdefault(group, {"gap": [], "tip": []})
        if entry.get("gap_mean") is not None:
            store["gap"].append(entry["gap_mean"])
        if entry.get("tip_mean") is not None:
            store["tip"].append(entry["tip_mean"])

    summary: Dict[str, Dict[str, float]] = {}
    for group, values in grouped.items():
        gap_arr = np.asarray(values["gap"], dtype=float)
        tip_arr = np.asarray(values["tip"], dtype=float)
        if gap_arr.size == 0 or tip_arr.size == 0:
            continue
        summary[group] = {
            "gap_mean": float(gap_arr.mean()),
            "gap_std": float(gap_arr.std(ddof=1)) if gap_arr.size > 1 else 0.0,
            "tip_mean": float(tip_arr.mean()),
            "tip_std": float(tip_arr.std(ddof=1)) if tip_arr.size > 1 else 0.0,
        }
    return summary


def add_combined_sem_metric(
    sem_measurements: Dict[str, Dict[str, float]],
    measurement_series: Dict[str, Dict[str, Sequence[float]]],
    feature_keys: Sequence[str],
    output_key: str,
    target_fn=compute_sensitivity_stat,
):
    """
    Fit a linear combination of SEM features to a sensitivity target and cache score.

    Returns the coefficient vector, or None if insufficient data.
    """
    result = prepare_sem_regression_data(
        sem_measurements,
        measurement_series,
        feature_keys,
        target_fn=target_fn
    )
    if result is None:
        return None

    z_features, target, measurement_ids, _, _ = result

    coeffs, _, _, _ = np.linalg.lstsq(z_features, target, rcond=None)
    combined_scores = z_features.dot(coeffs)

    for measurement_id, value in zip(measurement_ids, combined_scores):
        sem_measurements[measurement_id][output_key] = float(value)

    return coeffs


def prepare_sem_regression_data(
    sem_measurements: Dict[str, Dict[str, float]],
    measurement_series: Dict[str, Dict[str, Sequence[float]]],
    feature_keys: Sequence[str],
    target_fn=compute_sensitivity_stat,
):
    """Return z-scored feature matrix, target vector, and stats for regression."""
    if not sem_measurements or not measurement_series:
        return None

    rows: List[List[float]] = []
    targets: List[float] = []
    measurement_ids: List[str] = []

    for measurement_id, sem_entry in sem_measurements.items():
        if measurement_id not in measurement_series:
            continue

        feature_values = []
        skip = False
        for key in feature_keys:
            value = sem_entry.get(key)
            if value is None:
                skip = True
                break
            feature_values.append(float(value))
        if skip:
            continue

        sensitivity_values = np.asarray(
            measurement_series[measurement_id]["values"],
            dtype=float
        )
        if sensitivity_values.size == 0:
            continue

        rows.append(feature_values)
        targets.append(float(target_fn(sensitivity_values)))
        measurement_ids.append(measurement_id)

    if not rows:
        return None

    features = np.asarray(rows, dtype=float)
    target = np.asarray(targets, dtype=float)
    mean = features.mean(axis=0)
    std = features.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    z_features = (features - mean) / std

    return z_features, target, measurement_ids, mean, std
