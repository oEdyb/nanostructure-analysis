import csv
import glob
from .spectra_functions import spectra_main, filter_spectra

def read_sem_measurements(csv_path):
    """Read SEM measurements CSV, excluding tilted entries."""
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['tilted_flag'] == 'True':
                continue
            label = row['label']
            data[label] = {k: float(v) for k, v in row.items()
                          if k not in ['id', 'label', 'tilted_flag']}
    return data


def filter_sem(sem_data, pattern, exclude=None):
    """Filter SEM data by pattern, similar to filter_spectra."""
    filtered = {k: v for k, v in sem_data.items() if glob.fnmatch.fnmatch(k, pattern)}
    if exclude:
        for excl_pattern in exclude:
            filtered = {k: v for k, v in filtered.items() if not glob.fnmatch.fnmatch(k, excl_pattern)}
    print(f"Found {len(filtered)} SEM entries matching '{pattern}'")
    return filtered


def match_sem_spectra(sem_data, spectra_data):
    """Match SEM and spectra data by [A-D][1-6] pattern."""
    import re
    pattern = re.compile(r'[A-D][1-6]')

    matched = {}
    for sem_key in sem_data:
        sem_match = pattern.search(sem_key)
        if not sem_match:
            continue
        sem_id = sem_match.group()

        for spec_key in spectra_data:
            spec_match = pattern.search(spec_key)
            if spec_match and spec_match.group() == sem_id:
                matched[sem_id] = {'sem': sem_data[sem_key], 'spectra': spectra_data[spec_key]}
                break

    print(f"Matched {len(matched)} SEM-spectra pairs")
    return matched


def get_average_sem_by_group(sem_data, groups=['A', 'B', 'C', 'D']):
    """Calculate average SEM parameters for each group (A, B, C, D).

    Args:
        sem_data: Dictionary of SEM measurements from read_sem_measurements
        groups: List of group letters to calculate averages for

    Returns:
        Dictionary mapping group letter to dict of averaged parameters
    """
    import re
    import numpy as np

    pattern = re.compile(r'[A-D][1-6]')

    # Group SEM data by letter
    grouped_data = {group: [] for group in groups}

    for sem_key, sem_values in sem_data.items():
        match = pattern.search(sem_key)
        if match:
            group_letter = match.group()[0]  # First character (A, B, C, or D)
            if group_letter in grouped_data:
                grouped_data[group_letter].append(sem_values)

    # Calculate averages for each group
    averages = {}
    for group, measurements in grouped_data.items():
        if not measurements:
            continue

        # Get all parameter names from first measurement
        param_names = measurements[0].keys()

        # Calculate average for each parameter
        group_avg = {}
        for param in param_names:
            values = [m[param] for m in measurements]
            group_avg[param] = np.mean(values)

        averages[group] = group_avg

    return averages


if __name__ == "__main__":
    sem_csv_path = r"Data/SEM/SEM_measurements_20251029_sample_21_gap_widths.csv"
    sem_data = read_sem_measurements(sem_csv_path)
    sem_data = filter_sem(sem_data, "*[A-D]*", exclude=["*tilted*"])

    # Example spectra data for matching
    spectra_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251024 - Sample 21 Gap Widths 24"

    spectra_data_raw, spectra_params_raw = spectra_main(spectra_path)
    spectra_data_raw, spectra_params_raw = filter_spectra(spectra_data_raw, spectra_params_raw, "[A-D]*", average=True, exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*B4*"])

    matched_data = match_sem_spectra(sem_data, spectra_data_raw)
    for key, value in matched_data.items():
        print(f"ID: {key}, SEM: {value['sem']}, Spectra shape: {value['spectra'].shape}")

