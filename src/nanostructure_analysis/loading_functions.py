from .apd_functions import *
from .confocal_functions import *
from .spectra_functions import *
import os
import pickle

# ========== Data Paths ==========

# 100nm samples (6 & 7)
paths_100nm = {
    'apd': {
        '6': r"Data\APD\2025.06.11 - Sample 6 Power Threshold",
        '7': r"Data\APD\2025.06.03 - Power Threshold Sample 7 box 4"
    },
    'confocal': {
        '6': r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.06.11 - Sample 6 Power Threshold",
        '7': r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.06.03 - Power Threshold Sample 7 box 4"
    },
    'spectra': r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250811 - sample6"
}

# 200nm sample (13)
paths_200nm = {
    'apd': {
        'normal': r"Data\APD\2025.08.21 - Sample 13 Power Threshold",
        'pt': r"Data\APD\2025.09.02 - Sample 13 PT high power"
    },
    'confocal': {
        'normal': r"Data\Confocal\2025.08.21 - Sample 13 Power Threshold box1",
        'pt': r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\2025.09.02 - Sample 13 before after break"
    },
    'spectra': {
        'before': r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250812 - sample13",
        'after': r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250821 - sample13 - after"
    }
}

# ========== Utility Functions ==========

def load_cached(path, loader):
    """Load data with cache."""
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = path.replace("/", "_").replace("\\", "_").replace(":", "").replace(" ", "_")
    cache_file = os.path.join(cache_dir, f"cache_{cache_key}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    data = loader(path)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    return data

# ========== Main Loading Function ==========

def load_samples_6_and_13():
    """Load and filter data for 100nm (samples 6&7) and 200nm (sample 13)."""

    # Load 100nm APD data
    apd_6 = apd_load_main(paths_100nm['apd']['6'])
    apd_6 = filter_apd(*apd_6, "*[A-D][1-6]*")
    apd_7 = apd_load_main(paths_100nm['apd']['7'])
    apd_7 = filter_apd(*apd_7, "*[A-D][1-6]*", exclude=["C4"])

    # Load 200nm APD data (combine normal + PT)
    apd_13_normal = apd_load_main(paths_200nm['apd']['normal'])
    apd_13_pt = apd_load_main(paths_200nm['apd']['pt'])
    apd_13_combined = {**apd_13_normal[0], **apd_13_pt[0]}
    apd_13_monitor_combined = {**apd_13_normal[1], **apd_13_pt[1]}
    apd_13_params_combined = {**apd_13_normal[2], **apd_13_pt[2]}
    apd_13_box1 = filter_apd(apd_13_combined, apd_13_monitor_combined, apd_13_params_combined, "*box1*")
    apd_13_box4 = filter_apd(apd_13_combined, apd_13_monitor_combined, apd_13_params_combined, "*box4*[!_D4_*]*")

    # Load confocal data with cache
    conf_6 = load_cached(paths_100nm['confocal']['6'], confocal_main)
    conf_7 = load_cached(paths_100nm['confocal']['7'], confocal_main)
    conf_13_normal = load_cached(paths_200nm['confocal']['normal'], confocal_main)
    conf_13_pt = load_cached(paths_200nm['confocal']['pt'], confocal_main)

    # Combine 200nm confocal
    conf_13_combined = (
        {**conf_13_normal[0], **conf_13_pt[0]},
        {**conf_13_normal[1], **conf_13_pt[1]},
        {**conf_13_normal[2], **conf_13_pt[2]},
        {**conf_13_normal[3], **conf_13_pt[3]},
        {**conf_13_normal[4], **conf_13_pt[4]}
    )

    # Filter confocal
    conf_6_before = filter_confocal(conf_6, "*", exclude=["after"])
    conf_6_after = filter_confocal(conf_6, "*after*")
    conf_7_before = filter_confocal(conf_7, "*", exclude=["after"])
    conf_7_after = filter_confocal(conf_7, "*after*")
    conf_13_box1_before = filter_confocal(conf_13_combined, "*box1*", exclude=["after", "C2"])
    conf_13_box1_after = filter_confocal(conf_13_combined, "*box1*after*", exclude=["C2"])
    conf_13_box4_before = filter_confocal(conf_13_combined, "*box4*", exclude=["after", "C2"])
    conf_13_box4_after = filter_confocal(conf_13_combined, "*box4*after*", exclude=["C2"])

    # Load spectra with cache
    spec_100nm = load_cached(paths_100nm['spectra'], lambda p: spectra_main(p))
    spec_200nm_before = load_cached(paths_200nm['spectra']['before'], lambda p: spectra_main(p))
    spec_200nm_after = load_cached(paths_200nm['spectra']['after'], lambda p: spectra_main(p))

    # Return structured data
    return {
        '100nm': {
            'apd': {'6': apd_6, '7': apd_7},
            'confocal': {'6': (conf_6_before, conf_6_after), '7': (conf_7_before, conf_7_after)},
            'spectra': spec_100nm
        },
        '200nm': {
            'apd': {'box1': apd_13_box1, 'box4': apd_13_box4},
            'confocal': {
                'box1': (conf_13_box1_before, conf_13_box1_after),
                'box4': (conf_13_box4_before, conf_13_box4_after)
            },
            'spectra': {'before': spec_200nm_before, 'after': spec_200nm_after}
        }
    }