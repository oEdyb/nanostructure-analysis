# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scientific data analysis package for optical characterization of gold nanostructures. Processes and correlates multiple measurement modalities: spectroscopy, confocal microscopy, APD (avalanche photodiode) time traces, and SEM (scanning electron microscopy) images. Analyzes plasmonic properties of Au nanoparticles (100nm, 200nm disks, rods, gaps) and correlates optical responses with geometry.

## Environment & Installation

**Package manager**: uv (modern Python package manager)

```bash
# Install dependencies
uv sync

# Install package in editable mode
uv pip install -e .

# Run with venv python
.venv/bin/python script.py
```

**Dependencies**: numpy, scipy, matplotlib, pandas, scikit-image (see `pyproject.toml`)

## Running Analysis Scripts

Analysis scripts are in `scripts/` directory (46 scripts following `work_*.py` pattern):

```bash
# From project root
python scripts/work_sample_21_all_24.py
python scripts/work_Au_200nm_vs_100nm.py
```

**Output**: Plots saved to `plots/` with subdirectories per script/sample (e.g., `plots/sample_21_all_24/`)

**Validation**: No test suite. Validate changes by re-running affected scripts and inspecting plots.

## Package Architecture

### Package Structure

```
src/nanostructure_analysis/
├── config.py                    # Centralized configuration (32 constants)
├── __init__.py                  # Package exports
├── apd_functions.py             # APD data loading
├── confocal_functions.py        # Confocal microscopy data loading
├── spectra_functions.py         # Spectroscopy data loading
├── sem_functions.py             # SEM data loading
├── *_plotting_functions.py      # Plotting per modality
└── loading_functions.py         # Multi-sample loaders
```

### Data Structures

The package uses **NamedTuples** for type-safe data containers:

- `APDData(transmission, monitor, params)` - APD measurements
- `ConfocalData(images, apd_traces, monitor_traces, xy_coords, z_scans)` - Confocal scans
- `ConfocalAnalysisResult(popt, sigma_x, sigma_y, snr_3x3, ...)` - Analysis results
- `PSFParameters` - Point spread function fit results
- `TraceStatistics` - APD trace statistics

### Centralized Configuration

**All paths and constants** are in `config.py`:

```python
import nanostructure_analysis as nsa

# Access configuration
nsa.config.APD_DOWNSAMPLED_DIR    # "Data/apd_downsampled"
nsa.config.DEFAULT_POWER_FACTOR   # 50.0 mW/V
nsa.config.CACHE_DIR               # "cache"
nsa.config.GRID_POSITION_PATTERN  # r'[A-D][1-6]'

# Modify at runtime if needed
nsa.config.DEFAULT_POWER_FACTOR = 55.0
```

Key config categories:
- Directory paths (DATA_DIR, CACHE_DIR, APD_DOWNSAMPLED_DIR)
- APD parameters (downsample factor, power calibration)
- Confocal parameters (PSF fitting, SNR calculation)
- Plotting defaults (figure sizes, fonts, colors)
- Network paths (for large raw data on `\\AMIPC045962\Cache (D)\daily_data`)
- Analysis parameters (wavelength cutoffs, normalization points)

### Core Workflow Pattern

All data loaders follow: **Load → Filter → Process → Plot**

```python
import nanostructure_analysis as nsa

# Load all data from directory
data, params = nsa.spectra_main(path)

# Filter by glob patterns
filtered_data, filtered_params = nsa.filter_spectra(
    data, params, "*box1*", exclude=["*bkg*"]
)

# Process (normalize, analyze, etc.)
normalized = nsa.normalize_spectra(filtered_data, reference_data)

# Plot
from nanostructure_analysis.spectra_plotting_functions import plot_spectra
plot_spectra(normalized, params)
```

### Dictionary-Based Data Model

All functions use `{label: data}` dictionaries:
- **Label**: Filename-derived key (first 2 timestamp parts stripped)
  - Example: `"20250101_143022_box1_A3.dat"` → `"box1_A3"`
- **Data**: numpy array (spectra: Nx2, confocal: 2D/3D images)
- **Filtering**: Via `glob.fnmatch.fnmatch(label, pattern)`

### Caching System

Automatic caching for slow operations (confocal, large datasets):

```python
# config.py provides cache helpers
cache_path = nsa.config.get_cache_path('confocal', 'sample_21')

# Functions use caching automatically
confocal_data = nsa.confocal_main(path)  # Cached on first load
```

Cache files: `cache/{type}_{sanitized_path}.pkl`

**Clear cache**: Delete `cache/*.pkl` to force reload

### APD Data Pipeline

Raw APD traces are large (network storage). Two-stage process:

1. **Preprocess** (first time only):
   ```python
   from nanostructure_analysis.apd_functions import preprocess_apd_directory
   preprocess_apd_directory(network_path, downsample_factor=100)
   # Saves to Data/apd_downsampled/
   ```

2. **Load** (subsequent uses):
   ```python
   apd_data = nsa.apd_main(path)  # Loads from Data/apd_downsampled/
   ```

**Power calibration**: Automatically computes `params['power'] = mean(monitor) * power_factor`

## Data Organization

```
Data/
├── Spectra/              # *_spectrum.dat + *_params.txt
├── Confocal/             # *_image.npy, *_xy_coords.npy, *_z_scan.npy
├── SEM/                  # CSV measurement files
├── APD/                  # Raw traces (large, network storage)
└── apd_downsampled/      # Downsampled traces (local, fast)

cache/                    # Pickle files for fast reloads
plots/                    # Generated plots (subdirs per script)
scripts/                  # Analysis scripts (work_*.py)
```

**Network storage**: `\\AMIPC045962\Cache (D)\daily_data\` - Large raw datasets
- Use raw strings: `r"\\AMIPC045962\..."`
- Configured in `config.NETWORK_DATA_PATH`

## Common Patterns

### Pattern Matching

Grid layout: 24-spot grid `[A-D][1-6]` (4 rows × 6 columns)

```python
# Sample positions
filter_spectra(data, params, "*[A-D][1-6]*")

# Box regions
filter_confocal(confocal_data, "*box1*", exclude=["after", "C2"])

# SEM (exclude tilted)
filter_sem(sem_data, "*[A-D]*", exclude=["*tilted*"])
```

### Combining Datasets

Before/after experiments or multi-sample comparisons:

```python
# Merge dictionaries
combined_data = {**before_data, **after_data}
combined_params = {**before_params, **after_params}

# Filter combined set
filtered = filter_spectra(combined_data, combined_params, "*box1*")
```

### Confocal Analysis

Returns 5-tuple of dicts: `(images, apd_traces, monitor_traces, xy_coords, z_scans)`

```python
confocal_data = nsa.confocal_main(path)
analysis_results = nsa.analyze_confocal(confocal_data)

# Access results
for label, result in analysis_results.items():
    print(f"{label}: SNR={result.snr_3x3:.2f}, sigma={result.sigma_x:.2f}px")
```

**Old vs new format**: Old `*_confocal_traces.npy`, new `*_confocal_apd_traces.npy` + `*_monitor_traces.npy`

### Spectra Processing

```python
from nanostructure_analysis import spectra_functions as sf

# Load
data, params = nsa.spectra_main(path)

# Normalize (baseline + reference)
normalized = sf.normalize_spectra_baseline(data, baseline, reference, lam=1e7)

# Or reference-only
normalized = sf.normalize_spectra(data, reference)

# Cosmic ray filtering
filtered_data, filtered_params = nsa.filter_spectra(
    data, params, pattern, average=True  # Removes >2σ outliers
)
```

**Tunable parameters** (in config.py):
- `SPECTRA_CUTOFF_WAVELENGTH` = 950 nm
- `SPECTRA_NORMALIZATION_WAVELENGTH` = 740 nm
- `SPECTRA_SAVGOL_WINDOW` = 31
- `COSMIC_RAY_SIGMA_THRESHOLD` = 2.0

### SEM Correlation

Match SEM measurements to optical data by grid position:

```python
from nanostructure_analysis.sem_functions import read_sem_measurements, match_sem_spectra

sem_data = read_sem_measurements("Data/SEM/measurements.csv")
spectra_data, _ = nsa.spectra_main(spectra_path)

matched = match_sem_spectra(sem_data, spectra_data)
# Returns {id: {'sem': {...}, 'spectra': ndarray}}
```

## Important Conventions

**Filename handling**:
- First 2 underscore-separated parts are timestamps → stripped
- Duplicate names get `_1`, `_2` suffixes
- Case-sensitive pattern matching

**Git exclusions** (in `.gitignore`):
- `Data/`, `cache/`, `plots/`
- `*.npy`, `*.pkl`
- `archive/` (old/experimental code)

**Sample naming**: Include acquisition date
- Example: `20251024 - Sample 21 Gap Widths 24`

## Adding New Analysis

1. Create script in `scripts/work_description.py`
2. Import package: `import nanostructure_analysis as nsa`
3. Load data using `nsa.{modality}_main(path)`
4. Filter with glob patterns
5. Save plots to `plots/description/` subdirectory
6. Use `if __name__ == "__main__"` guard

Example skeleton:

```python
import nanostructure_analysis as nsa
import matplotlib.pyplot as plt
import os

output_dir = "plots/my_analysis"
os.makedirs(output_dir, exist_ok=True)

# Load data
spectra_data, params = nsa.spectra_main(r"Data\Spectra\sample_path")
confocal_data = nsa.confocal_main(r"Data\Confocal\sample_path")

# Filter
filtered_spectra, filtered_params = nsa.filter_spectra(
    spectra_data, params, "*box1*[A-D][1-6]*", exclude=["bkg"]
)

# Analyze
analysis = nsa.analyze_confocal(confocal_data)

# Plot and save
plt.figure()
# ... plotting code ...
plt.savefig(f"{output_dir}/result.png", dpi=300, bbox_inches='tight')
```

## Module Import Patterns

Analysis scripts use **absolute imports from package**:

```python
# Preferred (with installed package)
import nanostructure_analysis as nsa
from nanostructure_analysis.spectra_functions import normalize_spectra

# Legacy pattern (in older scripts)
from spectra_functions import *  # Works if running from root
```

All modules use **relative imports internally**:

```python
# Inside package modules
from . import config
from .apd_functions import apd_main
```

## Performance Notes

- **Confocal loading**: Slowest (large .npy files). Always uses caching.
- **APD downsampling**: First-time only. Factor 100 reduces file size 100×.
- **Cache invalidation**: Delete `cache/*.pkl` if source data changes.
- **Network access**: Raw data on `\\AMIPC045962\...` requires VPN/network access.
