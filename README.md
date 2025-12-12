# Nanostructure Analysis

Scientific data analysis toolkit for optical characterization of gold nanostructures. Processes and correlates spectroscopy, confocal microscopy, APD time traces, and SEM images.

## Project Structure

```
nanostructure-analysis/
├── src/                      # Package source code
│   └── nanostructure_analysis/
│       ├── __init__.py
│       ├── apd_functions.py
│       ├── apd_plotting_functions.py
│       ├── confocal_functions.py
│       ├── confocal_plotting_functions.py
│       ├── loading_functions.py
│       ├── sem_functions.py
│       ├── sem_plotting_functions.py
│       ├── spectra_functions.py
│       └── spectra_plotting_functions.py
├── scripts/                  # Analysis scripts (46 work_*.py files)
├── Data/                     # Raw data (not in git)
│   ├── Spectra/              # Spectroscopy measurements
│   ├── Confocal/             # Confocal microscopy images
│   ├── SEM/                  # SEM analysis outputs
│   ├── APD/                  # Raw APD traces (large, on network)
│   └── apd_downsampled/      # Downsampled APD traces (local)
├── plots/                    # Generated plots (not in git)
├── cache/                    # Cached data for faster loads (not in git)
├── misc/                     # Miscellaneous files
├── archive/                  # Old/unused files & experimental code
├── main.py                   # Package demo
├── pyproject.toml            # Package configuration
└── README.md                 # This file
```

## Installation

1. Install dependencies using uv:
```bash
uv sync
```

2. Install package in editable mode:
```bash
uv pip install -e .
```

## Usage

### Import the package

```python
import nanostructure_analysis as nsa

# Load and analyze data
spectra_data, params = nsa.spectra_main("path/to/data")
filtered_data, filtered_params = nsa.filter_spectra(spectra_data, params, "*box1*")

# Load confocal data
confocal_data = nsa.confocal_main("path/to/confocal")
filtered_confocal = nsa.filter_confocal(confocal_data, "*box1*")

# Load APD data (downsampled traces from Data/apd_downsampled/)
apd_data = nsa.apd_main("path/to/apd")
```

### Access configuration

All paths and parameters are centralized in `config.py`:

```python
import nanostructure_analysis as nsa

# View configuration
print(nsa.config.APD_DOWNSAMPLED_DIR)    # "Data/apd_downsampled"
print(nsa.config.DEFAULT_POWER_FACTOR)   # 50.0 mW/V
print(nsa.config.CACHE_DIR)              # "cache"

# Modify at runtime (if needed)
nsa.config.DEFAULT_POWER_FACTOR = 55.0

# See config_example.py for full usage
```

### Run analysis scripts

```bash
python scripts/work_sample_21_all_24.py
python scripts/work_Au_200nm_vs_100nm.py
```

## Data Organization

- **Raw data** lives in `Data/` subdirectories (Spectra, Confocal, SEM, APD)
- **Large APD traces** are stored on network drive and downsampled to `Data/apd_downsampled/`
- **Cached data** in `cache/` speeds up repeated loads (can be deleted to force reload)
- **Plots** are saved to `plots/` with subdirectories per script/sample

## Development

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines and architecture documentation.

## Dependencies

- numpy >= 2.3.5
- scipy >= 1.15.0
- matplotlib >= 3.10.0
- pandas >= 2.2.0
- scikit-image >= 0.24.0
