# Nanostructure Analysis

Scientific data analysis package for optical characterization of gold nanostructures. Analyzes plasmonic properties by processing and correlating spectroscopy, confocal microscopy, APD time traces, and SEM images.

## Installation

```bash
uv sync                # Install dependencies
uv pip install -e .    # Install package in editable mode
```

## Quick Start

```python
import nanostructure_analysis as nsa

# Load data
spectra_data, params = nsa.spectra_main("path/to/data")
confocal_data = nsa.confocal_main("path/to/confocal")
apd_data = nsa.apd_main("path/to/apd")

# Filter by pattern
filtered_data, filtered_params = nsa.filter_spectra(spectra_data, params, "*box1*")

# Access configuration
print(nsa.config.DEFAULT_POWER_FACTOR)   # 50.0 mW/V
nsa.config.DEFAULT_POWER_FACTOR = 55.0   # Modify if needed
```

## Running Analysis

Analysis scripts are in `scripts/`:

```bash
python scripts/work_sample_21_all_24.py
python scripts/work_Au_200nm_vs_100nm.py
```

Plots are saved to `plots/` with subdirectories per script/sample.

## Project Structure

```
src/nanostructure_analysis/  # Package modules
scripts/                      # Analysis scripts (work_*.py)
Data/                         # Raw data (Spectra, Confocal, SEM, APD)
plots/                        # Generated plots
cache/                        # Cached data for faster loads
```

## Dependencies

See `pyproject.toml` for full dependency list. Requires Python >= 3.12.

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed architecture and development guidelines
