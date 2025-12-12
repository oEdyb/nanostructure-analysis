# Repository Guidelines

## Project Structure & Module Organization
- Core package lives in `src/nanostructure_analysis/` (data loaders, plotting, config). `config.py` centralizes paths and analysis constants.
- Analysis entry points are `scripts/work_*.py` (46 scripts); run them from the repo root. `main.py` offers a lightweight demo.
- `Data/` holds raw inputs (Spectra, Confocal, SEM, APD, apd_downsampled); `plots/` stores generated figures; `cache/` stores pickled caches to speed reloads. These directories are untracked or large—avoid committing outputs.
- `archive/` and `misc/` contain exploratory or legacy code; prefer updating modules in `src/` before touching archived files.

## Build, Test, and Development Commands
- `uv sync` — install project dependencies into `.venv`.
- `uv pip install -e .` — editable install for local development.
- `python main.py` — quick sanity check that imports and key paths work.
- `python scripts/work_sample_21_all_24.py` (or any `work_*.py`) — run an analysis script; outputs plots under `plots/<script_name>/`.
- Clear caches when debugging stale results: `rm cache/*.pkl`.

## Coding Style & Naming Conventions
- Python 3.12, 4-space indentation, limit line length near 100 chars. Keep imports standard → third-party → local.
- Use `snake_case` for functions/variables, `CapWords` for classes/NamedTuples. Mirror existing module naming (`*_functions.py`, `*_plotting_functions.py`).
- Prefer numpy/pandas vectorization over Python loops for data transforms. Keep plotting code in the plotting modules, not loaders.
- Configuration: read shared defaults from `nanostructure_analysis.config` instead of hardcoding paths or constants.

## Testing Guidelines
- No formal test suite yet. Validate changes by running the smallest relevant `work_*.py` script and confirming generated plots and cache contents are sensible.
- When adding new functions, include a minimal reproducible usage snippet in the docstring or script comment, and consider adding a lightweight script under `scripts/` for manual regression.

## Commit & Pull Request Guidelines
- Commits: one logical change per commit; concise, imperative subject (`Add SEM cache helpers`). Include brief context in the body when touching data-processing logic.
- Pull requests: describe intent, affected modules, commands run (`python scripts/...`), and any new data paths. Attach representative plot(s) or cache notes if behavior changes. Reference related issues or datasets when applicable.

## Data Handling & Caching
- Keep raw data and large outputs out of git (`Data/`, `cache/`, `plots/`). Use relative paths that work from the repo root and leverage `config.py` helpers for cache paths.
- If working with network-stored APD traces, preprocess once into `Data/apd_downsampled/` before running loaders. Document any new storage locations in PRs.
