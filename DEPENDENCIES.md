# Dependency and portability proposal

## Summary

The existing environment files are snapshots of broad NERSC analysis environments, not dependency specifications for LSS Forward Model. They include machine-learning, plotting, inference, GPU, and unrelated analysis packages. The proposed `environment.yml` contains only dependencies imported by the package plus the tools needed to run the main notebooks.

This is a **tested portable environment**, not yet a lock file. It was created from scratch on NERSC/Linux with Python 3.11 and NumPy 1.26, without using packages from the author's home environment.

## Dependency groups

| Layer | Packages | Why they are needed |
|---|---|---|
| Numerical core | NumPy, SciPy, Pandas | Arrays, interpolation, optimization, and catalogs |
| Spherical maps | healpy | HEALPix maps, harmonic transforms, smoothing, and rotations |
| Cosmology | CAMB, CCL, Astropy, `cosmology` | Background quantities, power spectra, distances, and lensing geometry |
| Lightcones/lensing | GLASS | Radial shells, missing-shell generation, convergence, and shear |
| Halo modeling | Colossus | Concentrations and halo-mass conversions |
| Baryons and tSZ | BaryonForge | Baryonification and pressure/Compton-y painting |
| File formats | HDF5/h5py, PyArrow | HDF5, Parquet, and simulation products |
| Utilities | frogress | Progress reporting used throughout the modules |
| HPC/notebooks | MPI4Py, Matplotlib, Jupyter | Parallel workflows and examples; not required by every library call |
| Survey masks | healsparse | Used by survey-preparation notebooks, not the package modules |

## Important architectural limitation

The proposed `pyproject.toml` lists a relatively broad base environment because the current modules import optional dependencies at import time:

- `maps.py` imports GLASS, CCL, CAMB, `cosmology`, and BaryonForge.
- `halos.py` imports CCL, Colossus, BaryonForge, and Astropy.
- `PKDGRAV_and_postprocessing_utilities.py` imports PyArrow.
- `LSS_forward_model/__init__.py` eagerly imports all major modules.

Consequently, `import LSS_forward_model` currently requires almost the whole stack. The optional dependency groups in `pyproject.toml` express the desired packaging direction, but a truly small core installation requires a code refactor:

1. Stop eagerly importing every submodule in `__init__.py`.
2. Move BaryonForge imports inside baryonification/tSZ functions.
3. Move PyArrow imports inside PKDGRAV conversion functions.
4. Keep Colossus local to halo conversion functions.
5. Raise clear `ImportError` messages explaining which extra to install.
6. Then move those packages into extras such as `baryons`, `simulation-io`, and `halos`.

## Packages deliberately omitted

The large NERSC YAML files contain many packages not imported by the package or canonical forward-model notebooks, including TensorFlow/Keras, Ray, SBI/normalizing-flow tooling, plotting utilities, coverage services, and numerous analysis-specific libraries. They should live in project-specific downstream environments rather than the LSS Forward Model environment.

Notebook-only packages such as BFD, PyWPH, Orphics, CosmoGrid utilities, and survey collaboration software are also omitted from the general environment. They belong in per-example add-on files because they are unavailable or unnecessary for most users.

EuclidEmulator2 is exposed as an optional `emulator` extra. `LimberTheory` already falls back to CAMB/MEAD when the emulator is unavailable, so it should not be forced into the portable baseline.

## External and simulation-specific dependencies

Some notebooks depend on software or data that cannot be made portable through a general package file:

- GLASS is pinned to commit `ba9b1b5`, the NumPy-1-compatible revision recorded by the older working environment. The current moving `gowerst` branch requires NumPy 2 and conflicts with BaryonForge.
- BaryonForge is pinned to commit `25e2e002201b9f47aaec1c309538bb5291b6deec` for reproducibility and requires NumPy 1.x.
- BaryonForge imports Joblib without declaring it in its package metadata, so the portable environment includes Joblib explicitly.
- CosmoGrid notebooks use `cosmogridv11` and `cosmolopy`.
- DES/BFD workflows use collaboration-specific BFD code and catalogs.
- Survey masks, redshift distributions, source catalogs, simulation shells, and halo catalogs are external data products.

These requirements should be documented beside the relevant notebook rather than added to every installation.

## Validated installation test

From a clean Linux machine or container:

```bash
conda env create -f environment.yml
conda activate lss-forward-model
python -c "import LSS_forward_model"
python -c "from LSS_forward_model import cosmology, maps, halos, lensing"
```

Then run a small-data smoke test covering:

1. Reading simulation metadata and shell geometry.
2. Loading or constructing a low-NSIDE density shell.
3. Computing Born-approximation convergence and shear.
4. Loading a small halo catalog.
5. Baryonifying one low-resolution shell.
6. Painting one low-resolution tSZ shell.
7. Producing a small survey-like noisy weak-lensing map.

The clean-environment validation passed for package imports, all BaryonForge APIs referenced by the code, and a finite low-NSIDE Born-lensing calculation. The full baryonification, tSZ, simulation-I/O, and survey-mock workflows still need small portable fixtures before they can become automated integration tests.

The test also exposed one compatibility requirement now included in `maps.py`: the pinned GLASS revision requires explicit imports of `glass.fields`, `glass.lensing`, and `glass.shells`, and field-generation functions live under `glass.fields`.

For a NERSC installation that keeps both the environment and caches off `$HOME`, create a scratch prefix and redirect the caches for the installation command:

```bash
export CONDA_PKGS_DIRS=/path/in/scratch/conda-pkgs
export PIP_CACHE_DIR=/path/in/scratch/pip-cache
conda env create --prefix /path/in/scratch/lss-forward-model -f environment.yml
conda activate /path/in/scratch/lss-forward-model
```

Exact transitive versions can subsequently be frozen with `conda-lock` or an exported explicit specification.

## Open questions before publication

- Does the code require the GLASS `gowerst` branch, or can it use a released GLASS version?
- Which BaryonForge release/API is the supported baseline?
- Is Python 3.10 also supported, or should the first public environment promise only 3.11?
- Which default parameter and correction files must be packaged under `LSS_forward_model/data/`?
- Which example is the canonical portable smoke test?
