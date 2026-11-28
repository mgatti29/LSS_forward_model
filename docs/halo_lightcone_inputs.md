# Halo lightcone inputs: formats and unit conventions

`halos.save_halocatalog` builds the halo lightcone used by baryonification
(`maps.load_and_baryonify_gower_st_shells`) and by the halo IA model
(`halo_ia.iter_IA_halo_sources`, `Projects/des_decade_run/make_mocks.py`).
It accepts a simulation either in its native PKDGRAV/Gower St layout (as on
Hypatia or Tursa) or after the NERSC preprocessing. There is nothing to configure
per cluster: the input layout and the distance convention are identified from the files.
`LSS_forward_model/halo_inputs.py` implements the input handling.

## Halo snapshots

For each PKDGRAV step `S` the first existing file is used:

| Order | File | Produced by |
|---|---|---|
| 1 | `run.SSSSS.fofstats.parquet` | `convert_fof_and_cleanup` (NERSC preprocessing) |
| 2 | `fof/run.SSSSS.fofstats.0` | FoF tarball extracted on Hypatia/Tursa (`make_mocks.extract_fof`) |
| 3 | `run.SSSSS.fofstats.0` | PKDGRAV output directory |

Native binaries come in two record layouts:
- legacy: 112 bytes per halo;
- SO: 120 bytes, with `r200c` and `m200c` appended.

`read_fof_binary` picks the layout from the file size. When the size is a
multiple of both layouts, it decodes the file both ways and keeps the
layout that gives physical records: |position| ≤ 0.5 box, mass > 0. Before
this, `convert_fof_and_cleanup` always picked SO for such files.

A native binary is encoded by `encode_fof_records`, the same function
`convert_fof_and_cleanup` uses to write the parquet files. The two formats
therefore give identical halo properties:

| Quantity | Unit / encoding |
|---|---|
| halo position | comoving Mpc in [0, L), L = dBoxSize[Mpc/h] / h, = L·(rPot + rcen + 0.5) |
| `log10M` → `log_M` | uint16, 1000·log10(M_FoF / M☉), M = fMAss·ρ_crit,0·L³ |
| `rmax` → `R` | uint16, kpc |
| `log10M200c`, `r200c` | uint16, as above; all zero when the run did not compute SO |
| inertia tensor | log-encoded diagonal and normalised cross terms (`inertia_auto`, `inertia_cross`) |

SO columns (`log_M200c_SO`, `R200c_S0`) are written to the lightcone only when
the snapshots contain non-zero SO masses. All-zero columns would otherwise be
read downstream as M200c = 1 M☉; for example, `halo_ia` uses `M_SO`
whenever the column exists.

## `z_values.txt`

`cosmology.recover_shell_info` reads the columns by position and does not
convert units. It is unchanged, and so is every use of the redshift columns:
`build_shells`, density maps, lensing, NLA, κ_CMB and halo IA. Two dialects exist:

| Dialect | Writer | Header | `cmd_*` columns |
|---|---|---|---|
| package | `build_z_values_file` (NERSC) | `cmd_far(Mpc)` | χ(z) in Mpc for the run's own (w0, wa) |
| gower | Gower St team (Hypatia/Tursa) | `cmd_far(Mpc/h)` | h·χ(z) in Mpc/h, computed for **w = −1** |

In both dialects `Step`, `z_far` and `z_near` are the PKDGRAV output redshifts
from `run.log`.

The header is not used to interpret the distances, because a legacy writer labels Mpc as Mpc/h.
`identify_distance_columns` compares the `cmd_far` values with the two dialects: χ(z) in Mpc for
the run's (w0, wa) (`package`, within 0.2%) and h·χ_{w=−1}(z) (`gower`, within 1%). Anything else
is `unknown`.

## Radial edges

`save_halocatalog(..., distance_convention='auto')` takes the halos of snapshot
`S` inside [χ(z_near), χ(z_far)) in comoving Mpc for the run's cosmology. The
convention is chosen as follows:

| Columns identified as | Edges used | Effect |
|---|---|---|
| `package` (within 0.2%) | file values, verbatim | NERSC catalogues are unchanged |
| `gower` (within 1%) | recomputed from the redshift columns with `shell_comoving_distance_Mpc` | identical to the distances `build_z_values_file` would have written |
| unknown | none: raises | pass `distance_convention='from_redshift'` after checking the cosmology |

`distance_convention` can also be set explicitly to `'file_Mpc'` or `'from_redshift'`.

**Do not convert Gower distances by dividing by h.** They are also computed
for w = −1, so a run with w0 = −0.33 would still be up to 40% off.

Halo redshifts are z(χ) of each halo's actual radius, computed with CCL for the run's
(w0, wa, Σm_ν). Catalogues built before 2026-07-13 used w = −1 here.

The catalogue records how it was built in its parquet key-value metadata,
under `lss_forward_model.halo_lightcone.*`:
- `distance_convention` and `distance_columns_dialect`;
- `snapshot_formats`;
- `max_redshift` and `log10_mass_limit`.

## Scripts that read `cmd_*` directly

Some scripts outside the package read the distance columns themselves. They
must not be run on a Gower-dialect file:
- scripts that select the column by the header name `cmd_near(Mpc)` raise a `KeyError` on it;
- scripts that read it by position silently get h·χ_{w=−1} in Mpc/h.

Use `halo_inputs.shell_radial_edges_Mpc` instead.

## Tests

- `tests/test_halo_lightcone_inputs.py` writes one synthetic run in both layouts:
  - native binaries with a Gower file;
  - parquet from `convert_fof_and_cleanup` with a package file.

  It runs for ΛCDM, wCDM and w0waCDM, with and without SO records. It requires the two
  lightcones to be identical in every column, every halo to lie in its
  snapshot's redshift range, the catalogue to reach `max_redshift`, and the
  ambiguous record sizes to be resolved. On NERSC it also identifies the real
  shell tables.
- `utilities/halo_lightcone_equivalence.py NATIVE NERSC OUT` runs the same
  comparison on a real simulation available in both layouts.
