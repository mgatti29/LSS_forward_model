# DES / DECADE / BFD shear mocks: halo IA + source clustering v2

`make_mocks.py` makes the DES Y6 metadetection, DES Y6 BFD, DECADE NGC and DECADE SGC shear mocks (plus noiseless
kappa_CMB, NSIDE 2048) for one Gower St simulation: per baryon draw (`delta_b_1024_{0,1}.npy`) and rotation, one IA draw
shared by the four samples; per sample an n(z)/m draw, a source-clustering draw and the map maker
(`lensing.make_WL_sample`, `bfd_utils.make_BFD_sample`).  Resumable; writes `mocks_done.json` with the package git hash.

## Contents

| path | content |
|---|---|
| `make_mocks.py` | driver (one simulation per call) |
| `PRIORS.md` | adopted IA priors and rationale |
| `config/ia_priors.json` | IA priors: halo IA (default) and NLA |
| `config/sc_priors.json` | source-clustering priors, calibration file per sample |
| `config/samples.json` | the four samples: catalogue, n(z)/m model, SC calibration, map maker |
| `config/large_inputs.json` | catalogues and n(z)/m realisations staged outside git (file name, size, sha256, NERSC source) |
| `data/` | SC v2 calibrations (`SC_v2_*_nside1024_sim1.json`) and mean n(z) of each sample |
| `hypatia/` | `smoke_test.sh`, `make_manifest.sh`, `submit_mocks.sh` |

## Model choices (package defaults, des_decade_v0.3)

* Halo IA: virtual-halo model, local cap s = min(1, sqrt(var_noise / var_IA)) per pixel, IA term divided by the
  source-clustering density D (`ia_over_D=True`).  NLA: pure NLA (`nla_over_D=False`, no b delta term on the IA);
  source clustering acts on the gravitational shear and on the noise.
* Source clustering v2: Poisson noise model, bias per bin with a within-bin z slope.
* BFD: count-normalised estimator, g = -(1/N) Rbar^-1 Q, N >= 3 mask (`cat_BFD_N3`, `mask_BFD_N3`, `meta_BFD_N3`).
* Rotations 2, 3 mirror the sky; BFD maps are in the catalogue frame, the other samples in the simulation frame.

## Large inputs

`DES_DECADE_INPUTS` (default `/global/cfs/cdirs/m5099/des_decade_run_inputs`, symlinks at NERSC) must contain the files
of `config/large_inputs.json`, except `cat_BFD` (unused).  On Hypatia: `rsync -L` to `/share/lustre/mgatti/des_decade_run_inputs`.

## Hypatia

* `sbatch -p CORES64 --ntasks=1 --cpus-per-task=64 --mem=480G hypatia/smoke_test.sh SUITE RUN_DIR`: one simulation,
  b0 r0, halo IA, kappa_CMB into a scratch directory; checks the staged inputs first.
* `hypatia/make_manifest.sh SUITE N`: simulations without `mocks_done.json` that have the inputs, in chunks of N.
* `hypatia/submit_mocks.sh SUITE MANIFEST`: one node per job, one task per simulation.  Peak RSS 170 GB per
  simulation -> 2 per 480 GB node (`--ntasks=2 --cpus-per-task=32 --mem=480G`); ~2.3 h for 2 baryon draws x 2 rotations x 2 IA models.

Validation and prior studies (tests, end-to-end checks, prior tests, data preparation) are kept outside the repository.
