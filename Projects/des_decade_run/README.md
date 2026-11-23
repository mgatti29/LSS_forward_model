# DES / DECADE / BFD mock production with halo IA + source clustering v2

Goal: remake the DES Y6 (metadetection and BFD), DECADE NGC and DECADE SGC shear mocks on the Gower St runs3A
suite with

* the virtual-halo IA model (`LSS_forward_model.halo_ia`, `IA_model='halo'`; NLA kept for comparison),
* source clustering v2 (`noise_model='poisson'`, bias per bin + mild within-bin z slope),
* the existing baryonified density shells (`delta_b_1024_{0,1}.npy`, NOT redone),
* final n(z) and shear biases (paired n(z)/m realisations for mdet and BFD).

Plan: test on a few Gower LCDM sims here (NLA vs halo, same density / rotations / noise seeds), then move to
Hypatia.  Hypatia suites: runsT, U, V, W (wCDM), X (LCDM), 3A, 3B (w0wa).  First ONE production round on the
LCDM suite (X) and a full analysis of the real data (DES mdet + BFD, DECADE NGC/SGC, ACT CMB lensing) to check
for surprises; then the other suites.

CMB lensing (in progress): noiseless kappa_CMB (NSIDE 2048, L <= 4000) per simulation and baryon realisation
for the ACT forward model; Born sum of native dark-matter-only shells + baryon difference from the existing
NSIDE-1024 fields + high-z extension (LSS_forward_model.coshirex); tests in `kcmb_test/`.

## Layout

| path | content |
|---|---|
| `PRIORS.md` | adopted IA priors, rationale and evidence |
| `config/ia_priors.json` | IA priors (halo + NLA comparison), noise-budget rejection rule |
| `config/sc_priors.json` | source-clustering priors and calibration files |
| `config/samples.json` | the four samples: catalogue, n(z)/m model, SC calibration, mock maker |
| `config/large_inputs.json` | manifest (size, sha256) of the large inputs staged outside git |
| `data/` | SC v2 calibrations (DES, NGC, SGC, BFD), mean n(z) of each sample |
| `data_prep/` | BFD light catalogue builder, BFD SC calibration, manifest script |
| `prior_tests/` | IA prior tests (TNG300 2pt envelope, Gower noise budget, Fisher, sign check) |
| `tests/` | `test_bfd_v2.py` (BFD legacy bit-identity + Poisson/halo smoke test) |

Large inputs are staged as symlinks in `/global/cfs/cdirs/m5099/des_decade_run_inputs/` (catalogues of the
four samples, 20k paired n(z)/m realisations for mdet and BFD).  Copy with `rsync -L`, verify with
`data_prep/make_manifest.py --check` (`DES_DECADE_INPUTS` points to the copy).

## Package changes made for this run (LSS_forward_model, uncommitted)

* `halo_ia.occupation(cap=0.98)`: if a shell cannot host the requested centrals (z < 0.02, small f_c), cap the
  hosted count and dilute the rest (no IA) instead of raising; feasible shells unchanged.
* `bfd_utils.make_BFD_sample`: `noise_model` ('poisson' default; 'inverse_density' = previous, bit-identical),
  SC v2 corrections with the bias slope, `IA_model` ('deltaNLA' | 'NLA' | 'halo', `halo_IA`), halo-IA white
  variance removed from the q noise, `bfd_expected_counts`.  In the Poisson path the injected shear is
  normalised by D (sum n (1 + b delta) gamma / D, as make_WL_sample; the previous BFD path does not divide by D).
  NB: `bfd_utils.py` is not in the GitHub repository yet.
* `halo_ia.make_halo_IA_numerator`: marks multiplied by `PACKAGE_IA_SIGN = -1`, so the numerator is in the
  lensing convention of make_WL_sample / make_BFD_sample and positive A_c, A_s are physical alignments (the halo
  shapes had the opposite E-mode sign; confirmed on TNG300 and on the Gower production path,
  `prior_tests/sign_check.py`, `prior_tests/gower_gi_calibration.py`).  Halo-IA mocks made before this change
  have the anti-physical sign (noise-only tests unaffected).

* 2026-11-18: local halo-IA cap is the default (`ia_cap_alpha=1.0` in `make_WL_sample` and `make_BFD_sample`,
  only with `IA_model='halo'`): the IA term is scaled by s = min(1, sqrt(var_noise / var_IA)) per pixel with the
  realised per-pixel self term (`make_WL_sample(ia_noise_nside=None)`, `make_BFD_sample(ia_self_nside=None)`;
  previous default 64).  `ia_cap_alpha=None` (+ NSIDE-64 self term) reproduces the previous code bit for bit
  (`tests/test_ia_cap.py`).  The noise-budget rejection of the IA marks is disabled (`config/ia_priors.json`,
  `max_fraction = 1`); the capped-pixel fraction per bin is still saved as `ia_noise_fraction`.  Validation:
  `Projects/ia_sc_validation/RESULTS.md`.  Backups: `/pscratch/sd/m/mgatti/IA_non_gaussian/tng300/tng_sc/ia_insert/code_backup_20261118/`.

* 2026-11-20: halo-IA prior revised (NLA-spanning, low shot noise): a_c = f_c A_c U[-0.2, 0.5] (A_c derived),
  eta_c U[-5, 0], A_s U[-0.05, 0.05], eta_s U[-0.5, 0.5], f_c U[0.7, 1], f_sat U[0.3, 0.8]; see `PRIORS.md`.
  `make_mocks.halo_draw` draws the population then the marks (A_c = a_c / f_c, a_c stored in IA_params).

* 2026-11-20: BFD switched to the count-normalised estimator (`make_BFD_sample(estimator='count', Rbar=..., nmin=3)`,
  `bfd_utils.make_bfd_map_count`): g_p = -(1/N_p) Rbar^-1 Q_p with the global full-catalogue response (N3 metadata,
  unrotated), fixed mask N_p >= 3 (N3 catalogue `cat_BFD_N3`), no |g| cut, library sign (correlates positively with
  mdet), unit shear response (map-level injection dm gamma), exact noise variance V cR / N^2, analytic SC normalisation
  (cv per mock; the ML-estimator `SC_v2_BFD` calibration is no longer used), halo-IA cap on that noise.  Data, signal
  and noise maps share one mask (fixes e1n stored on the signal mask).  Validation of the estimator:
  `/global/cfs/cdirs/m5099/des_decade_run_outputs/bfd_cnt_validation/REPORT.md`; implementation test
  `tests/test_bfd_count.py`.  `estimator='ml'` (default) is the previous map, bit-identical.  All earlier BFD mocks are
  superseded.

* 2026-11-23: halo IA divided by the source-clustering density (user decision): `ia_over_D=True` (default) in
  `make_WL_sample` and `make_BFD_sample`: IA term s I/D, self term 0.5 numerator_sq / D^2 per simulation pixel, D =
  sum_i n_i (1 + b_i delta_i) floored at 1e-3 (the validated "I/D" operator of `Projects/ia_sc_validation`,
  `tng_sc/ia_insert/real_local.py`; the pixel IA is the local mean alignment, like sum w e / sum w and like the NLA
  term, which was already inside the SC ratio).  `ia_over_D=False` = previous term I, bit-identical
  (`tests/test_ia_over_D.py`; backups `/pscratch/sd/m/mgatti/des_decade_test/code_backup_20261123/`).  Effect of
  I/D on the a_c <-> A0 prior mapping: `Projects/ia_sc_validation/ia_priors/over_D_mapping_check.py`: halo GI
  (l 100-1000) changes by -2..+6%, self term +1..+35% (bin 0, b = 1.7).  Open (pre-existing, not from I/D): the
  production NLA with SC weighting has 1.2-2.0x the GI of the plain NLA the mapping used.
* 2026-11-23: pure NLA (user: "the NLA has to be pure NLA, the b delta term for IA needs to be 0; this does not mean we
  have no SC"): `nla_over_D=False` (default) in `make_WL_sample` and `make_BFD_sample` (Poisson path), IA_model='NLA':
  the IA term sum_i n_i IA_i (+ TT) is added outside the source-clustering ratio (pixel / object mean), SC still acts
  on the gravitational shear and the noise.  The previous term was sum n IA / D ~ IA (1 - b delta).
  `nla_over_D=True` = previous, bit-identical; 'deltaNLA' unchanged.  Test `tests/test_nla_pure.py`.

## Per-simulation pipeline (to write: driver + submit scripts)

1. Extract only `fof/run.NNNNN.fofstats.0` with z < z_max_IA (plus metadata if missing).
2. IA halo lightcone: `halos.save_halocatalog(..., log10_mass_limit=12, include_ia_shapes=True)` (Gower
   reference: 19 min, 34 GB RSS, 28.6 GB parquet for z < 2); delete the FoF files.
3. For noise_rel in {0, 1}: density -> lensing fields; for rot in {0..3}: one IA draw (shared by the four
   samples); per sample: n(z)/m draw, SC draw, halo population (f_c, f_sat), numerator with noise-budget
   rejection of the marks, map maker, save.
4. Delete the halo parquet, write a DONE marker with the package git hash.

## Open items

* DECADE m and dz priors: arXiv:2509.03582 Table 1 (old script had SGC sigma_m2 = 0.004657, typo for 0.00657).
* mdet n(z)/m realisations (Jan 2026) confirmed final.
* IA-draw sharing across samples (current: one draw per sim / baryon draw / rotation for all four).
* runs3A halo lightcone on Hypatia (native fofstats.0 format, memory, time).

## Hypatia (hypatia/)

* Inputs: `rsync -L` the staged inputs (40 GB, BFD catalogue 28 GB) to `/share/lustre/mgatti/des_decade_run_inputs`,
  `DES_DECADE_INPUTS` points there, `python data_prep/make_manifest.py --check`.
* `smoke_test.sh SUITE RUN_DIR`: first job, one simulation, b0 r0, halo IA, kappa_CMB, into a scratch directory;
  exercises the native FoF tar extraction, the IA halo lightcone, `run.NNNNN.lightcone.npy` and the Hypatia
  `delta_b` / `sys_baryo` files, then `tests/validate_e2e.py`.
* `make_manifest.sh SUITE N`: simulations without `mocks_done.json` that have the inputs, in chunks of N.
* `submit_mocks.sh SUITE MANIFEST`: one node per job, one `srun` task per simulation (no MPI in the driver,
  which forks process pools), `--nproc` = cpus per task, resumable.  Measured at NERSC: peak RSS 164 GB per
  simulation -> 2 per 480 GB node (`-p CORES64 --ntasks=2 --cpus-per-task=32 --mem=480G`).
