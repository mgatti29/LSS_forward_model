# Intrinsic-alignment priors for the DES / DECADE / BFD mocks

Adopted 2026-11-06.  Machine-readable version: `config/ia_priors.json`.
Tests and outputs: `prior_tests/` (scripts) and `prior_tests/out/` (results).

## Adopted halo-IA prior, revision 2026-11-20 (supersedes the 2026-11-06 box below)

Goal: span approximately the adopted NLA prior (A0 ~ U(-0.5, 1.5) at z0 = 0.3, eta ~ N(0,1) in [-2, 2]) without
going far beyond it, and keep the halo-IA self (shot-noise) term small.

| parameter | prior | notes |
|---|---|---|
| a_c = f_c A_c | U[-0.2, 0.5] | GI amplitude; a_c ~ 0.33 A0_NLA (A0 = 0.5, eta = 0 needs a_c 0.154-0.198 for every structure) -> A0 ~ [-0.5, 1.5] |
| A_c   | derived, a_c / f_c | range [-0.29, 0.71] |
| eta_c | U[-5, 0]    | -4 covers eta_NLA in [-0.5, 1], -5 adds eta_NLA = -1; eta_NLA ~ -2 (eta_c -6..-10) excluded: largest self term and cap loss |
| A_s   | U[-0.05, 0.05] | independent of A_c; satellites are 1-1.5% of the self term (user: "barely matters") |
| eta_s | U[-0.5, 0.5] | |
| f_c   | U[0.7, 1.0] | self term at fixed GI scales as 1/f_c; NLA coverage independent of f_c; f_c -> 1 brings bin-4 II closer to NLA |
| f_sat | U[0.3, 0.8] | irrelevant for the self term |

Fixed parameters, the local IA cap (alpha = 1) and the disabled rejection are unchanged.
Evidence: `Projects/ia_sc_validation/RESULTS.md` (sections "Core-first halo box", "Self term of the candidate box",
"f_c study", "Survey forward-model test").  Expected: |A0| <= 0.5 -> <= 0.4% pixels over budget, cap removes <= 1.5% GI;
A0 = 1, f_c >= 0.7, eta away from -2 -> few % of budget.  NOT yet tested: A0 in (1, 1.5] on survey populations
(self ~ A^2, expect ~2x the A = 1 numbers), DECADE n(z) for the a_c <-> A0 conversion (derived on DES Y6).
Previous box kept in `config/ia_priors.json` (`halo.previous_prior_20261106`).

## Previous halo-IA prior (2026-11-06, superseded)

(virtual-halo model, `IA_model='halo'`)

| parameter | prior | TNG300 fits (DES Y6-like) | notes |
|---|---|---|---|
| A_c   | U[-2, 2.5]  | 2.57 (f_c 0.5) / 1.71 (f_c 0.7) | central alignment amplitude |
| eta_c | U[-2, 0]    | -1   | (1+z)/1.67 scaling of A_c |
| A_s   | U[0, 0.03]  | 0.019 / 0.025 | satellite (radial) alignment amplitude |
| eta_s | U[-0.5, 0.5]| 0    | |
| f_c   | U[0.2, 0.7] | 0.5 / 0.7 | central fraction |
| f_sat | U[0.3, 0.8] | 0.8 / 0.6 | satellite fraction |

Fixed: sigma_logM = 0.2, alpha = 1, b_c = b_s = 0, halo cut 1e12 Msun, virtual halos on.
Plus a **noise-budget rejection**: after the halo-IA numerator is built, the mark parameters
(A_c, eta_c, A_s, eta_s) are redrawn on the same halo population if, in any bin, more than 2% of the
footprint pixels have var_IA / var_noise > 1 (threshold provisional, to be checked on the test mocks).

Sign convention: positive amplitudes are physical (radial) alignments, the same sign as NLA A > 0.  This
is the convention of `make_halo_IA_numerator` after the sign fix (see below); the TNG300 fit values are quoted
in this convention.

Comparison model for the test mocks: NLA with A_IA, eta_IA ~ U[-2, 2], pivot (1+z)/1.67.

## Why these ranges

The user's starting proposal was A_c, A_s, eta_c in [-2, 2], eta_s in [-0.5, 0.5], f_c in [0.2, 0.7],
f_sat in [0.3, 0.8].  The requirement: the prior must contain TNG300, must not "explode" with respect to
NLA with A, eta in [-2, 2] at the shear two-point level, and must not create catastrophic noise.

**1. A_s: [-2, 2] -> [0, 0.03].**  Satellites carry a unit radial orientation (|u| = 1), so A_s = 2 gives
|e_IA| ~ 2 per satellite.  On TNG300 (DES Y6-like shear-shear, 10 bin pairs, 30 < ell < 2000, Gaussian DES
errors), the IA contribution under the original prior reaches 99th-percentile 8000 sigma (2900 x GG), and the
IA white variance exceeds the DES shape-noise variance in 73% of the draws; on Gower St halos A_s = 2 puts
97% of the DES pixels over the noise budget.  A_s in [-0.05, 0.05] still left a satellite 1-halo II tail of
6.7 x GG (99%), [0, 0.03] brings it to 1.5 x GG, the same as NLA.  Negative A_s (tangential satellites) is
unphysical; the fits give 0.02.

**2. eta_c: [-2, 2] -> [-2, 0].**  eta_c > 0 boosts the high-z centrals: it doubles the 99% tail of the IA
contribution and breaks the bin-4 noise budget on Gower (eta_c = 2, A_c >= 1.5: 52-97% of bin-4 pixels over
budget).  The fits prefer eta_c = -1.

**3. A_c: [-2, 2] -> [-2, 2.5].**  A_c and f_c are degenerate; TNG needs A_c ~ 2.6-2.7 at f_c ~ 0.45-0.5 or
1.7 at f_c = 0.7, so with A_c <= 2 a TNG-like signal is reachable only in the f_c = 0.7 corner and a TNG-like
truth would sit on the prior edge.  The upper limit is raised to 2.5 (rather than 3) so that fewer draws are
rejected by the noise budget (on Gower at f_c = 0.5, f_sat = 0.8, eta_c = -2: A_c = 2 puts <= 3% of pixels
over budget, A_c = 2.5 14-17% in bins 1-2, A_c = 3 37-44%).

**4. Noise-budget rejection instead of a hard cut.**  The observed shape dispersion already contains the IA
white variance, so a draw whose IA white variance exceeds the measured shape noise is excluded by the data.
Rejecting such draws (cheap: only the marks are redrawn) removes exactly the problematic corners (large A_c at
large f_c, large A_s, eta_c > 0) and keeps the rest of the box.

**5. f_c, f_sat unchanged.**  Below z = 0.02 (0.1% of the DES sources) small f_c has no solution in
`halo_ia.occupation` (Gower DES shell 0 needs f_c >= 0.6 / 0.38 / 0.17 at f_sat 0.3 / 0.55 / 0.8); above
z ~ 0.03 any f_c >= 0.1 works.  This is fixed in the code (cap the hosted fraction), not in the prior.

## Evidence (TNG300 DES Y6-like shear-shear, prior-predictive percentiles 50 / 90 / 99)

`prior_tests/out/prior_scan.json`, 9 (f_c, f_sat) populations, 20000 draws each:

| | chi2 of the IA term | max dC / GG | max dC / sigma | draws with mean var_IA/var_noise > 1 |
|---|---|---|---|---|
| NLA, A, eta in [-2, 2] | 290 / 1174 / 2561 | 0.34 / 0.80 / 1.53 | 2.7 / 5.6 / 8.1 | - |
| halo, adopted box (A_c [-2, 2.5]) | 184 / 1297 / 3390 | 0.38 / 0.85 / 1.67 | 2.8 / 7.3 / 17 | 0.07% |
| halo, same with A_c [-2, 2] | 147 / 1057 / 3351 | 0.36 / 0.76 / 1.49 | 2.6 / 6.3 / 14 | 0 |
| TNG300 galaxies        | 1382 | 0.84 | 7.1 | - |

Relative to GG the halo and NLA priors are equivalent; the larger halo tail in sigma units is small-scale power
(halo II) that NLA cannot produce.  The mean-over-sky ratio is the TNG300 number; the per-pixel Gower budget
(above) is stricter and is what the rejection rule uses.

Fisher test (`prior_tests/out/fisher_ia.json`; lensing amplitude a as S8 proxy, GG ~ a^2, GI ~ a; flat
priors as Gaussian width/sqrt(12)): sigma(a) = 0.0048 with IA fixed, 0.0074 with NLA, 0.0081 with all six
halo parameters (+9%).  If the truth is TNG300, NLA (best fit A = 3.9, eta = 1.5, chi2 433 for 120 points)
biases a by +0.055 (+7.4 sigma), the halo model by -0.4 sigma.

## Known issues to fix before the production mocks

* **Sign of the halo-IA term: FIXED (2026-11-06).**  The halo marks and TNG300 shapes have the opposite E-mode
  sign to the package lensing shear (<delta, g_obs> > 0; NLA A > 0: GI < 0, dI < 0; halo A_c > 0: GI > 0, dI > 0
  on TNG300 and on Gower through the production path: bin 1 GI/GG NLA(A=1) -0.26, halo(A_c=1) +0.13).
  `make_halo_IA_numerator` now multiplies the marks by -1 (`PACKAGE_IA_SIGN`).
* **A_c units: coherent signal transfers.**  GI/GG per unit A_c (f_c 0.5, f_sat 0.8, eta_c -1) is 0.10-0.13
  (bin 1) on Gower and 0.07-0.14 on TNG300 (`prior_tests/out/gower_gi_calibration_0.5_0.8.json`): Gower
  10-20-particle halo shapes cost white variance (noise budget, handled by the rejection), not coherent
  alignment.  No rescaling of the A_c prior needed.
* **Occupation at z < 0.02: FIXED** (`halo_ia.occupation(cap=0.98)`, see 5).
