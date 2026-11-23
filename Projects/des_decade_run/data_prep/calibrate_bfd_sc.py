#!/usr/bin/env python3
"""Source-clustering v2 noise calibration for BFD (DES Y6), as for DES / DECADE (des_decade_calibration.py).

BFD shear per pixel is e = -(sum q)/(sum R); source clustering rescales the stochastic q of the real objects
per pixel (make_bfd_map(q_noise_scale_map=...)).  With the Poisson noise model the variance modulation is
h(lambda D)/h(lambda) (source_clustering.noise_modulation), so q is scaled by sqrt(h(lambda D)/h(lambda)), with
    D       projected source density of the bin (shell_bias b [1 + s (z - zbar)]) at the object positions,
            averaged per catalogue pixel (as make_BFD_sample, rotation 0);
    lambda  expected number of real BFD objects per pixel, nbar * fracdet (NSIDE-4096 occupancy of the real
            objects of all bins), in the catalogue frame (independent of the rotation).
The rescaled noise maps are matched in variance and kurtosis to the randomly rotated data
(source_clustering.match_noise_moments) on the grid b in {0.8, 1.1, 1.4, 1.7} x s in {0, 0.3, 0.6, 0.9}.

Simulation: GowerSt2 Fiducial/1_big (no baryons), NSIDE 1024.  n(z): mean of the BFD-calibrated SOMPZ
realisations (bfd_nonneg_calibration_sompz_realisations_20261106).

usage:  calibrate_bfd_sc.py bin T        (one tomographic bin, T = 0..3; run the four in parallel)
        calibrate_bfd_sc.py merge        (polynomial fit, writes data/SC_v2_BFD_nside1024_sim1.json)
"""
import json
import sys
from pathlib import Path

import h5py
import healpy as hp
import numpy as np

sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
from LSS_forward_model.bfd_utils import make_bfd_map
from LSS_forward_model.cosmology import read_sims_params, recover_shell_info
from LSS_forward_model.maps import (apply_nz_shifts_and_build_shells, convert_to_pix_coord,
                                    load_and_baryonify_gower_st_shells, rotate_and_rebin)
from LSS_forward_model.source_clustering import (bin_mean_redshift, expected_counts_map, fit_SC_corrections,
                                                 fracdet_map, match_noise_moments, noise_modulation,
                                                 save_SC_calibration, shell_bias)

NSIDE = 1024
SIM = '/global/cfs/cdirs/m5099/GowerSt2/Fiducial/1_big/'
CAT = '/global/cfs/cdirs/m5099/DESY6_bfd/BFD_catalog_v5_UNBLINDED_light.npy'
NZ = ('/global/cfs/cdirs/m5099/DESY6_bfd/bfd_nonneg_calibration_sompz_realisations_20261106/'
      'bfd_sompz_calibrated_realisations.hdf5')
HERE = Path(__file__).resolve().parent
OUT = HERE.parent/'data'
WORK = HERE/'work_bfd_sc'
GRID = [dict(b=b, s=s) for b in (0.8, 1.1, 1.4, 1.7) for s in (0.0, 0.3, 0.6, 0.9)]
SEED = 20261106


def bfd_mean_nz():
    """Mean BFD n(z) on the grid of the realisations, with a leading z = 0 (as BF_nz_rebinned.npy)."""
    with h5py.File(NZ, 'r') as f:
        z = np.concatenate([[0.], f['z_mid'][:]])
        nz = np.concatenate([np.zeros((4, 1)), f['mean_nz_shifted'][:]], axis=1)
    return {'z_rebinned': z, 'nz_rebinned': nz}


def load_simulation():
    sims_parameters, cosmo_bundle = read_sims_params(SIM)
    shells_info = recover_shell_info(SIM + '/z_values.txt', max_z=49)
    nz = bfd_mean_nz()
    _, shells, _, zeff, ngal = apply_nz_shifts_and_build_shells(
        z_rebinned=nz['z_rebinned'], nz_all=nz['nz_rebinned'], dz_values=[0, 0, 0, 0], shells_info=shells_info)
    baryons = {'enabled': False, 'max_z_halo_catalog': 1.5, 'mass_cut': 13, 'do_tSZ': False,
               'base_params_path': '/global/u2/m/mgatti/LSS_forward_model/Data/Baryonification_wl_tsz_flamingo_parameters.npy',
               'filename_new_params': 'sys_baryo_0.npy', 'values_to_update': None}
    density, _ = load_and_baryonify_gower_st_shells(SIM, sims_parameters, cosmo_bundle, baryons, NSIDE,
                                                    shells_info, shells)
    return np.asarray(ngal), np.asarray(zeff), density


def run_bin(tomo):
    WORK.mkdir(exist_ok=True)
    ngal, zeff, density = load_simulation()
    cats = np.load(CAT, allow_pickle=True).item()
    real_cats = {t: {'ra': cats[t]['ra'][cats[t]['selection_term'] == 0],
                     'dec': cats[t]['dec'][cats[t]['selection_term'] == 0]} for t in cats}
    fracdet = fracdet_map(real_cats, NSIDE)
    cat = cats[tomo]
    del cats, real_cats
    real = cat['selection_term'] == 0
    lam = expected_counts_map(convert_to_pix_coord(cat['ra'][real], cat['dec'][real], nside=NSIDE), NSIDE, fracdet)

    # catalogue pixel and simulation pixel (rotation 0) of every row, as make_BFD_sample
    pix_cat = convert_to_pix_coord(cat['ra'], cat['dec'], nside=NSIDE)
    pix_sim = rotate_and_rebin(convert_to_pix_coord(cat['ra'], cat['dec'], nside=2*NSIDE), NSIDE, 0)
    count = np.bincount(pix_cat, minlength=hp.nside2npix(NSIDE)).astype(float)

    # data noise: randomly rotated BFD catalogue
    e1, e2, mask_ref, _ = make_bfd_map(cat, nside=NSIDE, random_rotate=True, rotate_selection=True,
                                       rng=np.random.default_rng(SEED + 1))
    y_all = e1**2 + e2**2

    zbar = bin_mean_redshift(ngal, zeff)
    rows = []
    for point in GRID:
        bias = shell_bias(ngal, zeff, [point['b']]*4, [point['s']]*4)
        d_tot = np.zeros(hp.nside2npix(NSIDE))
        for i in range(len(density)):
            d_tot += ngal[tomo, i]*(1 + bias[tomo, i]*density[i])
        d_map = np.zeros(hp.nside2npix(NSIDE))
        d_map[count > 0] = (np.bincount(pix_cat, weights=d_tot[pix_sim], minlength=len(count))[count > 0]
                            / count[count > 0])
        q_scale = np.sqrt(noise_modulation(d_map, lam, 'poisson'))
        e1, e2, mask, _ = make_bfd_map(cat, nside=NSIDE, random_rotate=True, rotate_selection=True,
                                       q_noise_scale_map=q_scale, rng=np.random.default_rng(SEED))
        use = (mask > 0) & (mask_ref > 0)
        corr_variance, coeff_kurtosis, A_corr = match_noise_moments((e1**2 + e2**2)[use], y_all[use])
        rows.append(dict(point, corr_variance=float(corr_variance), coeff_kurtosis=float(coeff_kurtosis),
                         A_corr=float(A_corr)))
        print(f"bin {tomo} b={point['b']} s={point['s']}: corr_variance {corr_variance:.4f} "
              f"coeff_kurtosis {coeff_kurtosis:.4f} A_corr {A_corr:.4f}", flush=True)
    (WORK/f'bin{tomo}.json').write_text(json.dumps(dict(rows=rows, zbar=float(zbar[tomo]),
                                                        lambda_mean=float(lam[lam > 0].mean()))) + '\n')


def merge():
    parts = [json.loads((WORK/f'bin{t}.json').read_text()) for t in range(4)]
    calib = []
    for k, point in enumerate(GRID):
        row = dict(point, mu=0.0)
        for key in ('corr_variance', 'coeff_kurtosis', 'A_corr'):
            row[key] = [parts[t]['rows'][k][key] for t in range(4)]
        calib.append(row)
    fits = fit_SC_corrections(calib, names=('b', 's'), degree=2)
    for key in ('corr_variance_fit', 'coeff_kurtosis_fit', 'A_corr_fit'):
        print(key, 'rms residual per bin:', [f"{f['rms_residual']:.2e}" for f in fits[key]])
    OUT.mkdir(exist_ok=True)
    path = OUT/'SC_v2_BFD_nside1024_sim1.json'
    save_SC_calibration(path, fits, calib=calib, zbar=[p['zbar'] for p in parts], nside=NSIDE, survey='BFD',
                        sim=1, lambda_mean=[p['lambda_mean'] for p in parts], catalogue=CAT, nz=NZ)
    print('wrote', path)


if __name__ == '__main__':
    if sys.argv[1] == 'bin':
        run_bin(int(sys.argv[2]))
    else:
        merge()
