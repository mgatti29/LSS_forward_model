#!/usr/bin/env python3
"""Noise budget of the halo IA prior on Gower St halos at DES Y6 density.

The halo IA map carries a white variance var_IA = 0.5 <numerator_sq> (expected on NSIDE 64 superpixels, as in
lensing.make_WL_sample) that must fit inside the shape-noise variance of the data, var_noise = 0.5 sum w^2 e^2 /
(sum w)^2 per pixel (source clustering neglected: it modulates both sides alike).  Where var_IA > var_noise the
mock noise is set to zero and the mock is no longer the data noise + IA.

numerator_sq is quadratic in the amplitudes and has no central-satellite cross term, so per population
(f_c, f_sat) it is stored for A_c = 1 (eta_c nodes) and A_s = 1 (eta_s nodes); any (A_c, A_s) follows:
    ratio = A_c^2 R_c(eta_c) + A_s^2 R_s(eta_s).

Gower St Fiducial/1_big, IA halo lightcone M > 1e12 Msun, z < 2 (the catalogue of des_noise_stats_test_v3).
Output: prior_tests/out/gower_ia_noise[_<f_c>_<f_sat>].json  (usage: gower_ia_noise.py [F_C F_SAT])
"""
import json
import sys
from pathlib import Path

import healpy as hp
import numpy as np

sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
from LSS_forward_model.cosmology import read_sims_params, recover_shell_info
from LSS_forward_model.halo_ia import build_IA_halo_sources_from_catalog, make_halo_IA_numerator
from LSS_forward_model.maps import (apply_nz_shifts_and_build_shells, convert_to_pix_coord,
                                    load_and_baryonify_gower_st_shells, rotate_and_rebin)
from LSS_forward_model.source_clustering import fracdet_map

path_simulation = '/global/cfs/cdirs/m5099/GowerSt2/Fiducial/1_big/'
path_halo_IA = '/pscratch/sd/m/mgatti/source_clustering_v2/halo_ia/halo_catalog_ia_z2_m12_Fiducial_1_big.parquet'
OUT = Path(__file__).resolve().parent/'out'
nside_maps = 1024
ia_noise_nside = 64
log10_mass_cut = 12.0
POPULATIONS = [(0.5, 0.8), (0.2, 0.3), (0.2, 0.8), (0.7, 0.3), (0.7, 0.8)]
ETA_C = (-2.0, 0.0, 2.0)
ETA_S = (-0.5, 0.5)
A_C = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
A_S = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)


def main():
    OUT.mkdir(exist_ok=True)
    sims_parameters, cosmo_bundle = read_sims_params(path_simulation)
    sims_parameters.update(dz=[0, 0, 0, 0], rot=0, delta_rot=0)
    baryons = {'enabled': False, 'max_z_halo_catalog': 1.5, 'mass_cut': 13, 'do_tSZ': False,
               'base_params_path': '/global/u2/m/mgatti/LSS_forward_model/Data/Baryonification_wl_tsz_flamingo_parameters.npy',
               'filename_new_params': 'sys_baryo_0.npy', 'values_to_update': None}
    shells_info = recover_shell_info(path_simulation + '/z_values.txt', max_z=49)
    nz = np.load('/global/cfs/cdirs/m5099/DESY3/nz_DESY6.npy', allow_pickle=True).item()
    nz_shifted, shells, steps, zeff_glass, ngal_glass = apply_nz_shifts_and_build_shells(
        z_rebinned=nz['z_rebinned'], nz_all=nz['nz_rebinned'], dz_values=sims_parameters['dz'],
        shells_info=shells_info)
    density, _ = load_and_baryonify_gower_st_shells(path_simulation, sims_parameters, cosmo_bundle, baryons,
                                                    nside_maps, shells_info, shells)

    # shape-noise variance per pixel (rot 0) ---------------------------------------------------------
    cats = np.load('/global/cfs/cdirs/m5099/DESY3/DESY6.npy', allow_pickle=True).item()
    fracdet = fracdet_map(cats, nside_maps)
    fsky = fracdet.sum()/len(fracdet)
    bin_counts = np.array([len(cats[t]['e1']) for t in range(4)])
    npix = hp.nside2npix(nside_maps)
    var_noise, occupied = np.zeros((4, npix)), []
    for t in range(4):
        pix = rotate_and_rebin(convert_to_pix_coord(cats[t]['ra'], cats[t]['dec'], nside=nside_maps*2),
                               nside_maps, 0, delta_=0)
        w = cats[t]['w']
        e_sq = cats[t]['e1']**2 + cats[t]['e2']**2
        n_map = np.bincount(pix, weights=w, minlength=npix)
        good = n_map > 0
        var_noise[t, good] = 0.5*np.bincount(pix, weights=w**2*e_sq, minlength=npix)[good]/n_map[good]**2
        occupied.append(good)
    del cats

    def expected(numerator_sq, good):
        """numerator_sq averaged on NSIDE-64 superpixels, normalised to the realised footprint total."""
        out = hp.ud_grade(hp.ud_grade(numerator_sq, ia_noise_nside), nside_maps)
        return out*np.sum(numerator_sq[good])/np.sum(out[good])

    nz_total = np.sum([bin_counts[t]*nz_shifted[t] for t in range(4)], axis=0)
    if len(sys.argv) > 3:   # drop the n(z) tail below ZMIN (occupation infeasible at z < 0.02 for small f_c)
        nz_total[np.asarray(nz['z_rebinned']) < float(sys.argv[3])] = 0.
    results = {'populations': {}, 'A_c': A_C, 'A_s': A_S, 'eta_c': ETA_C, 'eta_s': ETA_S}
    populations = POPULATIONS if len(sys.argv) < 3 else [(float(sys.argv[1]), float(sys.argv[2]))]
    tag = '' if len(sys.argv) < 3 else f'_{sys.argv[1]}_{sys.argv[2]}'
    for f_c, f_sat in populations:
        population = {'f_c': f_c, 'f_sat': f_sat, 'sigma_logM': .2, 'alpha': 1.}
        model_objects = build_IA_halo_sources_from_catalog(
            path_halo_cat=path_halo_IA, sims_parameters=sims_parameters, halo_catalog_log10mass_cut=log10_mass_cut,
            total_number_counts=bin_counts.sum()/fsky, nz_redshift=nz['z_rebinned'], nz=nz_total,
            shells_info=shells_info, model_parameters={'population': population, 'ia': {}},
            cosmo_bundle=cosmo_bundle, seed=913603, show_progress=False)
        ratio_maps = {}
        for kind, etas in (('c', ETA_C), ('s', ETA_S)):
            for eta in etas:
                ia = dict(A_c=1. if kind == 'c' else 0., A_s=1. if kind == 's' else 0., eta_c=eta if kind == 'c' else 0.,
                          eta_s=eta if kind == 's' else 0., b_c=0., b_s=0.)
                halo_IA = make_halo_IA_numerator(model_objects, {'ia': ia}, density, ngal_glass, nside_maps,
                                                 bin_counts=bin_counts, seed=0)
                ratio_maps[(kind, eta)] = np.array([
                    np.where(occupied[t], 0.5*expected(halo_IA['numerator_sq'][t], occupied[t])
                             / np.where(occupied[t], var_noise[t], 1.), 0.)
                    for t in range(4)])
                print(f'f_c {f_c} f_sat {f_sat} {kind} eta {eta}: mean ratio (A=1)',
                      [float(ratio_maps[(kind, eta)][t][occupied[t]].mean()) for t in range(4)], flush=True)
        del model_objects

        rows = []
        for eta_c in ETA_C:
            for eta_s in ETA_S:
                for A_c in A_C:
                    for A_s in A_S:
                        r = A_c**2*ratio_maps[('c', eta_c)] + A_s**2*ratio_maps[('s', eta_s)]
                        rows.append(dict(eta_c=eta_c, eta_s=eta_s, A_c=A_c, A_s=A_s,
                                         mean_ratio=[float(r[t][occupied[t]].mean()) for t in range(4)],
                                         frac_gt1=[float(np.mean(r[t][occupied[t]] > 1)) for t in range(4)],
                                         frac_gt05=[float(np.mean(r[t][occupied[t]] > 0.5)) for t in range(4)]))
        results['populations'][f'{f_c}_{f_sat}'] = dict(
            unit_mean_ratio={f'{k[0]}_{k[1]}': [float(m[t][occupied[t]].mean()) for t in range(4)]
                             for k, m in ratio_maps.items()},
            grid=rows)
        (OUT/f'gower_ia_noise{tag}.json').write_text(json.dumps(results) + '\n')
        print(f'population {f_c} {f_sat} saved', flush=True)


if __name__ == '__main__':
    main()
