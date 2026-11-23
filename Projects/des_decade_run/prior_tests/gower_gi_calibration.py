#!/usr/bin/env python3
"""Halo-IA vs NLA signal per unit amplitude on Gower St halos, through the production code path.

Fields per tomographic bin t (DES Y6 n(z), full sky, NSIDE 1024), exactly as lensing.make_WL_sample builds
them (source clustering off):
    G_t      = - sum_i ngal[t,i] gamma_i                              (observed-convention lensing shear)
    NLA_t    = - sum_i ngal[t,i] F_nla(z_i; A=1, eta) IA_shear_i       (IA_model='NLA')
    halo_t   = make_halo_IA_numerator(...)['numerator'][t]             (IA_model='halo', added as is)
for the halo model with only centrals (A_c=1, eta_c) and only satellites (A_s=1, eta_s=0).
E-mode bandpowers on the cheap_ia_validation ell bins: GI/GG = <G_t I_t>/<G_t G_t> and the coherent
II/GG (white term 0.5 <numerator_sq> Omega_pix removed for the halo fields).  Compare with the same numbers on
TNG300 (sign_check.py: halo A_c=1 eta_c=-1 bin 1 GI/GG ~ +0.12 in the TNG shape convention, NLA A=1 ~ -0.14).

usage: gower_gi_calibration.py [F_C F_SAT]
output: prior_tests/out/gower_gi_calibration_<f_c>_<f_sat>.json
"""
import json
import sys
from pathlib import Path

import healpy as hp
import numpy as np
import pyccl as ccl

sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
from LSS_forward_model.cosmology import read_sims_params, recover_shell_info
from LSS_forward_model.halo_ia import build_IA_halo_sources_from_catalog, make_halo_IA_numerator
from LSS_forward_model.maps import (F_nla, apply_nz_shifts_and_build_shells, compute_lensing_fields,
                                    load_and_baryonify_gower_st_shells)
from LSS_forward_model.source_clustering import fracdet_map

path_simulation = '/global/cfs/cdirs/m5099/GowerSt2/Fiducial/1_big/'
path_halo_IA = '/pscratch/sd/m/mgatti/source_clustering_v2/halo_ia/halo_catalog_ia_z2_m12_Fiducial_1_big.parquet'
OUT = Path(__file__).resolve().parent/'out'
NSIDE = 1024
LMAX = 2000
ELL_EDGES = np.unique(np.rint(np.geomspace(30, LMAX, 13)).astype(int))      # as cheap_ia_validation
NLA_ETA = (-1.0, 0.0)
ETA_C = -1.0


def bandpower(cl):
    ell = np.arange(len(cl))
    return np.array([np.sum((2*ell[u] + 1)*cl[u])/np.sum(2*ell[u] + 1)
                     for u in [(ell >= lo) & (ell < hi) for lo, hi in zip(ELL_EDGES[:-1], ELL_EDGES[1:])]])


def E_alm(field):
    return hp.map2alm_spin([np.ascontiguousarray(field.real), np.ascontiguousarray(field.imag)], 2, lmax=LMAX)[0]


def main():
    f_c, f_sat = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (0.5, 0.8)
    OUT.mkdir(exist_ok=True)
    sims_parameters, cosmo_bundle = read_sims_params(path_simulation)
    sims_parameters.update(dz=[0, 0, 0, 0], rot=0, delta_rot=0)
    baryons = {'enabled': False, 'max_z_halo_catalog': 1.5, 'mass_cut': 13, 'do_tSZ': False,
               'base_params_path': '/global/u2/m/mgatti/LSS_forward_model/Data/Baryonification_wl_tsz_flamingo_parameters.npy',
               'filename_new_params': 'sys_baryo_0.npy', 'values_to_update': None}
    shells_info = recover_shell_info(path_simulation + '/z_values.txt', max_z=49)
    nz = np.load('/global/cfs/cdirs/m5099/DESY3/nz_DESY6.npy', allow_pickle=True).item()
    nz_shifted, shells, steps, zeff, ngal = apply_nz_shifts_and_build_shells(
        z_rebinned=nz['z_rebinned'], nz_all=nz['nz_rebinned'], dz_values=sims_parameters['dz'], shells_info=shells_info)
    density, _ = load_and_baryonify_gower_st_shells(path_simulation, sims_parameters, cosmo_bundle, baryons,
                                                    NSIDE, shells_info, shells)
    cats = np.load('/global/cfs/cdirs/m5099/DESY3/DESY6.npy', allow_pickle=True).item()
    fsky = fracdet_map(cats, NSIDE).mean()
    bin_counts = np.array([len(cats[t]['e1']) for t in range(4)])
    del cats

    # lensing shear and NLA fields per bin, as make_WL_sample ------------------------------------------
    fields = compute_lensing_fields(density, shells, cosmo_bundle['pars_camb'], NSIDE,
                                    do_kappa=False, do_shear=True, do_IA=True)
    rho_c1 = 5e-14*ccl.physical_constants.RHO_CRITICAL
    G, NLA = [], {eta: [] for eta in NLA_ETA}
    for t in range(4):
        g = np.zeros(hp.nside2npix(NSIDE), dtype=np.complex128)
        nla = {eta: np.zeros_like(g) for eta in NLA_ETA}
        for i in range(len(density)):
            if ngal[t, i] == 0:
                continue
            g -= ngal[t, i]*fields['gamma'][i][0]
            for eta in NLA_ETA:
                f = F_nla(z=zeff[i], om0=sims_parameters['Omega_m'], A_ia=1.0, rho_c1=rho_c1, eta=eta, z0=0.67,
                          cosmo=cosmo_bundle['cosmo_pyccl'])
                nla[eta] -= ngal[t, i]*f*fields['IA_shear'][i][0]
        G.append(E_alm(g))
        for eta in NLA_ETA:
            NLA[eta].append(E_alm(nla[eta]))
    del fields
    print('lensing and NLA fields done', flush=True)

    # halo model, centrals only and satellites only ---------------------------------------------------
    nz_total = np.sum([bin_counts[t]*nz_shifted[t] for t in range(4)], axis=0)
    nz_total[np.asarray(nz['z_rebinned']) < 0.02] = 0.     # occupation infeasible below z=0.02 for small f_c
    population = {'f_c': f_c, 'f_sat': f_sat, 'sigma_logM': .2, 'alpha': 1.}
    model_objects = build_IA_halo_sources_from_catalog(
        path_halo_cat=path_halo_IA, sims_parameters=sims_parameters, halo_catalog_log10mass_cut=12.0,
        total_number_counts=bin_counts.sum()/fsky, nz_redshift=nz['z_rebinned'], nz=nz_total,
        shells_info=shells_info, model_parameters={'population': population, 'ia': {}},
        cosmo_bundle=cosmo_bundle, seed=913603, show_progress=False)
    omega = hp.nside2pixarea(NSIDE)
    halo, white = {}, {}
    for name, ia in (('central', dict(A_c=1., eta_c=ETA_C, A_s=0., eta_s=0.)),
                     ('satellite', dict(A_c=0., eta_c=0., A_s=1., eta_s=0.))):
        ia.update(b_c=0., b_s=0.)
        h = make_halo_IA_numerator(model_objects, {'ia': ia}, density, ngal, NSIDE, bin_counts=bin_counts, seed=0)
        halo[name] = [E_alm(h['numerator'][t]) for t in range(4)]
        white[name] = [float(0.5*np.mean(h['numerator_sq'][t])*omega) for t in range(4)]
    del model_objects
    print('halo fields done', flush=True)

    out = {'f_c': f_c, 'f_sat': f_sat, 'eta_c': ETA_C, 'ell_edges': ELL_EDGES.tolist(), 'bins': []}
    for t in range(4):
        gg = bandpower(hp.alm2cl(G[t], G[t]))
        row = {'bin': t + 1, 'GG': gg.tolist()}
        for eta in NLA_ETA:
            row[f'NLA eta={eta} GI/GG'] = (bandpower(hp.alm2cl(G[t], NLA[eta][t]))/gg).tolist()
            row[f'NLA eta={eta} II/GG'] = (bandpower(hp.alm2cl(NLA[eta][t], NLA[eta][t]))/gg).tolist()
        for name in halo:
            row[f'halo {name} GI/GG'] = (bandpower(hp.alm2cl(G[t], halo[name][t]))/gg).tolist()
            row[f'halo {name} II/GG'] = ((bandpower(hp.alm2cl(halo[name][t], halo[name][t])) - white[name][t])/gg).tolist()
            row[f'halo {name} white/GG'] = (white[name][t]/gg).tolist()
        out['bins'].append(row)
        print(f"bin {t+1}: GI/GG  NLA(A=1,eta=0) {np.round(row['NLA eta=0.0 GI/GG'][::3], 4)}  "
              f"halo centrals(A_c=1) {np.round(row['halo central GI/GG'][::3], 4)}  "
              f"satellites(A_s=1) {np.round(row['halo satellite GI/GG'][::3], 4)}", flush=True)
    (OUT/f'gower_gi_calibration_{f_c}_{f_sat}.json').write_text(json.dumps(out, indent=1) + '\n')
    print('saved', flush=True)


if __name__ == '__main__':
    main()
