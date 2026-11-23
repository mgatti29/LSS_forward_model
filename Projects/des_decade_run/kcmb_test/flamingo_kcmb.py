#!/usr/bin/env python3
"""Baryons in the CMB lensing convergence at 2000 < L < 4000, and the NSIDE-1024 baryon-difference shortcut (FLAMINGO).

Born kappa_CMB from the FLAMINGO L1000N1800 lightcone overdensity shells (z < 3, observer 0) of the hydro fiducial
(L1_m9), its dark-matter-only twin (L1_m9_DMO, same initial conditions), fgas-8sigma (strongest feedback) and Jet:
    kappa = sum_i (3/2) Omega_m (H0/c)^2 chi_i (chi* - chi_i)/chi* (1 + z_i) delta_i dchi_i
(z < 3 only: baryons act at low redshift; the z > 3 part dilutes the ratios to the full kappa).
Outputs the C_ell ratios to the DMO run up to ell = 4096, and the hybrid
    kappa_hybrid = kappa_DMO,2048 + upgrade(kappa_hydro,1024 - kappa_DMO,1024)
(the production shortcut: baryons from the NSIDE-1024 fields, small scales from the dark-matter-only field)
against the true kappa_hydro,2048.

usage: flamingo_kcmb.py maps RUN NSIDE   (one map per call; run in parallel)
       flamingo_kcmb.py stats
"""
import json
import sys
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import pyccl as ccl

FLAMINGO = Path('/global/cfs/cdirs/m5099/Flamingo/L1000N1800')
OUT = Path('/pscratch/sd/m/mgatti/kcmb_test/flamingo')
LMAX = 4096
# FLAMINGO D3A cosmology (Mpc units)
COSMO = ccl.Cosmology(Omega_c=0.306 - 0.0486 - 0.0014, Omega_b=0.0486, h=0.681, sigma8=0.807, n_s=0.967,
                      m_nu=0.06, matter_power_spectrum='linear')
OMEGA_M, H0_OVER_C = 0.306, 0.681*100/299792.458


def born_kappa(run, nside):
    a_star = 1/1090.
    chi_star = ccl.comoving_radial_distance(COSMO, a_star)
    kappa = np.zeros(hp.nside2npix(nside))
    with h5py.File(FLAMINGO/run/f'overdensity_nside{nside}_lc0.hdf5', 'r') as f:
        for name in sorted(f.keys()):
            shell = f[name]
            chi_in, chi_out = shell.attrs['comoving_inner_radius'], shell.attrs['comoving_outer_radius']
            chi = 0.5*(chi_in + chi_out)
            z = 1/ccl.scale_factor_of_chi(COSMO, chi) - 1
            weight = 1.5*OMEGA_M*H0_OVER_C**2*chi*(chi_star - chi)/chi_star*(1 + z)*(chi_out - chi_in)
            kappa += weight*shell['overdensity'][:]
            print(f'{run} {nside} {name} z={z:.3f}', flush=True)
    return kappa


def run_maps(run, nside):
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT/f'kappa_{run}_{nside}.npy', born_kappa(run, nside).astype(np.float64))


def run_stats():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    load = lambda run, nside: np.load(OUT/f'kappa_{run}_{nside}.npy')
    alm = {(r, n): hp.map2alm(load(r, n), lmax=LMAX if n == 2048 else 2*n)
           for r, n in (('L1_m9', 2048), ('L1_m9_DMO', 2048), ('L1_m9', 1024), ('L1_m9_DMO', 1024),
                        ('fgas-8sigma', 2048), ('Jet', 2048)) if (OUT/f'kappa_{r}_{n}.npy').exists()}
    edges = np.unique(np.rint(np.geomspace(20, LMAX, 25)).astype(int))
    centre = np.sqrt(edges[:-1]*edges[1:])
    bp = lambda cl: np.array([cl[lo:hi].mean() for lo, hi in zip(edges[:-1], edges[1:])])
    dmo = bp(hp.alm2cl(alm[('L1_m9_DMO', 2048)]))
    out = {'ell': centre.tolist()}
    for run in ('L1_m9', 'fgas-8sigma', 'Jet'):
        if (run, 2048) in alm:
            out[f'{run}/DMO (z<3)'] = (bp(hp.alm2cl(alm[(run, 2048)]))/dmo).tolist()
    # hybrid: DMO at 2048 + baryon difference from the 1024 maps (upgraded with the pixel-window ratio)
    lmax_lo = 2*1024
    diff = alm[('L1_m9', 1024)] - alm[('L1_m9_DMO', 1024)]
    diff = hp.almxfl(diff, hp.pixwin(2048, lmax=lmax_lo)/hp.pixwin(1024, lmax=lmax_lo))
    hybrid = alm[('L1_m9_DMO', 2048)].copy()
    ell, m = hp.Alm.getlm(LMAX)
    idx = hp.Alm.getidx(LMAX, *hp.Alm.getlm(lmax_lo))
    hybrid[idx] += diff
    hydro = bp(hp.alm2cl(alm[('L1_m9', 2048)]))
    out['hybrid/hydro (z<3)'] = (bp(hp.alm2cl(hybrid))/hydro).tolist()
    out['DMO/hydro (z<3)'] = (dmo/hydro).tolist()
    out['r(hybrid, hydro)'] = (bp(hp.alm2cl(hybrid, alm[('L1_m9', 2048)]))/np.sqrt(bp(hp.alm2cl(hybrid))*hydro)).tolist()
    (OUT/'flamingo_kcmb.json').write_text(json.dumps(out, indent=1) + '\n')

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for key in out:
        if key.endswith('/DMO (z<3)'):
            ax[0].plot(centre, out[key], label=key)
    for key in ('hybrid/hydro (z<3)', 'DMO/hydro (z<3)'):
        ax[1].plot(centre, out[key], label=key)
    for a, t in zip(ax, ('baryon suppression of C_kk (FLAMINGO, z < 3 part)', 'NSIDE-1024 baryon shortcut vs truth')):
        a.set_xscale('log'); a.axhline(1, color='k', lw=0.5); a.axvline(2048, color='0.6', ls=':'); a.legend(); a.set_title(t)
    fig.tight_layout()
    fig.savefig(OUT/'flamingo_kcmb.png', dpi=110)
    for key, value in out.items():
        if key != 'ell':
            print(key, np.round(np.array(value)[[6, 12, 16, 19, 21, 23]], 4).tolist())
    print('ell', np.round(centre[[6, 12, 16, 19, 21, 23]]).tolist())


if __name__ == '__main__':
    if sys.argv[1] == 'maps':
        run_maps(sys.argv[2], int(sys.argv[3]))
    else:
        run_stats()
