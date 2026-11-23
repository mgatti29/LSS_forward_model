#!/usr/bin/env python3
"""CMB lensing convergence for the production run: which construction?  Test on Gower St runs3A run001.

Versions (Born, NSIDE 2048, ell <= 4096; all share the same z < 2 shells):
    A   native shells to the end of the lightcone (z ~ 20 here, step 1 is incomplete) + Gaussian remainder
    C   native shells to z = 2 + independent Gaussian remainder (current approach of maps.cmb_lensing_from_glass_plus_gaussian)
    B   native shells to z = 2 + conditional lognormal ladder (LSS_forward_model.coshirex)
    b0, b1  baryon term: Born sum of Delta_b = delta_b,1024 - delta_DMO,1024 (z < 1 shells; the existing baryonified
            fields minus the identical harmonic downgrade of the native counts), upgraded to NSIDE 2048
Native shells: NSIDE-4096 particle counts summed into NSIDE-2048 pixels, delta = counts / mean - 1.

steps:  shells   (parallel over shells)  -> WORK/delta2048/, WORK/dbaryon{0,1}/, WORK/baryon_check.json
        kappa VERSION                    -> WORK/kappa_<VERSION>.npy
        theory                           -> WORK/theory.npz  (CAMB HMcode C_kk, full and beyond the cuts)
        stats                            -> WORK/stats.json, WORK/kcmb_test.png
"""
import copy
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import camb
import healpy as hp
import numpy as np

sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
from LSS_forward_model.coshirex import born_weights, kappa_cmb, ladder_kappa
from LSS_forward_model.cosmology import read_sims_params, recover_shell_info
from LSS_forward_model.maps import build_shells

# KCMB_SIM, KCMB_WORK, KCMB_BARYON select the simulation (defaults: runs3A run001, no matched baryon fields)
SIM = Path(os.environ.get('KCMB_SIM', '/pscratch/sd/m/mgatti/kcmb_test/run001'))
BARYON = Path(os.environ.get('KCMB_BARYON', '/pscratch/sd/m/mgatti/run001'))   # delta_b_1024_{0,1}.npy (Hypatia)
WORK = Path(os.environ.get('KCMB_WORK', '/pscratch/sd/m/mgatti/kcmb_test/work'))
NSIDE, NSIDE_LO, LMAX = 2048, 1024, 4096
Z_BASE = 2.0
Z_BARYON = 1.0
SEED = 20261106


def setup():
    sims_parameters, cosmo_bundle = read_sims_params(str(SIM))
    shells_info = recover_shell_info(str(SIM/'z_values.txt'), max_z=49)
    shells, steps, zeff = build_shells(shells_info)
    steps = np.asarray(shells_info['Step'])[::-1]            # low redshift first, as build_shells
    return sims_parameters, cosmo_bundle, shells, steps, np.asarray(zeff)


def counts_of(step):
    path = SIM/f'run.{int(step):05d}.lightcone.npy'
    return np.load(path).astype(np.float64) if path.exists() else None


def one_shell(args):
    i, step, z_low = args
    counts = counts_of(step)
    if counts is None:
        return i, None
    nside_native = hp.npix2nside(counts.size)
    c2048 = hp.ud_grade(counts, NSIDE, power=-2)               # sum of the sub-pixels
    np.save(WORK/'delta2048'/f'shell_{i:03d}.npy', (c2048/c2048.mean() - 1).astype(np.float32))
    out = None
    if z_low < Z_BARYON + 0.15:
        # the harmonic downgrade used inside the baryonification (maps.make_tsz_and_baryonified_density)
        pw = hp.pixwin(nside_native)
        alm = hp.almxfl(hp.map2alm(counts, lmax=NSIDE_LO*2), 1/pw[:NSIDE_LO*2 + 1])
        c1024 = hp.alm2map(alm, nside=NSIDE_LO, pixwin=True)*(nside_native/NSIDE_LO)**2
        d_dmo = c1024/c1024.mean() - 1
        out = {}
        for rel in (0, 1):
            d_b = np.load(BARYON/f'delta_b_1024_{rel}.npy', mmap_mode='r')[i].astype(np.float64)
            diff = d_b - d_dmo
            out[rel] = float(np.std(diff))
            # upgrade to NSIDE 2048: pixel window of 1024 -> 2048, band limit 2 * NSIDE_LO
            a = hp.map2alm(diff, lmax=2*NSIDE_LO)
            ratio = hp.pixwin(NSIDE, lmax=2*NSIDE_LO)/hp.pixwin(NSIDE_LO, lmax=2*NSIDE_LO)
            up = hp.alm2map(hp.almxfl(a, ratio), NSIDE, lmax=2*NSIDE_LO)
            np.save(WORK/f'dbaryon{rel}'/f'shell_{i:03d}.npy', up.astype(np.float32))
    return i, out


def run_shells(nproc=12):
    for d in ('delta2048', 'dbaryon0', 'dbaryon1'):
        (WORK/d).mkdir(parents=True, exist_ok=True)
    _, _, shells, steps, zeff = setup()
    jobs = [(i, steps[i], float(np.min(shells[i].za))) for i in range(len(shells))]
    check = {}
    with Pool(nproc) as pool:
        for i, out in pool.imap_unordered(one_shell, jobs):
            print(f'shell {i} (z={zeff[i]:.3f}) done' + ('' if out is None else f', std(delta_b - delta_dmo) {out}'),
                  flush=True)
            if out is not None:
                check[i] = dict(zeff=float(zeff[i]), std=out)
    (WORK/'baryon_check.json').write_text(json.dumps(check, indent=1) + '\n')


def available(shells):
    return [i for i in range(len(shells)) if (WORK/'delta2048'/f'shell_{i:03d}.npy').exists()]


def shell_stream(index, shells, folder='delta2048'):
    for i in index:
        path = WORK/folder/f'shell_{i:03d}.npy'
        yield (np.load(path) if path.exists() else np.zeros(hp.nside2npix(NSIDE), np.float32)), shells[i]


def weyl_limber(pars, chi_cut_mpc, lmax=LMAX, nz=600):
    """C_kk beyond comoving distance chi_cut (Mpc) to the CMB, Limber with the nonlinear Weyl power (CAMB)."""
    p = copy.deepcopy(pars)
    p.NonLinear = camb.model.NonLinear_both
    results = camb.get_results(p)
    chistar = results.conformal_time(0) - results.tau_maxvis
    chis = np.linspace(chi_cut_mpc, chistar, nz)
    zs = results.redshift_at_comoving_radial_distance(chis)
    dchi = np.gradient(chis)
    pk = camb.get_matter_power_interpolator(p, nonlinear=True, hubble_units=False, k_hunit=False, kmax=50.,
                                            var1=camb.model.Transfer_Weyl, var2=camb.model.Transfer_Weyl,
                                            zmax=float(zs[-1]))
    win = ((chistar - chis)/(chis**2*chistar))**2
    ell = np.arange(lmax + 1, dtype=float)
    cl = np.zeros(lmax + 1)
    for l in range(2, lmax + 1):
        k = (l + 0.5)/chis
        w = (k > 1e-4) & (k < 50.)
        cl[l] = np.sum(dchi[w]*pk.P(zs[w], k[w], grid=False)*win[w]/k[w]**4)
    return cl*(ell*(ell + 1))**2


def gaussian_remainder(pars, z_cut, seed):
    results = camb.get_background(pars)
    cl = weyl_limber(pars, results.comoving_radial_distance(z_cut))
    rng_state = np.random.get_state()
    np.random.seed(seed)
    alm = hp.synalm(cl, lmax=LMAX, new=True)
    np.random.set_state(rng_state)
    return hp.alm2map(hp.almxfl(alm, hp.pixwin(NSIDE, lmax=LMAX)), NSIDE, lmax=LMAX)


def run_kappa(version):
    sims_parameters, cosmo_bundle, shells, steps, zeff = setup()
    pars = cosmo_bundle['pars_camb']
    have = available(shells)
    base = [i for i in have if np.max(shells[i].za) <= Z_BASE + 1e-6]
    t0 = time.time()
    if version == 'A':
        kappa = kappa_cmb(shell_stream(have, shells), pars)
        kappa += gaussian_remainder(pars, float(np.max(shells[have[-1]].za)), SEED)
    elif version == 'C':
        kappa = kappa_cmb(shell_stream(base, shells), pars)
        kappa += gaussian_remainder(pars, float(np.max(shells[base[-1]].za)), SEED)
    elif version in ('L', 'Lonly', 'H', 'Honly'):
        # L: native z < 2 + independent lognormal shells (fast); H: ell <= 100 conditioned (exact CAMB spectra)
        results = camb.get_background(pars)
        chi = np.array([results.comoving_radial_distance(shells[i].zeff) for i in base])*pars.H0/100.
        last = [i for i in base if chi[-1] - chi[base.index(i)] < 400.]
        hybrid = version.startswith('H')
        kappa = ladder_kappa(pars, float(np.max(shells[base[-1]].za)), NSIDE, seed=SEED, l_exact=100 if hybrid else 0,
                             last_shells=[np.load(WORK/'delta2048'/f'shell_{i:03d}.npy') for i in last] if hybrid else None,
                             last_windows=[shells[i] for i in last] if hybrid else None)
        print(f'  ladder done ({time.time() - t0:.0f}s)', flush=True)
        if not version.endswith('only'):
            q = born_weights([shells[i] for i in base], pars)
            for i, w in zip(base, q):
                kappa += w*np.load(WORK/'delta2048'/f'shell_{i:03d}.npy')
    elif version == 'lo':                       # native z < 2 only (no remainder)
        kappa = kappa_cmb(shell_stream(base, shells), pars)
    elif version == 'mid':                      # native 2 < z < end of lightcone only
        kappa = kappa_cmb(shell_stream([i for i in have if i not in base], shells), pars)
    elif version in ('b0', 'b1'):
        low = [i for i in have if np.min(shells[i].za) < Z_BARYON + 0.15]
        kappa = kappa_cmb(shell_stream(low, shells, f'dbaryon{version[1]}'), pars)
    else:
        raise ValueError(version)
    np.save(WORK/f'kappa_{version}.npy', kappa.astype(np.float32))
    print(f'kappa {version} done in {time.time() - t0:.0f}s', flush=True)


def run_theory():
    sims_parameters, cosmo_bundle, shells, steps, zeff = setup()
    pars = cosmo_bundle['pars_camb']
    have = available(shells)
    out = dict(full=weyl_limber(pars, 1.0),
               beyond_A=weyl_limber(pars, camb.get_background(pars).comoving_radial_distance(float(np.max(shells[have[-1]].za)))),
               beyond_2=weyl_limber(pars, camb.get_background(pars).comoving_radial_distance(Z_BASE)),
               pixwin=hp.pixwin(NSIDE, lmax=LMAX))
    np.savez(WORK/'theory.npz', **out)


def run_stats():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    th = np.load(WORK/'theory.npz')
    ell = np.arange(LMAX + 1)
    edges = np.unique(np.rint(np.geomspace(10, LMAX, 30)).astype(int))
    centre = np.sqrt(edges[:-1]*edges[1:])
    bp = lambda cl: np.array([cl[lo:hi].mean() for lo, hi in zip(edges[:-1], edges[1:])])
    target = th['full']*th['pixwin']**2
    maps = {v: np.load(WORK/f'kappa_{v}.npy').astype(np.float64) for v in ('A', 'B', 'L', 'H', 'C', 'b0', 'b1')
            if (WORK/f'kappa_{v}.npy').exists()}
    alms = {v: hp.map2alm(m, lmax=LMAX) for v, m in maps.items()}
    out = {'ell': centre.tolist()}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for v in ('A', 'B', 'L', 'H', 'C'):
        if v not in alms:
            continue
        ratio = bp(hp.alm2cl(alms[v]))/bp(target)
        out[f'{v}/theory'] = ratio.tolist()
        ax[0].plot(centre, ratio, label=v)
        for rel in ('b0', 'b1'):
            if rel in alms:
                with_b = bp(hp.alm2cl(alms[v] + alms[rel]))/bp(hp.alm2cl(alms[v]))
                out[f'{v}+{rel}/{v}'] = with_b.tolist()
                if v == 'A':
                    ax[1].plot(centre, with_b, label=f'A + baryons ({rel})')
    # partial maps against their own theory windows (Limber pieces add: no cross term)
    parts = {'lo': th['full'] - th['beyond_2'], 'mid': th['beyond_2'] - th['beyond_A'], 'Lonly': th['beyond_2'], 'Honly': th['beyond_2']}
    for v, cl_theory in parts.items():
        if (WORK/f'kappa_{v}.npy').exists():
            m = np.load(WORK/f'kappa_{v}.npy').astype(np.float64)
            out[f'{v}/theory'] = (bp(hp.anafast(m, lmax=LMAX))/bp(cl_theory*th['pixwin']**2)).tolist()
    for a, b in (('A', 'B'), ('A', 'C'), ('B', 'C'), ('A', 'L'), ('B', 'L'), ('B', 'H'), ('L', 'H')):
        if a in alms and b in alms:
            r = bp(hp.alm2cl(alms[a], alms[b]))/np.sqrt(bp(hp.alm2cl(alms[a]))*bp(hp.alm2cl(alms[b])))
            out[f'r({a},{b})'] = r.tolist()
            ax[2].plot(centre, r, label=f'r({a},{b})')
    for v, m in maps.items():
        if v in ('A', 'B', 'L', 'H', 'C'):
            sm = hp.smoothing(m, fwhm=np.radians(5/60.), lmax=LMAX)
            s = sm.std()
            out[f'{v} pdf moments'] = dict(std=float(s), skew=float(np.mean((sm - sm.mean())**3)/s**3),
                                          kurt=float(np.mean((sm - sm.mean())**4)/s**4 - 3))
    for a_, t in zip(ax, ('C_kk / (HMcode x pixwin^2)', 'with baryons / without', 'cross-correlation coefficient')):
        a_.set_xscale('log'); a_.set_title(t); a_.legend(); a_.axhline(1, color='k', lw=0.5)
    fig.tight_layout()
    fig.savefig(WORK/'kcmb_test.png', dpi=110)
    (WORK/'stats.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if 'moments' in k}, indent=1))


if __name__ == '__main__':
    step = sys.argv[1]
    WORK.mkdir(parents=True, exist_ok=True)
    if step == 'shells':
        run_shells(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    elif step == 'kappa':
        run_kappa(sys.argv[2])
    elif step == 'theory':
        run_theory()
    else:
        run_stats()
