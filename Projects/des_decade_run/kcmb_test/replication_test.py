#!/usr/bin/env python3
"""Does box replication spoil the sample variance of the high-z CMB lensing contribution? (runs3A run001)

Maps (NSIDE 2048): kappa_mid = native shells 2 < z < 20 (replicated boxes), kappa_lo = native z < 2, and
Gaussian realisations with the same measured C_ell (independent modes by construction).

1. rotation correlation: r(ell) = C(kappa, R kappa) / C(kappa, kappa) for the mock rotations R of
   maps.rotate_and_rebin (rot 1: 180 deg, rot 2: 90 deg + mirror, rot 3: 270 deg + mirror); for an independent
   field r scatters around 0 with sigma ~ 1/sqrt(modes in the band).
2. patch sample variance: 48 equal-area patches (NSIDE-2 pixels); band powers of each patch (masked anafast /
   patch area); the scatter across patches and the mean correlation between different patches, compared with
   the same numbers for N_GAUSS Gaussian realisations.

output: kcmb_test/out_replication.json
"""
import json
import sys
from pathlib import Path

import healpy as hp
import numpy as np

sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
from LSS_forward_model.maps import convert_to_pix_coord, rotate_and_rebin

WORK = Path('/pscratch/sd/m/mgatti/kcmb_test/work')
NSIDE, LMAX = 2048, 2000
N_GAUSS = 4
EDGES = np.array([30, 60, 100, 160, 250, 400, 650, 1000, 1500, 2000])


def bandpowers(cl):
    return np.array([cl[lo:hi].mean() for lo, hi in zip(EDGES[:-1], EDGES[1:])])


def rotated(m, rot):
    """Map value seen in the rotated frame, with the same pixel mapping as the mock footprints."""
    npix = hp.nside2npix(NSIDE)
    theta, phi = hp.pix2ang(NSIDE, np.arange(npix))
    ra, dec = np.degrees(phi), 90 - np.degrees(theta)
    pix = rotate_and_rebin(convert_to_pix_coord(ra, dec, nside=2*NSIDE), NSIDE, rot)
    return m[pix]


def patch_stats(m):
    patch = hp.ang2pix(2, *hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE))))
    bp = []
    for p in range(48):
        mask = (patch == p).astype(float)
        bp.append(bandpowers(hp.anafast(m*mask, lmax=LMAX))/mask.mean())
    bp = np.array(bp)                                         # (48, nband)
    scatter = bp.std(0)/bp.mean(0)
    z = (bp - bp.mean(0))/bp.std(0)
    corr = (z.T @ z)/len(z)                                   # band-band across patches (diagnostic)
    patch_corr = np.corrcoef(z)                               # patch-patch over bands
    off = patch_corr[~np.eye(48, dtype=bool)].mean()
    return scatter, off


def main():
    out = {'bands': EDGES.tolist()}
    for name in (sys.argv[1:] or ['mid', 'lo']):
        m = np.load(WORK/f'kappa_{name}.npy').astype(np.float64)
        alm = hp.map2alm(m, lmax=LMAX)
        cl = hp.alm2cl(alm)
        entry = {}
        for rot in (1, 2, 3):
            r = bandpowers(hp.anafast(m, rotated(m, rot), lmax=LMAX))/bandpowers(cl)
            entry[f'r_rot{rot}'] = r.tolist()
        modes = np.array([np.sum(2*np.arange(lo, hi) + 1) for lo, hi in zip(EDGES[:-1], EDGES[1:])])
        entry['r_expected_sigma'] = (1/np.sqrt(modes)).tolist()
        entry['patch_scatter'], entry['patch_offdiag_corr'] = [x.tolist() if hasattr(x, 'tolist') else float(x)
                                                               for x in patch_stats(m)]
        gauss_scatter, gauss_off, gauss_r = [], [], []
        for seed in range(N_GAUSS):
            np.random.seed(100 + seed)
            g = hp.synfast(cl, NSIDE, lmax=LMAX, new=True)
            s, o = patch_stats(g)
            gauss_scatter.append(s); gauss_off.append(o)
            gauss_r.append(bandpowers(hp.anafast(g, rotated(g, 2), lmax=LMAX))/bandpowers(hp.anafast(g, lmax=LMAX)))
        entry['gauss_patch_scatter_mean'] = np.mean(gauss_scatter, 0).tolist()
        entry['gauss_patch_scatter_std'] = np.std(gauss_scatter, 0).tolist()
        entry['gauss_patch_offdiag_corr'] = [float(x) for x in gauss_off]
        entry['gauss_r_rot2'] = np.array(gauss_r).tolist()
        out[name] = entry
        print(name, 'rot r:', {k: np.round(v, 3).tolist() for k, v in entry.items() if k.startswith('r_rot')}, flush=True)
        print(name, 'patch scatter native / gaussian:',
              np.round(np.array(entry['patch_scatter'])/np.array(entry['gauss_patch_scatter_mean']), 3).tolist(),
              ' (gaussian spread', np.round(np.array(entry['gauss_patch_scatter_std'])/np.array(entry['gauss_patch_scatter_mean']), 3).tolist(), ')',
              'patch-patch corr native', round(entry['patch_offdiag_corr'], 4), 'gaussian', np.round(gauss_off, 4).tolist(), flush=True)
    (Path(__file__).resolve().parent/f"out_replication_{'_'.join(sys.argv[1:]) or 'native'}.json").write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
