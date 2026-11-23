#!/usr/bin/env python3
"""make_WL_sample (footprint-restricted shell sums) and make_halo_IA_numerator (single bincount) vs the previous
code, on synthetic NSIDE-64 inputs: NLA and halo IA, Poisson SC v2.  usage: test_speedups_equivalence.py OLD_PKG_DIR"""
import copy, sys
import numpy as np, healpy as hp
sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
sys.path.insert(0, sys.argv[1])
import lfm_old.lensing as old_lensing, lfm_old.halo_ia as old_halo
from LSS_forward_model import lensing as new_lensing, halo_ia as new_halo
from LSS_forward_model.cosmology import make_cosmo_bundle, sigma8_to_As_CAMB
from LSS_forward_model.source_clustering import load_SC_calibration

NSIDE, NSHELL = 64, 6
npix = hp.nside2npix(NSIDE)
rng = np.random.default_rng(0)
fields = {'density': rng.normal(0, .3, (NSHELL, npix)),
          'gamma': rng.normal(0, .02, (NSHELL, 1, npix)) + 1j*rng.normal(0, .02, (NSHELL, 1, npix)),
          'IA_shear': rng.normal(0, .5, (NSHELL, 1, npix)) + 1j*rng.normal(0, .5, (NSHELL, 1, npix))}
ngal = rng.uniform(0, 1, (4, NSHELL)); ngal[0, -1] = 0; ngal /= ngal.sum(1, keepdims=True)
zeff = np.linspace(.2, 1.4, NSHELL)
cats = {}
for t in range(4):
    n = 40000
    ra, dec = rng.uniform(0, 90, n), np.degrees(np.arcsin(rng.uniform(-.6, .1, n)))
    cats[t] = dict(ra=ra, dec=dec, e1=rng.normal(0, .25, n), e2=rng.normal(0, .25, n), w=rng.uniform(.5, 1.5, n))
sp = dict(h=.7, Omega_m=.3, Omega_b=.05, Omega_cdm=.25, sigma_8=.8, n_s=.96, m_nu=.06, n_nu=1, w0=-1., wa=0.)
sp['As'] = sigma8_to_As_CAMB(.3, .05, .7, .96, -1., 0., .06, .8)
cb = make_cosmo_bundle(sp)
sp.update(A_IA=.7, eta_IA=.5, A_IA2=0., bias_sc=np.array([1.1, 1.2, 1.3, 1.4]), bias_sc_slope=np.full(4, .5),
          bias_TA=np.zeros(4), dm=np.ones(4), rot=1, delta_rot=0)
sc = load_SC_calibration('/global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/data/SC_v2_DES_nside1024_sim1.json')

# model sources for the halo numerator
nsrc = 300000
objs = dict(shell_index=np.sort(rng.integers(0, NSHELL, nsrc)).astype(np.uint16), ra_deg=rng.uniform(0, 360, nsrc),
            dec_deg=np.degrees(np.arcsin(rng.uniform(-1, 1, nsrc))), redshift=rng.uniform(.1, 1.5, nsrc),
            is_central=rng.random(nsrc) < .6, e1=rng.normal(0, .2, nsrc), e2=rng.normal(0, .2, nsrc),
            hosted_fraction=np.full(nsrc, .9, np.float32), footprint_fraction=np.full(nsrc, .4, np.float32))
ia = {'ia': dict(A_c=1.2, eta_c=-1., A_s=.02, eta_s=.3, b_c=0., b_s=0.)}
for bc in (np.array([4e4, 4e4, 3e4, 2e4]), None):
    h_old = old_halo.make_halo_IA_numerator(objs, ia, fields['density'], ngal, NSIDE, bin_counts=bc, seed=5)
    h_new = new_halo.make_halo_IA_numerator(objs, ia, fields['density'], ngal, NSIDE, bin_counts=bc, seed=5)
    for k in ('numerator', 'numerator_sq'):
        d = np.max(np.abs(h_old[k] - h_new[k]))/np.max(np.abs(h_old[k]))
        print(f'halo numerator {k} (bin_counts {"given" if bc is not None else "None"}): max rel diff {d:.2e}')
        assert d < 1e-12

for model in ('NLA', 'halo'):
    out = []
    for mod in (old_lensing, new_lensing):
        np.random.seed(11)
        m, _ = mod.make_WL_sample(ngal, zeff, cb, copy.deepcopy(sp), NSIDE, fields, cats, SC_corrections=sc,
                                  include_SC=True, compact_savings=False, noise_model='poisson', IA_model=model,
                                  halo_IA=h_new if model == 'halo' else None,
                                  **(dict(ia_cap_alpha=None, ia_noise_nside=64) if mod is new_lensing else {}))  # pre-cap defaults
        out.append(m)
    for t in range(4):
        for k in out[0][t]:
            a, b = out[0][t][k], out[1][t][k]
            assert np.array_equal(a, b), (model, t, k, np.max(np.abs(a - b)))
    print(f'make_WL_sample {model}: all maps bit-identical')
print('OK')
